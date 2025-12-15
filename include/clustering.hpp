#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/dnn.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <fstream>
#include <utility> 

struct Detection {
    float top = -1;
    float left = -1;
    float w = -1;
    float h = -1;
    float score = -1;
    int classId = -1;
};

class YOLOHelper {
    private:
    cv::dnn::Net net;
    const cv::Size targetSize; //the size of YOLO input
    const cv::Size frameSize; //original frame size
    cv::Size unpaddedSize; //the size of the original frame rescaled to fit YOLO input size (without padding)
    float scale;
    int pad_x;
    int pad_y;
    const float confThreshold;
    const float nmsThreshold;
    
    public:
    YOLOHelper(const std::string modelPath, 
        const cv::Size frameSize, 
        const cv::Size targetSize, 
        const float confThreshold,  
        const float nmsThreshold);
    std::pair<std::vector<cv::Rect>, std::vector<cv::Rect>>  getBBOfPeople(cv::Mat& frame);
    
    private:
    cv::Mat letterbox(const cv::Mat& d_src);
    //void parseDetections(cv::Mat& d_out, std::vector<cv::Rect>& boxes, std::vector<float>& confidences);
    std::vector<Detection> parseDetections(const std::vector<cv::Mat>& outputs,
                                       int inputWidth, int inputHeight,
                                       float confThreshold,
                                       float nmsThreshold);
};


struct Centroid {
    int classId;
    float x, y, h, s, v;
};

struct DataPoint {
    float features[5];
    int classId = -1;
    int isSupervised = 0;
};

class Clusterer {
    private:
    YOLOHelper boxFinder;
    int k;
    thrust::device_vector<Centroid> d_centroids;
    thrust::device_vector<Centroid> d_oldCentroids;
    thrust::device_vector<DataPoint> d_keypoints;
    const int frameSetSize;
    const float shiftThreshold;
    bool isFirstIteration = true;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Rect> boxesHeads;

    public:
    Clusterer(YOLOHelper& boxFinder, const float shiftThreshold, const int frameSetSize);
    void clusterize(cv::Mat frame, cv::cuda::GpuMat d_unclusteredKP);
    void inheritClusters(cv::cuda::GpuMat d_statusKP, cv::cuda::GpuMat d_unclusteredKP, cv::cuda::GpuMat d_frame);
    thrust::device_vector<DataPoint> getDatapoints();
    thrust::device_vector<Centroid> getCentroids();
    std::pair<std::vector<cv::Rect>, std::vector<cv::Rect>> getBoxes();
    
    float updateCentroids();
    void updateKeypoints();
};


struct Cluster {
    cv::Point2f mu;     // center
    float eh, ew;       // ellipse semi-axes (height, width)
    float pi;           // mixture prior
};


class EMClusterer{
    private:
    YOLOHelper boxFinder;
    int k;
    int maxIter = 10;
    thrust::host_vector<cv::Rect> boundingBoxes;
    thrust::host_vector<Centroid> ellipses;
    cv::cuda::GpuMat kp;
    std::vector<cv::Point2f> mu;            // cluster centers
    std::vector<cv::Point2f> oldMu;         // previous iteration cluster center to compute distance
    std::vector<float> ew;                  // ellipse semi-axis width
    std::vector<float> eh;                  // ellipse semi-axis height
    std::vector<float> pi;                  // priors
    std::vector<cv::Point2f> pts;
    int img_w, img_h;
    cv::Mat halfPersonPlane;
    int halfPersonZ;
    cv::Mat P;
    
    const float SIGMA2 = 100.0f;
    const float INV_2PI_SIGMA2 = 1.0f / (2.0f * CV_PI * SIGMA2);
    const double DELTA = 1.0;   // 0.5–1.0 px threshold

    // m = fraction of Gaussian mass inside “30% ellipse”
    const float m = 0.773f;
   
    // c = sqrt( -1 / (2 log(1 - m)) )  (formula 9 of the paper)
    const float c = std::sqrt(-1.0f / (2.0f * std::log(1.0f - m)));

    public: 
    EMClusterer(YOLOHelper yh, cv::Mat halfPersonPlane, int halfPersonZ, cv::Mat P);
    thrust::host_vector<Cluster> clusterize(cv::cuda::GpuMat keypoints, cv::Mat frame);
    
    private:
    void initializeGaussians();
    void EStep(std::vector<std::vector<float>>& r, int N, int K);
    void MStep(std::vector<std::vector<float>>& r, int N, int K);

};

/*
void EM_step_paperModel(
    const std::vector<cv::Point2f>& pts,
    float m,                                 // = 0.773 from paper
    std::vector<cv::Point2f>& mu,            // cluster centers
    std::vector<float>& ew,                  // ellipse semi-axis width
    std::vector<float>& eh,                  // ellipse semi-axis height
    std::vector<float>& pi)                  // priors
{
    const int N = pts.size();
    const int K = mu.size();

    // responsibilities r[n][k]
    std::vector<std::vector<float>> r(N, std::vector<float>(K));

    // ---------------------------------------------------------------
    //  CONSTANT c FROM FORMULA (9)
    //  c = sqrt( -1 / (2 log(1 - m)) )  ---- exactly from the paper
    // ---------------------------------------------------------------
    float c = std::sqrt(-1.0f / (2.0f * std::log(1.0f - m)));

    // ======================================================================
    //  E-STEP
    // ======================================================================
    for (int n = 0; n < N; ++n)
    {
        float denom = 0.0f;

        for (int k = 0; k < K; ++k)
        {
            // -----------------------------------------------------------------
            // Compute ellipse area A_e  (= π * eh * ew)
            // -----------------------------------------------------------------
            float Ae = CV_PI * ew[k] * eh[k];

            // -----------------------------------------------------------------
            // Compute Σ from c * ellipse axes:
            //      σx = c * ew[k]
            //      σy = c * eh[k]
            // -----------------------------------------------------------------
            float sx = c * ew[k];
            float sy = c * eh[k];

            float invNorm = 1.0f / (2.0f * CV_PI * sx * sy);

            // -----------------------------------------------------------------
            // Check if point is inside ellipse
            // -----------------------------------------------------------------
            float dx = pts[n].x - mu[k].x;
            float dy = pts[n].y - mu[k].y;

            bool inside =
                (dx * dx) / (ew[k] * ew[k]) +
                (dy * dy) / (eh[k] * eh[k]) <= 1.0f;

            // -----------------------------------------------------------------
            //  CASE 1: inside ellipse → uniform: m / Ae
            //  CASE 2: outside ellipse → full Gaussian
            //
            //  (EXACTLY as written in Equation (8))
            // -----------------------------------------------------------------

            float h;
            if (inside)
            {
                h = m / Ae;
            }
            else
            {
                float ex = -0.5f * (dx * dx) / (sx * sx)
                           -0.5f * (dy * dy) / (sy * sy);
                h = std::exp(ex) * invNorm;
            }

            float val = pi[k] * h;
            r[n][k] = val;
            denom += val;
        }

        // normalize responsibilities
        if (denom > 0)
        {
            for (int k = 0; k < K; ++k)
                r[n][k] /= denom;
        }
    }

    // ======================================================================
    //  M-STEP  (update pi[k] and mu[k])
    //          (ellipse sizes ew[], eh[] DO NOT CHANGE in paper’s EM)
    // ======================================================================
    for (int k = 0; k < K; ++k)
    {
        // Nk = effective cluster population
        float Nk = 0.0f;

        for (int n = 0; n < N; ++n)
            Nk += r[n][k];

        // update prior
        pi[k] = Nk / float(N);

        // update center μ_k (the only parameter that moves)
        float sumx = 0.0f, sumy = 0.0f;

        for (int n = 0; n < N; ++n)
        {
            sumx += r[n][k] * pts[n].x;
            sumy += r[n][k] * pts[n].y;
        }

        if (Nk > 1e-6f)
        {
            mu[k].x = sumx / Nk;
            mu[k].y = sumy / Nk;
        }
    }
}

*/