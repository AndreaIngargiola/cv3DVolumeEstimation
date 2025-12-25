#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/dnn.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vision.hpp>
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
    const cv::Size targetSize;  //the size of YOLO input
    const cv::Size frameSize;   //original frame size
    cv::Size unpaddedSize;      //the size of the original frame rescaled to fit YOLO input size (without padding)
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

struct Cluster {
    cv::Point2f mu;     // center
    float eh, ew;       // ellipse semi-axes (height, width)
    float pi;           // mixture prior
};

class EMClusterer{
    private:
    YOLOHelper boxFinder;
    int N;
    int K;
    int maxIter = 10;
    //thrust::host_vector<cv::Rect> boundingBoxes;
    std::pair<std::vector<cv::Rect>, std::vector<cv::Rect>> boundingBoxes;
    thrust::host_vector<Cluster> ellipses;
    cv::cuda::GpuMat kp;                    // clustering input, output of KPExtractor
    std::vector<cv::Point2f> mu;            // cluster centers
    std::vector<cv::Point2f> oldMu;         // previous iteration cluster center to compute distance
    std::vector<float> ew;                  // ellipse semi-axis width
    std::vector<float> eh;                  // ellipse semi-axis height
    std::vector<float> pi;                  // priors
    std::vector<cv::Point2f> pts;           // refined KPExtractor output in the correct reference frame and filtered out of non-people kp
    std::vector<std::vector<float>> r;       // responsability matrix N x K
    int img_w, img_h;
    cv::Mat halfPersonPlane;
    int halfPersonZ;
    cv::Mat P;

    std::vector<DataPoint> datapoints;
    
    const float SIGMA2 = 100.0f;
    const float INV_2PI_SIGMA2 = 1.0f / (2.0f * CV_PI * SIGMA2);
    const double DELTA = 1.0;   // 0.5–1.0 px threshold

    // m = fraction of Gaussian mass inside “30% ellipse”
    const float m = 0.773f;
   
    // c = sqrt( -1 / (2 log(1 - m)) )  (formula 9 of the paper)
    const float c = std::sqrt(-1.0f / (2.0f * std::log(1.0f - m)));

    public: 
    EMClusterer(YOLOHelper yh, cv::Mat halfPersonPlane, int halfPersonZ, cv::Mat P);
    std::pair<thrust::host_vector<Cluster>, std::vector<DataPoint>> clusterizeKeyPoints(cv::cuda::GpuMat keypoints, cv::Mat frame);
    std::pair<thrust::host_vector<Cluster>, std::vector<DataPoint>> clusterizeDataPoints(std::vector<DataPoint> datapoints, int k);
    
    private:
    void preClusteringCleanUp();
    void importDataPoints(std::vector<DataPoint> dp);
    void importKeyPoints();
    void initializeGaussians();
    void runExpectationMaximization();
    void EStep();
    void MStep();
    void postProcessResults();
};