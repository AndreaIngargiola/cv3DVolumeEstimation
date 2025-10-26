#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/dnn.hpp>
#include <thrust/device_vector.h>
#include <fstream>

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
    std::vector<cv::Rect> getBBOfPeople(cv::Mat& frame);
    
    private:
    cv::Mat letterbox(const cv::Mat& d_src);
    void parseDetections(cv::Mat& d_out, std::vector<cv::Rect>& boxes, std::vector<float>& confidences);

};

struct Detection {
    float top, left, w, h;
    float score;
    int classId;
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

    public:
    Clusterer(YOLOHelper& boxFinder, const float shiftThreshold, const int frameSetSize);
    void clusterize(cv::Mat frame, cv::cuda::GpuMat d_unclusteredKP);
    void inheritClusters(cv::cuda::GpuMat d_statusKP, cv::cuda::GpuMat d_unclusteredKP, cv::cuda::GpuMat d_frame);
    thrust::device_vector<DataPoint> getDatapoints();
    thrust::device_vector<Centroid> getCentroids();
    std::vector<cv::Rect> getBoxes();
    
    float updateCentroids();
    void updateKeypoints();
};

/*
    clusterize(d_unclusteredKP);
        1) boxFinder.getBBOfPeople (result is a vector of cv::Rect on CPU)
        2) d_unclusteredKP -> Keypoints (define classId and isSupervised)
        3) res = updateCentroids(keypoints.isSupervised == true) (create Centroids with lastUpdateDistance = MAX_FLOAT)
        while (res > threshold) {
            updateKeypoints(keypoints.isSupervised = false)
            res = updateCentroids()
        //}

*/