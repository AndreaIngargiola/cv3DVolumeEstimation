#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/dnn.hpp>
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
    float cx, cy, w, h;
    float score;
    int classId;
};