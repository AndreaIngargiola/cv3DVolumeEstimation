#include <gtest/gtest.h>
#include <customMath.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vision.hpp>
#include <filesystem>
#include <fstream>
#include <clustering.hpp>

TEST(ClusteringTest, YOLOtest) {
    using namespace cv;
    using namespace cv::dnn;
    using namespace std;

    std::string modelPath = "../../data/yolov5s.onnx";
    std::string classFile = "../../data/coco.names";
    std::string imagePath = "../../data/test_YOLO.jpg"; // change to your image

    // Load image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Cannot load image: " << imagePath << std::endl;
    }

    YOLOHelper yh = YOLOHelper(modelPath, image.size(), cv::Size(640,640), 0.5f, 0.45f);
    
    Mat frameToShow;
    image.copyTo(frameToShow);

    std::vector<cv::Rect> boundingBoxes = yh.getBBOfPeople(image);
    
    // Save results in an image
    cv::RNG rng(cv::getTickCount());
    cv::Scalar brightColor(
        50 + rng.uniform(0, 206),
        50 + rng.uniform(0, 206),
        50 + rng.uniform(0, 206)
    );

    for (cv::Rect box : boundingBoxes) {
        rectangle(frameToShow, box, brightColor, 2);
    }

    cv::imwrite("testYOLO.png", frameToShow);
}

