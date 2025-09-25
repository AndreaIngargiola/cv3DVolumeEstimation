#include <gtest/gtest.h>
#include <customMath.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vision.hpp>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

TEST(CustomMathTest, InvertAxis) {
    cv::Vec2i p(2, 3);
    auto result = customMath::invert2dAxisI(p, 10, 10);
    EXPECT_EQ(result[0], 8);
    EXPECT_EQ(result[1], 7);
}

TEST(VisionTest, Calibration){
    Calibrator c("../../data/calibration/values.yml", "../../data/calibration", cv::Size(9,6), 1);
    // Print results
    std::cout << "RMS error = " << c.getReprojectionError() << std::endl;
    std::cout << "Camera matrix K =\n" << c.getK() << std::endl;
    std::cout << "Distortion coefficients =\n" << c.getDistCoeffs() << std::endl;
}

TEST(VisionTest, Homography) {
    using namespace cv;
    using namespace std;

    Point3f origin = {0,0,0};
    float size = 4.0f;
    string posePath = "../../data/floor_surface/piano_pav_(1).jpg";
    Mat img = imread(posePath);

    // Define Calibrator and Homographer (tests subjects)
    Calibrator c("../../data/calibration/values.yml", "../../data/calibration", cv::Size(9,6), 1);
    Homographer hom(c, 2, posePath);
    Mat H = hom.getGoundPlane();
    Mat P = hom.getP();

    // Define cube corners relative to origin and size
    vector<Point3d> cube3D = {
        {origin.x, origin.y, origin.z},
        {origin.x + size, origin.y, origin.z},
        {origin.x + size, origin.y + size, origin.z},
        {origin.x, origin.y + size, origin.z},
        {origin.x, origin.y, origin.z + size},
        {origin.x + size, origin.y, origin.z + size},
        {origin.x + size, origin.y + size, origin.z + size},
        {origin.x, origin.y + size, origin.z + size}
    };

    // Convert 3D points to 2D using inverse homography
    vector<Point2d> projectedPoints;

    for (int i = 0; i < 4; i++) {
        projectedPoints.push_back(customMath::projectOnImgFromPlane(cube3D[i], H));
    }

    for (int i = 4; i < 8; i++) {
        projectedPoints.push_back(customMath::projectOnImgFrom3D(cube3D[i], P));
    }
    
    vector<Point2i> toDraw;
    for(Point2d p : projectedPoints){
        toDraw.push_back(Point2i(round(img.size().width - p.x), round(img.size().height - p.y)));
    }

    std::cout << "pojectedPoints = " << projectedPoints << std::endl;
    // Define cube edges grouped by type
    vector<pair<int,int>> bottomEdges = {{0,1},{1,2},{2,3},{3,0}};
    vector<pair<int,int>> topEdges    = {{4,5},{5,6},{6,7},{7,4}};
    vector<pair<int,int>> verticalEdges = {{0,4},{1,5},{2,6},{3,7}};

    // Draw edges with different colors
    for (auto e : bottomEdges) line(img, toDraw[e.first], toDraw[e.second], Scalar(0,0,255), 2); // red
    for (auto e : topEdges)    line(img, toDraw[e.first], toDraw[e.second], Scalar(0,255,0), 2); // green
    for (auto e : verticalEdges) line(img, toDraw[e.first], toDraw[e.second], Scalar(255,0,0), 2); // blue

    imwrite("testHomOutput.png", img);
}

TEST(VisionTest, ParallelPlanes) {
    using namespace cv;
    using namespace std;

    Point3f origin = {0,0,0};
    float size = 4.0f;
    string posePath = "../../data/floor_surface/piano_pav_(1).jpg";
    Mat img = imread(posePath);

    vector<Point2i> toDraw;

    // Define Calibrator and Homographer (tests subjects)
    Calibrator c("../../data/calibration/values.yml", "../../data/calibration", cv::Size(9,6), 1);
    Homographer hom(c, 2, posePath);
    Mat P = hom.getP();

    //defiine control group points
    vector<Point3d> withP = {Point3d(1,0,0), Point3d(1,0,10), Point3d(1,0,20), Point3d(1,0,30)};
    for (const Point3d controlPoint : withP) {
        Point2d p = customMath::projectOnImgFrom3D(controlPoint, P);
        toDraw.push_back(Point2i(round(img.size().width - p.x), round(img.size().height - p.y)));
    }

    //compute test group homographies and their respective origin on the image
    int numberOfPlanes = hom.computeHomographies(10, 60);
    
    for (const auto& entry : hom.getHomographies()) {
        Point2d p = customMath::projectOnImgFromPlane(origin, entry.second);
        toDraw.push_back(Point2i(round(img.size().width - p.x), round(img.size().height - p.y)));
    }

    for (int i=0; i < 8; i++) {
        if(i < 4) circle(img, toDraw[i], 2, Scalar(0,0,255), 2); // red
        else circle(img, toDraw[i], 2, Scalar(0,0,0), 2);        // black
    }
    
    imwrite("testPP.png", img);
}

cv::Mat letterbox(const cv::Mat& src, int target_width, int target_height,
                float& scale, int& pad_x, int& pad_y) {
    float r = std::min(target_width / (float)src.cols,
                    target_height / (float)src.rows);
    int new_w = int(round(src.cols * r));
    int new_h = int(round(src.rows * r));
    scale = r;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    pad_x = (target_width - new_w) / 2;
    pad_y = (target_height - new_h) / 2;

    cv::Mat dst;
    cv::copyMakeBorder(resized, dst,
                    pad_y, target_height - new_h - pad_y,
                    pad_x, target_width - new_w - pad_x,
                    cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return dst;
}

TEST(VisionTest, YOLOtest) {
    using namespace cv;
    using namespace cv::dnn;
    using namespace std;

    std::string modelPath = "../../data/yolov5s.onnx";
    std::string classFile = "../../data/coco.names";
    std::string imagePath = "../../data/test_YOLO.jpg"; // change to your image

    // Load class names
    std::vector<std::string> classNames;
    std::ifstream ifs(classFile);
    std::string line;
    while (std::getline(ifs, line)) classNames.push_back(line);

    // Load image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Cannot load image: " << imagePath << std::endl;
    }

    // Load YOLO ONNX model
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    // Set backend to CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    float scale;
    int pad_x, pad_y;
    cv::Mat blob_input = letterbox(image, 640, 640, scale, pad_x, pad_y);
    // Preprocess image with letterbox crop
    Size inputSize(640, 640);
    Mat blob;
    blobFromImage(blob_input, blob, 1/255.0, inputSize, Scalar(), true, false); // crop=true
    net.setInput(blob);

    // Forward pass
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Reshape output to [numDetections x 85]
    Mat outMat = outputs[0].reshape(1, outputs[0].size[1]);
    int dimensions = outMat.cols;   // 85
    int numClasses = dimensions - 5;

    // Letterbox parameters
    int orig_w = image.cols;
    int orig_h = image.rows;

    // Detection containers
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    float confThreshold = 0.5f;
    float nmsThreshold = 0.45f;

    // Parse detections
    for (int i = 0; i < outMat.rows; i++) {
        float objectness = outMat.at<float>(i, 4);
        if (objectness < confThreshold) continue;

        Mat scores = outMat.row(i).colRange(5, dimensions);
        Point classIdPoint;
        double maxClassScore;
        minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

        float confidence = objectness * (float)maxClassScore;
        if (confidence > confThreshold) {
            // YOLO box (normalized)
            float cx = outMat.at<float>(i, 0);// * inputSize.width;
            float cy = outMat.at<float>(i, 1);// * inputSize.height;
            float bw = outMat.at<float>(i, 2);// * inputSize.width;
            float bh = outMat.at<float>(i, 3);// * inputSize.height;

            // Undo letterbox: remove padding and scale back to original image
            float x = (cx - pad_x) / scale;
            float y = (cy - pad_y) / scale;
            float width  = bw / scale;
            float height = bh / scale;

            int left = int(x - width / 2);
            int top  = int(y - height / 2);

            // Clip to image bounds
            left = max(0, left);
            top = max(0, top);
            width = min(width, (float)(orig_w - left));
            height = min(height, (float)(orig_h - top));

            classIds.push_back(classIdPoint.x);
            confidences.push_back(confidence);
            boxes.push_back(Rect(left, top, (int)width, (int)height));
        }
    }

    // Non-Maximum Suppression
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    cout << "classNames " << classNames[0] << endl; 
    // Draw detections
    for (int idx : indices) {
        Rect box = boxes[idx];
        int classId = classIds[idx];
        string label = (classId < classNames.size()) ? classNames[classId] : to_string(classId);
        float conf = confidences[idx];
        cout << "box " << box << " classId: " << classId << endl; 
        rectangle(image, box, Scalar(0, 0, 0), 2);
        putText(image, format("%s: %.2f", label.c_str(), conf),
                Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(0, 0, 0), 2);
    }

    cv::imwrite("testYOLO.png", image);
}