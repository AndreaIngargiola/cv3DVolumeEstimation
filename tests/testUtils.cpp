#include <gtest/gtest.h>
#include <customMath.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vision.hpp>
#include <filesystem>

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