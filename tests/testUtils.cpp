#include <gtest/gtest.h>
#include <customMath.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vision.hpp>
#include <filesystem>
#include <fstream>
#include <segmentation.hpp>
#include <clustering.hpp>

namespace fs = std::filesystem;

TEST(CustomMathTest, InvertAxis) {
    cv::Vec2i p(2, 3);
    auto result = customMath::invert2dAxisI(p, 10, 10);
    EXPECT_EQ(result[0], 8);
    EXPECT_EQ(result[1], 7);
}

TEST(VisionTest, Calibration){
    Calibrator c("visionTest/values.json", cv::Size(9,6), 1, 0, "../../data/calibration_pics", "../../data/floor_surface/piano_pav_(1).jpg");
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
    Calibrator c("visionTest/values.json", cv::Size(9,6), 1, 0, "../../data/calibration_pics", "../../data/floor_surface/piano_pav_(1).jpg");
    Mat K = c.getK();
    Mat R = c.getR();
    
    customMath::flipImageOrigin(K, img.size().width, img.size().height);
    customMath::flipZaxis(R);

    Homographer hom(
        K, R, c.gett(), 
        2, 1);
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
        projectedPoints.push_back(customMath::projectOnImgFromPlane(Point2d(cube3D[i].x, cube3D[i].y), H));
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

    imwrite("visionTest/testHomography.png", img);
}

TEST(VisionTest, ParallelPlanes) {
    using namespace cv;
    using namespace std;

    Point3f origin = {5,0,0};
    float size = 4.0f;
    string posePath = "../../data/floor_surface/piano_pav_(1).jpg";
    Mat img = imread(posePath);

    vector<Point2i> toDraw;

    // Define Calibrator and Homographer (tests subjects)
    Calibrator c("visionTest/values.json", cv::Size(9,6), 1, 0, "../../data/calibration_pics", "../../data/floor_surface/piano_pav_(1).jpg");
    Mat K = c.getK();
    Mat R = c.getR();
    
    customMath::flipImageOrigin(K, img.size().width, img.size().height);
    customMath::flipZaxis(R);

    Homographer hom(
        K, R, c.gett(), 
        2, 1);

    Mat P = hom.getP();

    //define control group points
    vector<Point3d> withP = {Point3d(origin.x,origin.y,0), Point3d(origin.x,origin.y,10), Point3d(origin.x,origin.y,20), Point3d(origin.x,origin.y, 30)};
    for (const Point3d controlPoint : withP) {
        Point2d p = customMath::projectOnImgFrom3D(controlPoint, P);
        toDraw.push_back(Point2i(round(img.size().width - p.x), round(img.size().height - p.y)));
    }

    //compute test group homographies and their respective origin on the image
    int numberOfPlanes = hom.computeHomographies(10, 60);
    
    for (const auto& entry : hom.getHomographies()) {
        Point2d p = customMath::projectOnImgFromPlane(Point2d(origin.x + 1,origin.y), entry.second);
        cout << "p(" << entry.first << ") = " << p <<endl;
        toDraw.push_back(Point2i(round(img.size().width - p.x), round(img.size().height - p.y)));
    }

    for (int i=0; i < 8; i++) {
        if(i < 4) circle(img, toDraw[i], 2, Scalar(0,0,255), 2); // red
        else circle(img, toDraw[i], 2, Scalar(0,0,0), 2);        // black
    }
    
    imwrite("visionTest/testPP.png", img);
}

TEST(VisionTest, ParallelPlanesDS) {
    using namespace cv;
    using namespace std;

    int idCamera = 1;
   
    float size = 50.0f;
    size *= 20;
    string calPath = "../../data/calibrations_dataset/industry_safety/calibrations.json";
    Mat img = imread("../../data/industry_safety_0/rgb_00000_1.jpg");

    vector<Point2i> toDraw;

    // Define Calibrator and Homographer (tests subjects)
    Calibrator c(calPath, cv::Size(9,6), 1, idCamera);
    Mat K = c.getK();
    customMath::flipImageOrigin(K, img.size().width, img.size().height);
    Homographer hom(K, c.getR(), c.gett(), 0.1, 20);

    Mat H = hom.getGoundPlane();
    Mat P = hom.getP();

    cv::circle(img, cv::Point2i(300, 300), 2, cv::Scalar(0,0,0), 4);
    cv::Vec3d p_img_hom(img.size().width - 300, img.size().height - 300, 1);
    cv::Mat p_world_hom = H.inv() * p_img_hom ;
    cv::Vec3f p_world_euc(p_world_hom.at<double>(0), p_world_hom.at<double>(1), p_world_hom.at<double>(2));
    p_world_euc /= p_world_euc[2];
    p_world_euc[2] = 0;

    Point3f origin = {p_world_euc[0], p_world_euc[1], 0};
   

    //define control group points
    vector<Point3d> withP = {
        Point3d(origin.x, origin.y + 50, 0 * 50), 
        Point3d(origin.x, origin.y + 50, 10* 50), 
        Point3d(origin.x, origin.y + 50, 20* 50), 
        Point3d(origin.x, origin.y + 50, 30* 50)
    };

    for (const Point3d controlPoint : withP) {
        Point2d p = customMath::projectOnImgFrom3D(controlPoint, P);
        toDraw.push_back(Point2i(round(img.size().width - p.x), round(img.size().height - p.y)));
    }

    //compute test group homographies and their respective origin on the image
    int numberOfPlanes = hom.computeHomographies(500, 200);
    
    for (const auto& entry : hom.getHomographies()) {
        cout << " computing z = " << entry.first << endl;
        Point2d p = customMath::projectOnImgFromPlane(Point2d(origin.x, origin.y), entry.second);
        toDraw.push_back(Point2i(round(img.size().width - p.x), round(img.size().height - p.y)));
    }

    for (int i=0; i < 9; i++) {
        if(i < 4) circle(img, toDraw[i], 4, Scalar(0,0,255), 2); // red
        else circle(img, toDraw[i], 2, Scalar(255,0,0), 2);        // blue
    }
    
    imwrite("visionTest/testPP_ds.png", img);
}

TEST(VisionTest, DatasetHomography) {
    using namespace cv;
    using namespace std;

    int idCamera = 1;
   
    float size = 20.0f;
    size *= 20;
    string calPath = "../../data/calibrations_dataset/industry_safety/calibrations.json";
    Mat img = imread("../../data/industry_safety_0/rgb_00000_1.jpg");
    std::cout << "img h x w = " << img.size() << std::endl;
    // Define Calibrator and Homographer (tests subjects)
    Calibrator c(calPath, cv::Size(9,6), 1, idCamera);
    Mat K = c.getK();
    customMath::flipImageOrigin(K, img.size().width, img.size().height);

    Homographer hom(K, c.getR(), c.gett(), 0.1, 10);
    Mat H = hom.getGoundPlane();
    hom.computeHomographyForPlaneZ(size);
    Mat HZ = hom.getPlane(size);
    Mat P = hom.getP();

    cv::circle(img, cv::Point2i(300, 300), 2, cv::Scalar(0,0,0), 4);
    cv::Vec3d p_img_hom(img.size().width - 300, img.size().height - 300, 1);
    cv::Mat p_world_hom = H.inv() * p_img_hom ;
    cv::Vec3f p_world_euc(p_world_hom.at<double>(0), p_world_hom.at<double>(1), p_world_hom.at<double>(2));
    p_world_euc /= p_world_euc[2];
    
    Point3f origin = {p_world_euc[0], p_world_euc[1], 0};
    
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
        projectedPoints.push_back(customMath::projectOnImgFromPlane(Point2d(cube3D[i].x, cube3D[i].y), H));
    }

    for (int i = 0; i < 4; i++) {
        projectedPoints.push_back(customMath::projectOnImgFromPlane(Point2d(cube3D[i].x, cube3D[i].y), HZ));
    }
    
    
    vector<Point2i> toDraw;
    for(Point2d p : projectedPoints){
        toDraw.push_back(Point2i(round(img.size().width - p.x), round(img.size().height - p.y)));
    }

    // Define cube edges grouped by type
    vector<pair<int,int>> bottomEdges = {{0,1},{1,2},{2,3},{3,0}};
    vector<pair<int,int>> topEdges    = {{4,5},{5,6},{6,7},{7,4}};
    vector<pair<int,int>> verticalEdges = {{0,4},{1,5},{2,6},{3,7}};

    // Draw edges with different colors
    for (auto e : bottomEdges) line(img, toDraw[e.first], toDraw[e.second], Scalar(0,0,255), 2); // red
    for (auto e : topEdges)    line(img, toDraw[e.first], toDraw[e.second], Scalar(0,255,0), 2); // green
    for (auto e : verticalEdges) line(img, toDraw[e.first], toDraw[e.second], Scalar(255,0,0), 2); // blue

    imwrite("visionTest/datasetHomography.png", img);
}