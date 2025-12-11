#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vision.hpp>
#include <filesystem>
#include <fstream>  
#include <customMath.hpp>
#include <nlohmann/json.hpp>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

Homographer::Homographer(
    const cv::Mat K, const cv::Mat R, const cv::Mat t,
    const float zSizeInCm, const int calibrationScaleFactor, 
    const cv::Vec3i baseOn3d)
    : K(K), R(R), t(t), 
    zSizeInCm(zSizeInCm), calibrationScaleFactor(calibrationScaleFactor)
{
    // Build Rt and P
    Mat Rt;                         // Concatenate R and t into [R|t]
    hconcat(this->R, this->t, Rt);  // size = 3x4
    this->P = this->K * Rt;         // Multiply K * [R|t] (size = 3x4)
    
    this->base = Point3d(baseOn3d[0], baseOn3d[1], baseOn3d[2]);
    this->computeHomographyForPlaneZ(0);
}

Mat Homographer::getGoundPlane() {
    return this->homographies.at(0);
}

Mat Homographer::getPlane(int z) {
    return this->homographies.at(z);
}

unordered_map<int, cv::Mat> Homographer::getHomographies() {
    return this->homographies;
}

Mat Homographer::getP() {
    return this->P;
}

int Homographer::computeHomographies(const int deltaZ, const int maxHeightInCm) {
    
    // Compute the Z equally spaced values given the maximum height i wanna reach and 
    // the delta between the values (starting from the ground)

    int maxHeightInZ = maxHeightInCm / this->zSizeInCm;
    int numberOfPlanes = maxHeightInZ / deltaZ;

    std::vector<double> zValues;
    for (double v = 0; v <= maxHeightInZ; v += deltaZ) {
        zValues.push_back(v);
    }
  
    // Compute the homographies
    for(int i = 1; i <= numberOfPlanes; i++) {
        computeHomographyForPlaneZ(zValues[i]);
    }

    return numberOfPlanes;
}

void Homographer::computeHomographyForPlaneZ(const double z){

    if(this->homographies.find(z) != this->homographies.end()) return;

    vector<Point2d> planePts;  // (X,Y) on plane
    vector<Point2d> imgPts;    // image projections

    // pick 100 good points on the plane
    // e.g. plane corners in world units (cm)
    vector<Point3d> samples;

    for(int i=0; i < 10; ++i ){
        for(int j=0; j<10; ++j){
            samples.emplace_back(Point3d(this->base.x + i * this->calibrationScaleFactor, this->base.y + j * this->calibrationScaleFactor, z));
        }
    }

    for (auto& planePoint : samples) {
        // world coords → homogeneous
        //Mat X = (Mat_<double>(4,1) << planePoint.x, planePoint.y, planePoint.z, 1.0);
        Mat imagePoint = P * (Mat_<double>(4,1) << planePoint.x, planePoint.y, planePoint.z, 1.0);
        imagePoint /= imagePoint.at<double>(2,0);
        double x_img = imagePoint.at<double>(0);
        double y_img = imagePoint.at<double>(1);

        // plane coordinates (x,y) = (X,Y) in same units as calibration
        planePts.emplace_back(planePoint.x, planePoint.y);
        imgPts.emplace_back(x_img, y_img);
    }

    // homography from plane coords → image coords
    Mat HZ = findHomography(planePts, imgPts, RANSAC);

    // Save the new Homography matrix with its respective Z
    this->homographies.emplace(z, HZ);
}