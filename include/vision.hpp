#pragma once

#include <string>
#include <opencv2/core.hpp>
#include <unordered_map>

class Calibrator {
private:
    const std::string valuesPath;
    const std::string samplesPath;
    const cv::Size patternShape;
    const int patternType;

    cv::Mat K;
    cv::Mat distCoeffs;
    float reprojectionError;


public:
    Calibrator( const std::string& valuesPath,
                const std::string& samplesPath,
                const cv::Size& patternShape,
                const int patternType);
    
    cv::Mat getK();
    cv::Mat getDistCoeffs();
    float getReprojectionError();
    cv::Size getPatternShape();
    void findPatternFromImg( const std::string& imgPath, 
                            std::vector<std::vector<cv::Point2f>>& cornersAccumulator, 
                            std::vector<std::vector<cv::Point3f>>& objpAccumulator);

private:
    void calibrate();
    void saveCalibration();
    void loadCalibration();
};

class Homographer {

private:
    cv::Mat P;          //projection matrix P = K @ Rt
    cv::Mat K;          //intrinsics matrix
    cv::Mat R;          //rotation matrix
    cv::Mat t;          //translation vector
    cv::Vec3d v;        //vanishing line for Z axis
    cv::Vec3d l;        //vanishing line for X and Y axis
    cv::Vec3d base;     //homogeneous pixel coordinates of the point laying on Z = 0
    const int zSizeInCm;
    std::unordered_map<int, cv::Mat> homographies;

public:
    Homographer(Calibrator cal,
                const int zSizeInCm,
                const std::string& poseImgPath);

    void strikeThePose(const std::string& poseImgPath, Calibrator cal);
    int computeHomographies(const int deltaZ, const int maxHeightInCm);

    cv::Mat getGoundPlane();
    std::unordered_map<int, cv::Mat> getHomographies();
    cv::Mat getPlane(const int z);
    cv::Mat getP();

private:
    void computeHomographyForPlaneZ(const double z, cv::Vec3d top);
};