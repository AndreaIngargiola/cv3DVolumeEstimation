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

private:
    void calibrate();
    void saveCalibration();
    void loadCalibration();
};

class Homographer {

private:
    cv::Mat P;          //projection matrix P = K @ Rt
    cv::Mat K;          //intrinsics matrix
    cv::Mat distCoeffs;
    cv::Mat R;          //rotation matrix
    cv::Mat t;          //translation vector
    cv::Vec3d v;        //vanishing line for Z axis
    cv::Vec3d l;        //vanishing line for X and Y axis
    cv::Vec3i b;        //homogeneous pixel coordinates of the point laying on Z = 0
    const int zSizeInCm;
    const cv::Size& patternShape;
    std::unordered_map<int, cv::Mat> homographies;

public:
    Homographer(cv::Mat K,
                cv::Mat distCoeffs,
                const int zSizeInCm,
                const cv::Size& patternShape,
                const std::string& poseImgPath);

    void strikeThePose(const std::string& poseImgPath);
    void computeHomographies(const int delta, const int maxHeightInCm);

    cv::Mat getGoundPlane();
    std::unordered_map<int, cv::Mat> getHomographies();
    cv::Mat getPlane(const int z);

private:
    cv::Vec3i getTforPlaneZ(const int z);
    void computeHomographyForPlaneZ(const int z);
};