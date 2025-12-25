#pragma once

#include <string>
#include <opencv2/core.hpp>
#include <unordered_map>
#include <thrust/host_vector.h>

class Calibrator {
private:
    const std::string valuesPath;
    const std::string samplesPath;
    std::string poseImgPath;
    const cv::Size patternShape;
    const int patternType;

    const int cameraId;

    cv::Mat K;          //intrinsics matrix
    cv::Mat R;          //rotation matrix
    cv::Mat t;          //translation vector
    cv::Mat distCoeffs;
    float reprojectionError;


public:
    Calibrator( const std::string& valuesPath,
                const cv::Size& patternShape,
                const int patternType,
                const int cameraId,
                const std::string& samplesPath = "",
                const std::string& poseImgPath = "");
    
    cv::Mat getK();
    cv::Mat getR();
    cv::Mat gett();
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
    cv::Point3d base;     //homogeneous pixel coordinates of the point laying on Z = 0
    
    const float zSizeInCm;
    const int calibrationScaleFactor;
    std::unordered_map<int, cv::Mat> homographies;

public:
    Homographer(const cv::Mat K, const cv::Mat R, const cv::Mat t,
                const float zSizeInCm, const int calibrationScaleFactor, 
                const cv::Vec3i baseOn3d = cv::Vec3i(0,0,0));

    int computeHomographies(const int deltaZ, const int maxHeightInCm);
    cv::Mat getGoundPlane();
    std::unordered_map<int, cv::Mat> getHomographies();
    cv::Mat getPlane(const int z);
    cv::Mat getP();
    void computeHomographyForPlaneZ(const double z);
};

struct Cuboid{
    cv::Point3d origin;
    float dx;
    float dy;
    float dz;
    std::vector<cv::Point3d> pts;
    int classId;
};

struct DataPoint {
    float x, y;
    int classId = -1;
};


class TridimensionalReconstructor {
    private:
    cv::Mat K;          //intrinsics matrix
    cv::Mat R;          //rotation matrix
    cv::Mat t;          //translation vector
    thrust::host_vector<Cuboid> boundingBoxes;
    std::vector<std::vector<DataPoint>> kp;
    float zDimensionInCm;
    std::unordered_map<int, cv::Mat> zPlanes;

    public:
    TridimensionalReconstructor(const cv::Mat K, const cv::Mat R, const cv::Mat t, Homographer hom, float zDimensionInCm);
    std::vector<Cuboid> get3DBoundingBoxes(cv::Mat frame, std::vector<DataPoint> kp);
    void computeClusterDim(std::vector<DataPoint> kp, cv::Point3d& base, cv::Point3d& top, float& height, float& width);
    Cuboid computeCluster3d(std::vector<DataPoint> kp);
};