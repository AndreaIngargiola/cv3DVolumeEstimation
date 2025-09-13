#pragma once

#include <string>
#include <opencv2/core.hpp>

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