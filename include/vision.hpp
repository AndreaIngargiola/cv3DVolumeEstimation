#pragma once

#include <string>
#include <opencv2/core.hpp>

class Calibrator {
private:
   
    const std::string valuesPath;
    const std::string samplesPath;
    const cv::Size patternShape;
    const int patternType;

    cv::Matx33f K;
    cv::Mat distCoeffs;
    float reprojectionError;


public:

    Calibrator( const std::string& valuesPath,
                const std::string& samplesPath,
                const cv::Size& patternShape,
                const int patternType);
    
    cv::Matx33f getK();
    cv::Mat getDistCoeffs();
    float getReprojectionError();

private:
    void retrieveValues();
    void calibrate();
};