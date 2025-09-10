#pragma once

#include <string>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

class Calibrator {
private:
   
    const string valuesPath;
    const string samplesPath;
    const Vec2i patternShape;

    Matx33f K, R;
    Matx31f t;
    float reprojectionError;


public:

    Calibrator(const string& values,
               const string& samples,
               const Vec2i& pattern);
    
    Matx33f getK();
    Matx33f getR();
    Matx31f gett();
    float getReprojectionError();

private:
    void retrieveValues();
    void calibrate();
};