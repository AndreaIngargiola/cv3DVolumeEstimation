#include <string>
#include <opencv2/core.hpp>
#include <vision.hpp>

using namespace cv;
using namespace std;


Calibrator::Calibrator(const string& values,
            const string& samples,
            const Vec2i& pattern)
            : valuesPath(values),
            samplesPath(samples),
            patternShape(pattern)
{}

Matx33f Calibrator::getK() {
    return this->K;
}

Matx33f Calibrator::getR() {
    return this->R;
}

Matx31f Calibrator::gett() {
    return this->t;
}

float Calibrator::getReprojectionError() {
    return this->reprojectionError;
}

void Calibrator::calibrate() {

}

void Calibrator::retrieveValues() {

}