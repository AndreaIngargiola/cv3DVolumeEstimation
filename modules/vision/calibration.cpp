#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vision.hpp>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;


Calibrator::Calibrator(const string& valuesPath,
            const string& samplesPath,
            const Size& patternShape,
            const int patternType)
            : valuesPath(valuesPath),
            samplesPath(samplesPath),
            patternShape(patternShape),
            patternType(patternType)
{
    this->calibrate();
}

Matx33f Calibrator::getK() {
    return this->K;
}

Mat Calibrator::getDistCoeffs() {
    return this->distCoeffs;
}

float Calibrator::getReprojectionError() {
    return this->reprojectionError;
}

void Calibrator::calibrate() {
    
    std::vector<Mat> rvecs; 
    std::vector<Mat> tvecs;
    std::vector<std::vector<Point2f>> imgPoints;
    std::vector<std::vector<Point3f>> objPoints;

    // Prepare one set of object points
    std::vector<Point3f> objp;
    for (int i = 0; i < patternShape.height; i++) {
        for (int j = 0; j < patternShape.width; j++) {
            objp.emplace_back(j, i, 0.0f);
        }
    }

    // Process all images in folder
    for (const auto& entry : fs::directory_iterator(samplesPath)) {
        if (entry.is_regular_file()) {
            Mat img = imread(entry.path().string());
            if (img.empty()) continue;

            std::vector<Point2f> corners;
            bool found = findChessboardCorners(img, patternShape, corners);

            if (found) {
                Mat gray;
                cvtColor(img, gray, COLOR_BGR2GRAY);
                cornerSubPix(
                    gray, corners, Size(11, 11), Size(-1, -1),
                    TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001)
                );

                imgPoints.push_back(corners);
                objPoints.push_back(objp);
            }
        }
    }

    Size imgSize = imread(fs::directory_iterator(samplesPath)->path().string()).size();

    calibrateCamera(objPoints, imgPoints, imgSize, K, distCoeffs, rvecs, tvecs);
}

void Calibrator::retrieveValues() {

}