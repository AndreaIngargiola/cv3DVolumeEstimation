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
    if (fs::exists(this->valuesPath) && fs::is_regular_file(this->valuesPath)) {
        this->loadCalibration();
    } else {
        this->calibrate();
        this->saveCalibration();
    }
    
}

Mat Calibrator::getK() {
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
    for (const auto& entry : fs::directory_iterator(this->samplesPath)) {
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

    Size imgSize = imread(fs::directory_iterator(this->samplesPath)->path().string()).size();

    calibrateCamera(objPoints, imgPoints, imgSize, this->K, this->distCoeffs, rvecs, tvecs);
}

void Calibrator::saveCalibration() 
{
    // Open file for writing
    FileStorage fs(this->valuesPath, FileStorage::WRITE);

    if (!fs.isOpened()) {
        std::cerr << "Error: Cannot open file " << this->valuesPath << " for writing." << std::endl;
        return;
    }

    // Save data
    fs << "K" << this->K;
    fs << "distCoeffs" << this->distCoeffs;

    fs.release();
    std::cout << "Calibration saved to " << this->valuesPath << std::endl;
}

void Calibrator::loadCalibration() {
    FileStorage fs(this->valuesPath, FileStorage::READ);

    if (!fs.isOpened()) {
        std::cerr << "Error: Cannot open file " << this->valuesPath << " for reading." << std::endl;
        return;
    }

    //If yml fields exists, load them, or else calibrate and save 
    if (!fs["K"].empty() && !fs["distCoeffs"].empty()) {

        // Load raw data
        fs["K"] >> this->K;
        fs["distCoeffs"] >> this->distCoeffs;

        fs.release();

        // Ensure double precision (CV_64F)
        if (this->K.type() != CV_64F) {
            K.convertTo(K, CV_64F);
        }
        if (distCoeffs.type() != CV_64F) {
            distCoeffs.convertTo(distCoeffs, CV_64F);
        }

        std::cout << "Calibration loaded from " << this->valuesPath << std::endl;
        std::cout << "K = " << K << std::endl;
        std::cout << "distCoeffs = " << distCoeffs << std::endl;
    } else {
        fs.release();
        this->calibrate();
        this->saveCalibration();
    }
}