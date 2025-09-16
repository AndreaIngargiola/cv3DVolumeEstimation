#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vision.hpp>
#include <filesystem>
#include <customMath.hpp>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

Homographer::Homographer(Calibrator cal,
                const int zSizeInCm,
                const std::string& poseImgPath)
                : zSizeInCm(zSizeInCm)
{
    this->strikeThePose(poseImgPath, cal);
}

void Homographer::strikeThePose(const std::string& poseImgPath, Calibrator cal) {

    // Check if poseImg exists and extract the corners from it
    Mat img = imread(poseImgPath);
    if(img.empty()){
        throw std::invalid_argument("Image not found: pose not set up");
    }

    int img_w, img_h;
    img_w = img.size().width;
    img_h = img.size().height;

    // Accumulators are necessary because findPatternFromImg can work for a batch of images (only 1 img in this case)
    std::vector<std::vector<Point2f>> imgPointsAcc;
    std::vector<std::vector<Point3f>> objPointsAcc;

    cal.findPatternFromImg(poseImgPath, imgPointsAcc, objPointsAcc);
    if(imgPointsAcc.size() == 0 || objPointsAcc.size() == 0) {
        throw std::invalid_argument("Pattern not found in poseImg: pose not set up");
    }

    std::vector<Point2f> corners = imgPointsAcc[0];
    std::vector<Point3f> objp = objPointsAcc[0];

    // Optimize K for selected pose
    this->K = getOptimalNewCameraMatrix(cal.getK(), cal.getDistCoeffs(), Size(img_w, img_h), 0);

    // Find new rvecs and tvecs (computing current pose)
    Mat rvec;
    solvePnP(objp, corners, this->K, cal.getDistCoeffs(), rvec, this->t);
    this->t = this->t.reshape(1,3);

    Rodrigues(rvec, this->R);   // Convert 1x3 rvecs to 3x3 R

    // Invert axis to have positive Z exiting the chessboard and origin of image plane in bottom-right
    customMath::invert3dAxis(this->K, this->R, img_w, img_h);

    // Compute P:  
    Mat Rt;                         // Concatenate R and t into [R|t]
    hconcat(this->R, this->t, Rt);  // size = 3x4

    this->P = this->K * Rt;         // Multiply K * [R|t] (size = 3x4)

    //compute H for ground floor plane and put it in the homographies array
    Mat H0 = (Mat_<double>(3,3));
    hconcat(std::vector<Mat>{ this->P.col(0), this->P.col(1), this->P.col(3) }, H0);
    this->homographies.emplace(0, H0);

    // Extract l and v from P
    this->v = P.col(2);
    this->l = P.col(3);
}

Mat Homographer::getGoundPlane() {
    return this->homographies.at(0);
}

Mat Homographer::getP() {
    return this->P;
}