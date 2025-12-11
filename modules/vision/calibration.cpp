#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vision.hpp>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <customMath.hpp>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

Calibrator::Calibrator(const string& valuesPath,
            const Size& patternShape,
            const int patternType,
            const int cameraId,
            const string& samplesPath,
            const string& poseImgPath)
            : valuesPath(valuesPath),
            samplesPath(samplesPath),
            poseImgPath(poseImgPath),
            patternShape(patternShape),
            patternType(patternType),
            cameraId(cameraId)
{
    if (fs::exists(this->valuesPath)) {
        this->loadCalibration();
    } else {
        this->calibrate();
        this->saveCalibration();
    }
}

Mat Calibrator::getK() {
    return this->K;
}

Mat Calibrator::getR() {
    return this->R;
}

Mat Calibrator::gett() {
    return this->t;
}

Mat Calibrator::getDistCoeffs() {
    return this->distCoeffs;
}

float Calibrator::getReprojectionError() {
    return this->reprojectionError;
}

Size Calibrator::getPatternShape() {
    return this->patternShape;
}

void Calibrator::findPatternFromImg(const string& imgPath, 
                                    vector<vector<Point2f>>& cornersAccumulator, 
                                    vector<vector<Point3f>>& objpAccumulator) 
{
    Mat img = imread(imgPath);
    if (img.empty()) {
        throw std::invalid_argument("Image not found at path " + imgPath);
    }

    // Prepare one set of object points
    std::vector<Point3f> objp;
    for (int i = 0; i < patternShape.height; i++) {
        for (int j = 0; j < patternShape.width; j++) {
            objp.emplace_back(j, i, 0.0f);
        }
    }

    std::vector<Point2f> corners;
    bool found = findChessboardCorners(img, patternShape, corners);

    if (found) {
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        cornerSubPix(
            gray, corners, Size(11, 11), Size(-1, -1),
            TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001)
        );

        cornersAccumulator.push_back(corners);
        objpAccumulator.push_back(objp);
    }

}

void Calibrator::calibrate() {
    vector<Mat> rvecs; 
    vector<Mat> tvecs;
    vector<vector<Point2f>> imgPoints;
    vector<vector<Point3f>> objPoints;

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
            this->findPatternFromImg(entry.path().string(), imgPoints, objPoints);
        }
    }

    Size imgSize = imread(fs::directory_iterator(this->samplesPath)->path().string()).size();

    calibrateCamera(objPoints, imgPoints, imgSize, this->K, this->distCoeffs, rvecs, tvecs);

    Mat poseImg = imread(this->poseImgPath);
    if(poseImg.empty()) throw Exception();
    int img_w, img_h;
    img_w = poseImg.size().width;
    img_h = poseImg.size().height;

    // Accumulators are necessary because findPatternFromImg can work for a batch of images (only 1 img in this case)
    vector<vector<Point2f>> imgPointsAcc;
    vector<vector<Point3f>> objPointsAcc;

    findPatternFromImg(poseImgPath, imgPointsAcc, objPointsAcc);
    if(imgPointsAcc.size() == 0 || objPointsAcc.size() == 0) {
        throw std::invalid_argument("Pattern not found in poseImg: pose not set up");
    }

    vector<Point2f> corners = imgPointsAcc[0];
    objp.clear();
    objp = objPointsAcc[0];

    // Optimize K for selected pose
    this->K = getOptimalNewCameraMatrix(getK(), getDistCoeffs(), Size(img_w, img_h), 0);

    // Find new rvecs and tvecs (computing current pose)
    Mat rvec;
    solvePnP(objp, corners, this->K, getDistCoeffs(), rvec, this->t);
    this->t = this->t.reshape(1,3);

    Rodrigues(rvec, this->R);       // Convert 1x3 rvecs to 3x3 R
}

void Calibrator::saveCalibration() 
{
    using json = nlohmann::json;
    json j;
    json camEntry;
    camEntry["CameraId"] = this->cameraId;

    // Intrinsic
    camEntry["IntrinsicParameters"] = {
        {"Fx", this->K.at<double>(0,0)},
        {"Fy", this->K.at<double>(1,1)},
        {"Cx", this->K.at<double>(0,2)},
        {"Cy", this->K.at<double>(1,2)}
    };

    // Extrinsic
    camEntry["ExtrinsicParameters"] = {
        {"Rotation", {
            this->R.at<double>(0,0), this->R.at<double>(1,0), this->R.at<double>(2,0),
            this->R.at<double>(0,1), this->R.at<double>(1,1), this->R.at<double>(2,1),
            this->R.at<double>(0,2), this->R.at<double>(1,2), this->R.at<double>(2,2)
        }},
        {"Translation", {
            this->t.at<double>(0), this->t.at<double>(1), this->t.at<double>(2)
        }}
    };

    j["Cameras"].push_back(camEntry);
    
    // Write file
    fs::path p(this->valuesPath);

    // Create parent directory if needed
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    std::ofstream out(this->valuesPath);
    if (!out.is_open()) {
        cerr << "ERROR: cannot write to " << this->valuesPath << "\n";
        return;
    }

    out << std::setw(4) << j;    // pretty print
    out.close();
    cout << "Calibration saved to " << this->valuesPath << std::endl;
}

void Calibrator::loadCalibration() {

    // Load JSON file
    using json = nlohmann::json;
    ifstream f(this->valuesPath);
    if (!f.is_open()) {
        cerr << "Cannot open: " << this->valuesPath << std::endl;
    }

    json j;
    f >> j;

    // Loop through cameras
    for (auto& cam : j["Cameras"]) {
        int cameraId = cam["CameraId"];
        if(cameraId != this->cameraId) {
            continue;
        }
        // Intrinsic parameters
        double fx = cam["IntrinsicParameters"]["Fx"];
        double fy = cam["IntrinsicParameters"]["Fy"];
        double cx = cam["IntrinsicParameters"]["Cx"];
        double cy = cam["IntrinsicParameters"]["Cy"];

        this->K = (Mat_<double>(3,3) <<
            fx,  0.f, cx,
             0.f, fy, cy,
             0.f,  0.f,  1.f
        );

        // Extrinsic parameters
        auto Rjson = cam["ExtrinsicParameters"]["Rotation"];
        auto Tjson = cam["ExtrinsicParameters"]["Translation"];

        this->R = (Mat_<double>(3,3) <<
            (double)Rjson[0], (double)Rjson[3], (double)Rjson[6],
            (double)Rjson[1], (double)Rjson[4], (double)Rjson[7],
            (double)Rjson[2], (double)Rjson[5], (double)Rjson[8]
        );
       
        this->t = (Mat_<double>(3,1) << 
            (double)Tjson[0], 
            (double)Tjson[1], 
            (double)Tjson[2]
        );
    }

    if(this->K.empty() || this->R.empty() || this->t.empty()) {
        this->calibrate();
        this->saveCalibration();
    }
}