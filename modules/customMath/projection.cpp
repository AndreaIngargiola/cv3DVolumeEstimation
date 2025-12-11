#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

using namespace cv;
namespace customMath {
    Point2d projectOnImgFromPlane(Point2d X, Mat H) {
        Vec3d P_hom = Vec3d(X.x, X.y, 1.0); // To homogeneous coordinates
        Mat p_hom = H * P_hom;              // Homography
        p_hom /= p_hom.at<double>(2);     // Normalize
        return Point2d(p_hom.at<double>(0), p_hom.at<double>(1));
    }

    Point2d projectOnImgFrom3D(Point3d X, Mat P) {
        Vec4d X_hom = Vec4d(X.x, X.y, X.z, 1);  // To homogeneous coordinates
        Mat x_hom = P * X_hom;                  // Homography (3,1) = (3,4) X (4,1)
        x_hom /= x_hom.at<double>(2,0);         // Normalize
        return Point2d(x_hom.at<double>(0,0), x_hom.at<double>(0,1));
    }
}