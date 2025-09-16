#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

using namespace cv;
namespace customMath {
    Point2d projectOnImgFromPlane(Point3d P, Mat H) {
        if(P.z != 0) {
            std::cout << "projectFromPlane with inhomogeneous Z = " << P.z << std::endl;
        }
        Vec3d P_hom = Vec3d(P.x, P.y, 1.0); // To homogeneous coordinates
        Mat p_hom = H * P_hom;              // Homography
        p_hom /= p_hom.at<double>(2,0);     // Normalize
        return Point2d(p_hom.at<double>(0,0), p_hom.at<double>(0,1));
    }

    Point2d projectOnImgFrom3D(Point3d P, Mat K) {
        Vec4d P_hom = Vec4d(P.x, P.y, P.z, 1);  // To homogeneous coordinates
        Mat p_hom = K * P_hom;                  // Homography (3,1) = (3,4) X (4,1)
        p_hom /= p_hom.at<double>(2,0);         // Normalize
        return Point2d(p_hom.at<double>(0,0), p_hom.at<double>(0,1));
    }
}