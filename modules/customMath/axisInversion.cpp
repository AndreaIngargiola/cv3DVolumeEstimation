#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>


namespace customMath {

    using namespace cv;
    
    Vec2i invert2dAxisI(Vec2i p, int img_w, int img_h) {
        return Vec2i(img_w - p[0], img_h - p[1]);
    }

    Vec2f invert2dAxisF(Vec2f p, int img_w, int img_h) {
        return Vec2f(img_w - p[0], img_h - p[1]);
    }

    void flipZaxis(Mat& R) {
        Mat D = (Mat_<double>(3,3) <<
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, -1.0f
        );

        R = R * D; //flip Z (positive values of Z are out of the chessboard)
    }

    void flipImageOrigin(Mat& K, float img_w, float img_h) {
        
        //Transform matrix to flip axes
        Mat T = (Mat_<double>(3,3) <<
                -1.0f,  0.0f,  img_w,  // x' = -x + w  (origin on right)
                0.0f,   -1.0f, img_h,  // y' = -y + h  (origin on bottom)
                0.0f,   0.0f,  1.0f
        );
        
        K = T * K;
    }
}