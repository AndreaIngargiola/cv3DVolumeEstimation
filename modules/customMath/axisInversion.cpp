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

    void invert3dAxis(Matx33f& K, Matx33f& R, int img_w, int img_h) {
        
        //Transform matrix to flip axes
        Matx33f T(
            -1.0f,  0.0f,  img_w,  // x' = -x + w  (origin on right)
            0.0f,   -1.0f, img_h,  // y' = -y + h  (origin on bottom)
            0.0f,   0.0f,  1.0f
        );
        
        K = T * K;

        Matx33f D(
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, -1.0f
        );

        R = R * D; //flip Z (positive values of Z are out of the chessboard)
    }
}