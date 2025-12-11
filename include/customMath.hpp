#pragma once

#include <iostream>
#include <opencv2/core/types.hpp>

namespace customMath {
    using namespace cv; 

    Vec3f toHom(Vec2f euclideanVec);

    Vec3f toEuc(Vec3f homogeneousVec);

    Vec2i invert2dAxisI(Vec2i p, int img_w, int img_h);

    Vec2f invert2dAxisF(Vec2f p, int img_w, int img_h);

    void flipZaxis(Mat& R);

    void flipImageOrigin(Mat& K, float img_w, float img_h);

    Point2d projectOnImgFromPlane(Point2d X, Mat H);

    Point2d projectOnImgFrom3D(Point3d X, Mat P);
}