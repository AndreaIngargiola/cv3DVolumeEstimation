#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

namespace customMath {

    using namespace cv; 

    Vec3f toHom(Vec2f euclideanVec) {
        return Vec3f(euclideanVec[0], euclideanVec[1], 1.0);
    }

    Vec3f toEuc(Vec3f homogeneousVec) {
        return Vec3f(homogeneousVec[0] / homogeneousVec[2], homogeneousVec[0] / homogeneousVec[2], 0);
    }
}
