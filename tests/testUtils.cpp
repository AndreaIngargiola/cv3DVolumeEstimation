#include <gtest/gtest.h>
#include <customMath.hpp>
#include <opencv2/core.hpp>

TEST(CustomMathTest, InvertAxis) {
    cv::Vec2i p(2, 3);
    auto result = customMath::invert2dAxisI(p, 10, 10);
    EXPECT_EQ(result[0], 8);
    EXPECT_EQ(result[1], 7);
}


