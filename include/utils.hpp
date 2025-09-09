#pragma once
#include <opencv2/opencv.hpp>

// Invert image using CUDA
void invertImageCUDA(const cv::Mat& input, cv::Mat& output);
