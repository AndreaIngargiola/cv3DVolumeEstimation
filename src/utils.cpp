#include "utils.hpp"
#include <opencv2/opencv.hpp>

// Declaration of CUDA kernel wrapper (defined in main.cu)
void launchInvertKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels);

void invertImageCUDA(const cv::Mat& input, cv::Mat& output) {
    output.create(input.size(), input.type());

    // Launch CUDA kernel
    launchInvertKernel(input.data, output.data, input.cols, input.rows, input.channels());
}
