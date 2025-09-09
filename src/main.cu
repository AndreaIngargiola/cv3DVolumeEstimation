#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "customMath.hpp"

// CUDA kernel
__global__ void invertKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) { 
        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            output[idx + c] = 255 - input[idx + c];
        }
    }
}

// Host wrapper
void launchInvertKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    unsigned char *d_input, *d_output;
    size_t size = width * height * channels * sizeof(unsigned char);

    cv::Vec2i p = customMath::invert2dAxisI(cv::Vec2i(2,3), 5, 7);
    std::cout << "Vec: (" << p[0] << ", " << p[1] << ")" << std::endl;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    invertKernel<<<grid, block>>>(d_input, d_output, width, height, channels);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    cv::Mat image = cv::imread("../data/test.jpg");  // <-- put a test image here
    if (image.empty()) {
        std::cerr << "Error: Could not load image!\n";
        return -1;
    }

    cv::Mat inverted;
    invertImageCUDA(image, inverted);

    cv::imshow("Original", image);
    cv::imshow("Inverted (CUDA)", inverted);
    cv::waitKey(0);

    return 0;
}
