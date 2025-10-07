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


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>

using namespace cv::dnn;
using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video file: " << argv[1] << std::endl;
        return -1;
    }
    
    // CUDA background subtractor (MOG2)
    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> pBackSub =
        cv::cuda::createBackgroundSubtractorMOG2(500, 16.0, true);

    // CUDA goodFeaturesToTrack
    cv::Ptr<cv::cuda::CornersDetector> detector =
        cv::cuda::createGoodFeaturesToTrackDetector(
            CV_8UC1,   // input type (grayscale)
            1000,       // maxCorners
            0.01,      // qualityLevel
            10,        // minDistance
            3,         // blockSize
            false,     // useHarrisDetector
            0.04       // k
        );

    cv::Mat frame, gray, blurred;
    cv::cuda::GpuMat d_input, d_mask, d_corners, d_gray;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        // ---- CPU preprocessing ----
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        d_gray.upload(gray);
        cv::GaussianBlur(gray, blurred, cv::Size(5,5), 0);

        // ---- Upload to GPU ----
        d_input.upload(blurred);

        // ---- Background subtraction ----
        pBackSub->apply(d_input, d_mask);

        cv::Mat mask;
        d_mask.download(mask);

        // --- CPU morphological filtering ---
        cv::Mat maskClean;
        cv::morphologyEx(mask, maskClean, cv::MORPH_OPEN,
                        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
        cv::morphologyEx(maskClean, maskClean, cv::MORPH_CLOSE,
                        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7)));

        cv::morphologyEx(maskClean, maskClean, cv::MORPH_DILATE,
                 cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7)));
        // Optional: remove shadows (keep only 0/255)
        cv::threshold(maskClean, maskClean, 200, 255, cv::THRESH_BINARY);

        // Upload back to GPU for masking
        d_mask.upload(maskClean);

        cv::cuda::threshold(d_mask, d_mask, 200, 255, cv::THRESH_BINARY);
        cv::cuda::bitwise_and(d_gray, d_mask, d_gray);

        // ---- Detect corners on GPU ----
        detector->detect(d_gray, d_corners);

        // ---- Download results ----
        cv::Mat corners;
        d_mask.download(mask);
        d_corners.download(corners);

        // Draw corners
        for (int i = 0; i < corners.cols; i++) {
            cv::Point2f pt = corners.at<cv::Point2f>(i);
            cv::circle(frame, pt, 6, cv::Scalar(255,0,0), -1);
        }

        // Show results
        cv::imshow("Frame", frame);
        cv::imshow("Foreground Mask", mask);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) break; // ESC
    }

    return 0;
}