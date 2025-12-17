#include <segmentation.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace cv::cuda;

PreProcesser::PreProcesser(const GpuMat& d_mask) : d_mask(d_mask) {}

KPExtractor::KPExtractor(const int frameSetSize, GpuMat& d_mask, GpuMat& d_cumulativeStatus) 
                        : frameSetSize(frameSetSize),
                        d_mask(d_mask),
                        d_cumulativeStatus(d_cumulativeStatus),
                        preProc(this->d_mask) {}

GpuMat KPExtractor::getUnclusteredKeypoints(Mat& frame) {
    frame.copyTo(this->frame);
    this->preProc.preprocessFrame(this->frame, this->frame);  // conversion to greyscale and update of background subtractor
    this->d_mask = this->preProc.getMask();
    this->d_frame.copyTo(this->d_prevFrame);
    this->d_frame.upload(this->frame);
    cv::cuda::bitwise_and(d_frame, d_mask, d_frame); // remove background

    this->d_keypoints.copyTo(this->d_prevKeypoints);
    
    if(this->frameCounter == 0) {
        this->findNewKeypoints();
        this->frameCounter++;
    }
    else {
        if(this->d_cumulativeStatus.cols > 1) this->trackKeypoints(); //track keypoints only if there are keypoints to track
        ++this->frameCounter < frameSetSize ? this->frameCounter : this->frameCounter = 0;
    }

    return this->d_keypoints;
}

void KPExtractor::findNewKeypoints() {
    // Detect corners on GPU
    kpDetector->detect(d_frame, this->d_keypoints);
    
    // Reset cumulativeStatus
    int cols = std::max(1, d_keypoints.cols);
    this->d_cumulativeStatus.create(1, cols, CV_8UC1);
    this->d_cumulativeStatus.setTo(Scalar(255));
}

void KPExtractor::trackKeypoints()
{
    GpuMat d_status;

    this->lucasKanadeOpticalFlow->calc( this->d_prevFrame, 
                                        this->d_frame, 
                                        this->d_prevKeypoints, 
                                        this->d_keypoints, 
                                        d_status);
    
    // Update statuses accumulator (multiply guarantees that if a keypoint has status = 0 will remain 0 forever)
    cuda::multiply(this->d_cumulativeStatus, d_status, this->d_cumulativeStatus);

    // Overwrite dead keypoints in place
    cv::cuda::GpuMat d_invertedMask;
    cv::cuda::compare(this->d_cumulativeStatus, cv::Scalar(0), d_invertedMask, cv::CMP_EQ);
    this->d_keypoints.setTo(cv::Scalar(-1.0f, -1.0f), d_invertedMask);
}


__global__
void hist_contrib_kernel(const float* d_hist, double* d_num, double* d_den, int bins)
{
    // linear index over bins*bins elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = bins * bins;
    if (idx >= N) return;

    // mapping: we assume hist is row-major with row = g1, col = g2
    // i.e. flatten as idx = g1 * bins + g2
    int g1 = idx / bins;
    int g2 = idx % bins;

    double freq = static_cast<double>(d_hist[idx]); // frequency (possibly 0)
    if (freq == 0.0) {
        d_num[idx] = 0.0;
        d_den[idx] = 0.0;
        return;
    }

    d_num[idx] = (double)g1 * (double)g2 * freq;
    d_den[idx] = (double)g1 * (double)g1 * freq;
}


// Kernel: one thread per pixel
__global__ void comparagram_kernel(const unsigned char* img1,
                                   const unsigned char* img2,
                                   float* comparagram,
                                   int rows, int cols) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (px >= total) return;

    unsigned char g1 = img1[px];
    unsigned char g2 = img2[px];
    int cidx = (int)g1 * 256 + (int)g2;
    // atomic add on float
    atomicAdd(&comparagram[cidx], 1.0f);
}

float PreProcesser::normalizeBrightness(){
    const int bins = 256;
    
    cv::Mat h_comparagram(256, 256, CV_32F);//= cv::Mat::zeros(bins, bins, CV_32F);;

    // cv::cuda::GpuMat d_prevFrame, d_frame;  // CV_8UC1 same size

    GpuMat d_frame, d_prevFrame;
    d_frame.upload(this->h_frame);
    d_prevFrame.upload(this->h_prevFrame);
    int rows = d_prevFrame.rows;
    int cols = d_prevFrame.cols;
    int total_pixels = rows * cols;

    // Allocate device comparagram
    float* d_comparagram;
    cudaMalloc(&d_comparagram, 256 * 256 * sizeof(float));
    cudaMemset(d_comparagram, 0, 256 * 256 * sizeof(float));

    // Launch kernel
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;

    comparagram_kernel<<<blocks, threads>>>(
        d_prevFrame.ptr<uchar>(),
        d_frame.ptr<uchar>(),
        d_comparagram,
        rows, cols
    );
    cudaDeviceSynchronize();
    cudaMemcpy(h_comparagram.data, d_comparagram,
           256 * 256 * sizeof(float),
           cudaMemcpyDeviceToHost);

    cudaFree(d_comparagram);

    int N = bins * bins;
    // Ensure continuous layout
    cv::Mat comp_cont = h_comparagram.isContinuous() ? h_comparagram : h_comparagram.clone();
    const float* h_comp_ptr = comp_cont.ptr<float>();

    // Copy histogram to device (thrust device_vector for simplicity)
    thrust::device_vector<float> d_comp(h_comp_ptr, h_comp_ptr + N);
    float* d_comp_ptr = thrust::raw_pointer_cast(d_comp.data());

    // Allocate device arrays for per-bin contributions (double for precision)
    thrust::device_vector<double> d_num(N);
    thrust::device_vector<double> d_den(N);
    double* d_num_ptr = thrust::raw_pointer_cast(d_num.data());
    double* d_den_ptr = thrust::raw_pointer_cast(d_den.data());

    // Launch kernel to compute per-bin contributions
    int threadsPerBlock = 256;
    blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    hist_contrib_kernel<<<blocks, threadsPerBlock>>>(d_comp_ptr, d_num_ptr, d_den_ptr, bins);
    cudaDeviceSynchronize();

    // Reduce with Thrust to obtain numerator and denominator
    double numerator = thrust::reduce(thrust::device, d_num.begin(), d_num.end(), 0.0, thrust::plus<double>());
    double denominator = thrust::reduce(thrust::device, d_den.begin(), d_den.end(), 0.0, thrust::plus<double>());

    // Safety: avoid division by zero
    const double eps = 1e-12;
    if (denominator < eps) {
        // fallback: no variation in g1, return neutral ratio
        return 1.0f;
    }

    float r = static_cast<float>(numerator / denominator);
    return r;
}

void PreProcesser::preprocessFrame(Mat& src, Mat& dst){
    GpuMat d_blurred;
    Mat blurred, mask;

    cv::cvtColor(src, this->h_frame, cv::COLOR_BGR2GRAY);
    float r = (this->h_prevFrame.empty())? 1 : this->normalizeBrightness();
    cv::Mat img2f; 
    this->h_frame.convertTo(img2f, CV_32F);
    img2f = img2f / r;
    img2f.convertTo(this->h_frame, CV_8U);
    
    // CPU cblurring version of frame to compute the mask
    cv::GaussianBlur(this->h_frame, blurred, Size(3,3), 0);
    d_blurred.upload(blurred);
    
    // Creation of updated mask for background subtraction
    pBackSub->apply(d_blurred, d_mask);
    
    // Remove shadows (keep only 0/255)
    cv::cuda::threshold(d_mask, d_mask, 200, 255, THRESH_BINARY);
   
    this->d_mask.download(mask);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(3,3)));

    cv::morphologyEx(mask, mask, cv::MORPH_DILATE,
                    cv::getStructuringElement(cv::MORPH_RECT, Size(3,5)));

    // Upload back to GPU for masking
    this->d_mask.upload(mask);

    // Save already normalized current frame as prevFrame for next iteration
    this->h_frame.copyTo(this->h_prevFrame);
    this->h_frame.copyTo(dst);
}

GpuMat PreProcesser::getMask(){
    return this->d_mask;
}