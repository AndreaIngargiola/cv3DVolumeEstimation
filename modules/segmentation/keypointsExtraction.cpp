#include <segmentation.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>

using namespace cv;
using namespace cv::cuda;

KPExtractor::KPExtractor(const int frameSetSize, GpuMat& d_mask, GpuMat& d_cumulativeStatus) 
                        : frameSetSize(frameSetSize),
                        d_mask(d_mask),
                        d_cumulativeStatus(d_cumulativeStatus) {
}

GpuMat KPExtractor::getUnclusteredKeypoints(Mat& frame) {
    this->frame = frame;
    this->preprocessFrameAndBackSub();  // conversion to greyscale and update of background subtractor

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

void KPExtractor::preprocessFrameAndBackSub() {
    GpuMat d_blurred;
    Mat blurred, mask;

    // Save previous frame and kp set
    this->d_frame.copyTo(this->d_prevFrame);
    this->d_keypoints.copyTo(this->d_prevKeypoints);

    // CPU conversion to gray and blurring version of frame to compute the mask
    //cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    d_frame.upload(frame);

    cv::GaussianBlur(frame, blurred, Size(3,3), 0);
    d_blurred.upload(blurred);

    // Creation of updated mask for background subtraction
    pBackSub->apply(d_blurred, d_mask);
    
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    d_frame.upload(frame);
    // Remove shadows (keep only 0/255)
    cv::cuda::threshold(d_mask, d_mask, 200, 255, THRESH_BINARY);
    d_mask.download(mask);

    // CPU morphological filtering
    //cv::morphologyEx(mask, mask, cv::MORPH_ERODE,
    //            cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(3,3)));

    cv::morphologyEx(mask, mask, cv::MORPH_GRADIENT,
                    cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(5,5)));

    //cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
    //                cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(3,3)));

    // Upload back to GPU for masking
    d_mask.upload(mask);

    // Apply mask to grayscale frame
    cv::cuda::bitwise_and(d_frame, d_mask, d_frame);
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