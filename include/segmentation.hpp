#pragma once

#include <opencv2/core/types.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaoptflow.hpp>


class KPExtractor {
    private:
    const int frameSetSize;
    int frameCounter = 0;
    cv::cuda::GpuMat d_keypoints;
    cv::cuda::GpuMat d_prevKeypoints;
    cv::Mat frame; 
    cv::cuda::GpuMat d_frame;
    cv::cuda::GpuMat d_prevFrame;
    cv::cuda::GpuMat& d_mask;
    cv::cuda::GpuMat& d_cumulativeStatus;

    // CUDA background subtractor (MOG2)
    const cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> pBackSub =
        cv::cuda::createBackgroundSubtractorMOG2(150, 20.0, true);

    // CUDA goodFeaturesToTrack
    const cv::Ptr<cv::cuda::CornersDetector> kpDetector =
        cv::cuda::createGoodFeaturesToTrackDetector(
            CV_8UC1,   // input type (grayscale)
            1000,       // maxCorners
            0.01,      // qualityLevel
            10,        // minDistance
            3,         // blockSize
            false,     // useHarrisDetector
            0.04       // k
        );

    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lucasKanadeOpticalFlow = 
        cv::cuda::SparsePyrLKOpticalFlow::create();

    public:
    KPExtractor(const int frameSetSize, cv::cuda::GpuMat& d_mask, cv::cuda::GpuMat& d_cumulativeStatus);
    cv::cuda::GpuMat getUnclusteredKeypoints(cv::Mat& frame);

    private:
    void preprocessFrameAndBackSub();
    void findNewKeypoints();
    void trackKeypoints();
};
