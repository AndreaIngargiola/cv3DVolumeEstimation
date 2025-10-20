#include <gtest/gtest.h>
#include <customMath.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vision.hpp>
#include <filesystem>
#include <fstream>
#include <segmentation.hpp>



TEST(SegmentationTest, KPExtractor_tracking) {
    using namespace cv;
    VideoCapture cap("../../data/video/video.mp4");
    ASSERT_TRUE(cap.isOpened()) << "Video could not be opened!";

    Mat frame;
    cap.read(frame);
    int width = frame.size[1], height = frame.size[0];
    int fps = 30;

    // Define the codec and create VideoWriter
    VideoWriter writer("KPExtractorOutput.avi",
                       VideoWriter::fourcc('M','J','P','G'),
                       fps,
                       Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Could not open the output video file" << std::endl;
    }

    cuda::GpuMat d_mask;
    cuda::GpuMat d_keypoints, d_cumulativeStatus;
    KPExtractor kpex = KPExtractor(fps, d_mask, d_cumulativeStatus);
    Mat frameToShow, keypoints, status;
    cv::Scalar color;
    int i = 0;

    while(true){
        if (!cap.read(frame) || frame.empty() || i >= 750) break;
        frame.copyTo(frameToShow);
        d_keypoints = kpex.getUnclusteredKeypoints(frame);
        d_cumulativeStatus.download(status);
        d_keypoints.download(keypoints);
        if(i == 700) std::cout << status.row(0) << std::endl;

        // Draw corners
        for (int j = 0; j < keypoints.cols; j++) {
            cv::Point2f pt = keypoints.at<cv::Point2f>(j);
            if(i % fps == 0){               //if new keypoints show them blue
                color = cv::Scalar(255,0,0);
            } else {
                if(status.at<float>(0,j) > 0.0f) color = cv::Scalar(0,255,0); //if tracked by lukas kanade, show them green
                else  color = cv::Scalar(0,0,255); //if lost, show them red
            }
            cv::circle(frameToShow, pt, 6, color, -1);
        }

        // Write the frame to video
        writer.write(frameToShow);
        i++;
    }

    writer.release();
}

TEST(SegmentationTest, KPExtractor_backsub) {
    using namespace cv;

    VideoCapture cap("../../data/video/video.mp4");
    Mat frame, mask;
    cap.read(frame);
    int width = frame.size[1], height = frame.size[0];
    int fps = 30;

    // Define the codec and create VideoWriter
    VideoWriter writer("KPExtractorMask.avi",
                        VideoWriter::fourcc('M','J','P','G'),
                        fps,
                        Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Could not open the output video file" << std::endl;
    }

    cuda::GpuMat d_mask, d_cumulativeStatus;
    KPExtractor kpex = KPExtractor(fps, d_mask, d_cumulativeStatus);
    Mat frameToShow;
    int i = 0;
    while(true){
        if (!cap.read(frame) || frame.empty() || i >= 750) break;
        kpex.getUnclusteredKeypoints(frame);

        // Get the foreground mask and write it to video
        d_mask.download(mask);

        // For video writing, ensure it’s 3-channel
        cv::cvtColor(mask, frameToShow, cv::COLOR_GRAY2BGR);
        if(i == 10) std::cout << "frameToShow size: " << frameToShow.size() << std::endl;
        writer.write(frameToShow);
        i++;
    }

    writer.release();
}