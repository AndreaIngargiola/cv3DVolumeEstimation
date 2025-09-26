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
    cuda::GpuMat d_keypoints;
    KPExtractor kpex = KPExtractor(fps, d_mask);
    Mat frameToShow, keypoints;
    int i = 0;
    while(true){
        if (!cap.read(frame) || frame.empty() || i >= 750) break;
        frame.copyTo(frameToShow);
        d_keypoints = kpex.getUnclusteredKeypoints(frame);

        d_keypoints.download(keypoints);

        // Draw corners
        for (int j = 0; j < keypoints.cols; j++) {
            cv::Point2f pt = keypoints.at<cv::Point2f>(j);
            if(i % fps == 0){               //if new keypoints show them red
                cv::circle(frameToShow, pt, 6, cv::Scalar(0,0,255), -1);
            } else {
                cv::circle(frameToShow, pt, 6, cv::Scalar(255,0,0), -1);
            }
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

    cuda::GpuMat d_mask;
    KPExtractor kpex = KPExtractor(fps, d_mask);
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