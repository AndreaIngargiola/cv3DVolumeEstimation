#include <gtest/gtest.h>
#include <customMath.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vision.hpp>
#include <filesystem>
#include <fstream>
#include <clustering.hpp>
#include <segmentation.hpp>

TEST(ClusteringTest, YOLOtest) {
    using namespace cv;
    using namespace cv::dnn;
    using namespace std;

    std::string modelPath = "../../data/yolov5s.onnx";
    std::string classFile = "../../data/coco.names";
    std::string imagePath = "../../data/test_YOLO.jpg"; // change to your image

    // Load image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Cannot load image: " << imagePath << std::endl;
    }

    YOLOHelper yh = YOLOHelper(modelPath, image.size(), cv::Size(640,640), 0.5f, 0.45f);
    
    Mat frameToShow;
    image.copyTo(frameToShow);

    std::pair<std::vector<cv::Rect>, std::vector<cv::Rect>> boundingBoxes = yh.getBBOfPeople(image);
    
    // Save results in an image
    cv::RNG rng(cv::getTickCount());
    cv::Scalar brightColor(
        50 + rng.uniform(0, 206),
        50 + rng.uniform(0, 206),
        50 + rng.uniform(0, 206)
    );

    for (cv::Rect box : boundingBoxes.first) {
        rectangle(frameToShow, box, brightColor, 2);
    }

    cv::imwrite("testYOLO.png", frameToShow);
}

cv::Scalar getColorByClass(int classId) {
    switch(classId) {
            case -2: {
                return cv::Scalar(0,0,0); //TO CLUSTERIZE -> BLACK
                break;
            }
            case -1: {
                return cv::Scalar(255,255,255); //TO DELETE -> WHITE
                break;
            }
            
            case 0: {
                return cv::Scalar(0,0,255);  //CLASS 0 -> RED
                break;
            }

            case 1: {
                return cv::Scalar(0,255,0); //CLASS 1 -> GREEN
                break;
            }

            default: {
                return cv::Scalar(255,0,0); //CLASS 2...K -> BLUE
            }
        }
}

TEST(ClusteringTest, KMeansTest) {
    using namespace cv;
    using namespace cv::dnn;
    using namespace std;

    std::string modelPath = "../../data/yolov5s.onnx";
    std::string classFile = "../../data/coco.names";
    std::string imagePath = "../../data/testKMeans.png"; // change to your image

    // Load image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Cannot load image: " << imagePath << std::endl;
    }

    YOLOHelper yh = YOLOHelper(modelPath, image.size(), cv::Size(640,640), 0.65f, 0.45f);
    Clusterer cl = Clusterer(yh, 1e-3f, 30);
    
    cuda::GpuMat d_mask, d_status;
    KPExtractor kpe = KPExtractor(1, d_mask, d_status);
    
    Mat frameToShowUnc, frameToShowCl;
    image.copyTo(frameToShowUnc);
    image.copyTo(frameToShowCl);

    Mat grey;
    cuda::GpuMat d_image;
    cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
    d_image.upload(grey);
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
    cv::cuda::GpuMat unclusteredKp;
    kpDetector->detect(d_image, unclusteredKp);
    

    cl.clusterize(image, unclusteredKp);

    // Show results unclustered
    Mat h_unclusteredKp;
    unclusteredKp.download(h_unclusteredKp);
    
    for (int j = 0; j < h_unclusteredKp.cols; j++) {
        cv::Point2f pt = h_unclusteredKp.at<cv::Point2f>(j);
        cv::circle(frameToShowUnc, pt, 6, cv::Scalar(255,0,0), -1);
    }

    // Show results clustered (features initialization)
    thrust::device_vector<DataPoint> d_keypoints = cl.getDatapoints();
    std::vector<DataPoint> h_kp(d_keypoints.size());
    thrust::copy(d_keypoints.begin(), d_keypoints.end(), h_kp.begin());

    for(DataPoint dp : h_kp) {
        cv::Point2i pt = Point2i(dp.features[0] * image.size().width, dp.features[1] * image.size().height);
        cv::Scalar color = getColorByClass(dp.classId);
        cv::circle(frameToShowCl, pt, 6, color, -1); 
    }


    thrust::device_vector<Centroid> d_centroids = cl.getCentroids();
    std::vector<Centroid> h_cent(d_centroids.size());
    thrust::copy(d_centroids.begin(), d_centroids.end(), h_cent.begin());
    
    for(Centroid cent : h_cent) {
        cv::Point2i pt = Point2i(cent.x * image.size().width, cent.y * image.size().height);
        cv::Scalar color = getColorByClass(cent.classId);
        cv::rectangle(frameToShowCl, pt, Point2i(pt.x + 15, pt.y + 15), color, 6); 
    }

    cv::imwrite("testCL_unc.png", frameToShowUnc);
    cv::imwrite("testCL_cl.png", frameToShowCl);
}

TEST(ClusteringTest, KMeansVideoTest) {
    using namespace cv;
    VideoCapture cap("../../data/video/video.mp4");
    ASSERT_TRUE(cap.isOpened()) << "Video could not be opened!";

    std::string modelPath = "../../data/yolov5s.onnx";
    std::string classFile = "../../data/coco.names";
   

    // Load image
    cv::Mat frame;
    cap.read(frame);
  
    int fps = 30;
    YOLOHelper yh = YOLOHelper(modelPath, frame.size(), cv::Size(640,640), 0.5f, 0.20f);
    Clusterer cl = Clusterer(yh, 1e-3f, fps);
    
    cuda::GpuMat d_mask, d_cumulativeStatus;
    KPExtractor kpe = KPExtractor(fps, d_mask, d_cumulativeStatus);
    
    Mat frameToShow;
    frame.copyTo(frameToShow);
    int width = frame.size[1], height = frame.size[0];
    

    // Define the codec and create VideoWriter
    VideoWriter writer("KMeans.avi",
                       VideoWriter::fourcc('M','J','P','G'),
                       30,
                       Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Could not open the output video file" << std::endl;
    }

    cuda::GpuMat d_keypoints;
    Mat keypoints, status;
    cv::Scalar color;
    int i = 0;
    thrust::device_vector<Centroid> d_centroids;
    thrust::host_vector<Centroid> h_centroids;
    thrust::device_vector<DataPoint> d_dataPoints;
    thrust::host_vector<DataPoint> h_dataPoints;
    std::pair<std::vector<cv::Rect>, std::vector<cv::Rect>> boxes;

    while(true){
        if (!cap.read(frame) || frame.empty() || i >= 1400) break;
        if(i < 240) {
            i++;
            continue;
        }

        frame.copyTo(frameToShow);
        d_keypoints = kpe.getUnclusteredKeypoints(frame);
        d_cumulativeStatus.download(status);
        d_keypoints.download(keypoints);

        if(i % fps == 0) {
            cl.clusterize(frame, d_keypoints);
            boxes = cl.getBoxes();
        } else {
            cv::cuda::GpuMat d_frame;
            d_frame.upload(frame);
            cl.inheritClusters(d_cumulativeStatus, d_keypoints, d_frame);
        }
        h_centroids = cl.getCentroids();
        h_dataPoints = cl.getDatapoints();

        // Draw corners
        for (int j = 0; j < keypoints.cols; j++) {
            cv::Point2f pt = keypoints.at<Point2f>(0,j);
            DataPoint dp = h_dataPoints[j];
            
            cv::Scalar color = getColorByClass(dp.classId);
            cv::circle(frameToShow, pt, 6, color, -1);
        }

        //draw centroids
        for (Centroid c : h_centroids) {
            cv::Point2i pt = Point2i(c.x * frame.size().width, c.y * frame.size().height);
            cv::Scalar color = getColorByClass(c.classId);
            cv::rectangle(frameToShow, pt, Point2i(pt.x + 15, pt.y + 15), color, 6); 
        }

        for(cv::Rect box : boxes.first){
            cv::rectangle(frameToShow, Point2i(box.x, box.y), Point2i(box.x + box.width, box.y + box.height), cv::Scalar(0,0,0), 6); 
        }
       
        // Write the frame to video
        writer.write(frameToShow);
        i++;
    }

    writer.release();
}

#include <filesystem>
#include <regex>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <utility>

namespace fs = std::filesystem;

std::vector<std::string> getVideoFrames(const std::string& folder, int videoId, int maxFrames = -1)
{
    std::vector<std::pair<int, std::string>> frames;
    std::regex pattern(R"(rgb_(\d{5})_(\d+)\.jpg)");
    std::smatch match;

    for (const auto& entry : fs::directory_iterator(folder)) {
        std::string filename = entry.path().filename().string();
        if (std::regex_match(filename, match, pattern)) {
            int frameIdx = std::stoi(match[1]);
            int vidId = std::stoi(match[2]);
            //std::cout << vidId << std::endl;
            if (vidId == videoId && frameIdx <= maxFrames) {
                //std::cout << filename << std::endl;
                frames.emplace_back(frameIdx, entry.path().string());
                if (maxFrames > 0 && frames.size() >= (size_t)maxFrames)
                    break;  // Stop early
            }
        }
    }

    // Sort by frame index
    std::sort(frames.begin(), frames.end(),
              [](auto& a, auto& b){ return a.first < b.first; });
    std::cout << "HERE " << frames.size() << std::endl;
    // Extract sorted file paths
    std::vector<std::string> paths;
    for (auto& [idx, path] : frames)
        paths.push_back(path);

    return paths;
}

TEST(ClusteringTest, EMVideoTestFromDS) {
    using namespace cv;
    using namespace std;

    std::string modelPath = "../../data/crowdhuman_yolov5m-simplified.onnx";
    std::string folder = "../../data/industry_safety_0/";
    int videoId = 1;
    int maxFrames = 500;
    auto framePaths = getVideoFrames(folder, videoId, maxFrames);
    int i = 0;

    // Load image
    cv::Mat frame;
    frame = cv::imread(framePaths[i]);
  
    int fps = 15;
    YOLOHelper yh = YOLOHelper(modelPath, frame.size(), cv::Size(640,640), 0.50f, 0.30f);

    std::string calPath = "../../data/calibrations_dataset/industry_safety/calibrations.json";
    Mat img = imread("../../data/industry_safety_0/rgb_00000_1.jpg");

    // Define Calibrator and Homographer
    Calibrator c(calPath, cv::Size(9,6), 1, 1);
    Mat K = c.getK();
    customMath::flipImageOrigin(K, img.size().width, img.size().height);
    Homographer hom(K, c.getR(), c.gett(), 0.1, 20);

    Mat H = hom.getGoundPlane();
    Mat P = hom.getP();

    int zHalfPerson = 900;
    hom.computeHomographyForPlaneZ(zHalfPerson);
    cv::Mat halfPersonMat = hom.getPlane(zHalfPerson);

    EMClusterer cl = EMClusterer(yh, halfPersonMat, zHalfPerson, P);
    
    cuda::GpuMat d_mask, d_cumulativeStatus;
    KPExtractor kpe = KPExtractor(fps, d_mask, d_cumulativeStatus);
    
    Mat frameToShow, mask;
    frame.copyTo(frameToShow);
    int width = frame.size[1], height = frame.size[0];
    

    // Define the codec and create VideoWriter
    VideoWriter writer("clusteringTest/datasetTest.avi",
                       VideoWriter::fourcc('M','J','P','G'),
                       fps,
                       Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Could not open the output video file" << std::endl;
    }

    // Define the codec and create VideoWriter
    VideoWriter maskWriter("clusteringTest/datasetTestMask.avi",
                       VideoWriter::fourcc('M','J','P','G'),
                       fps,
                       Size(width, height));

    if (!maskWriter.isOpened()) {
        std::cerr << "Could not open the output video file" << std::endl;
    }

    cuda::GpuMat d_keypoints;
    Mat keypoints, status;
    cv::Scalar color;
    thrust::device_vector<Centroid> d_centroids;
    thrust::host_vector<Centroid> h_centroids;
    thrust::device_vector<DataPoint> d_dataPoints;
    thrust::host_vector<DataPoint> h_dataPoints;
    std::pair<std::vector<cv::Rect>, std::vector<cv::Rect>> boxes;
    thrust::host_vector<Cluster> h_clusters;

    while(true){
        if (i >= maxFrames) break;
        if(i < 300){
            frame = cv::imread(framePaths[i]);
            d_keypoints = kpe.getUnclusteredKeypoints(frame);
            i++;
            continue;
        }
        frame = cv::imread(framePaths[i]);
        frame.copyTo(frameToShow);
        d_keypoints = kpe.getUnclusteredKeypoints(frame);
        d_cumulativeStatus.download(status);
        d_keypoints.download(keypoints);

        if(i % fps == 0 && !d_keypoints.empty()) {
            h_clusters = cl.clusterize(d_keypoints, frame);
        }
        cv::cuda::GpuMat d_frame;
        d_frame.upload(frame);
        
        
        // Draw corners
        for (int j = 0; j < keypoints.cols; j++) {
            cv::Point2f pt = keypoints.at<Point2f>(0,j);
            cv::circle(frameToShow, pt, 2, cv::Scalar(255,0,0), -1);
        }

        // draw ellipses
        for(Cluster c: h_clusters){
            cv::ellipse(
                frameToShow,
                cv::Point(frame.size().width - c.mu.x, frame.size().height - c.mu.y),
                cv::Size(c.ew, c.eh),
                0.0,        // no rotation
                0.0,
                360.0,
                cv::Scalar(161, 0, 244),  // fuxia
                2
            );
        }
        // Write the frame to video
        writer.write(frameToShow);

        // Write mask to video
        d_mask.download(mask);
        cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
        maskWriter.write(mask);

        i++;
    }
    writer.release();
}

TEST(ClusteringTest, KMeansVideoTestFromDS) {
    using namespace cv;

    std::string modelPath = "../../data/crowdhuman_yolov5m-simplified.onnx";
    std::string folder = "../../data/industry_safety_0/";
    int videoId = 1;
    int maxFrames = 400;
    auto framePaths = getVideoFrames(folder, videoId, maxFrames);
    int i = 0;

    // Load image
    cv::Mat frame;
    frame = cv::imread(framePaths[i]);
  
    //int fps = 15;
    YOLOHelper yh = YOLOHelper(modelPath, frame.size(), cv::Size(640,640), 0.50f, 0.30f);
    /*EMClusterer em = EMClusterer(yh, )
    
    cuda::GpuMat d_mask, d_cumulativeStatus;
    KPExtractor kpe = KPExtractor(fps, d_mask, d_cumulativeStatus);
    
    Mat frameToShow, mask;
    frame.copyTo(frameToShow);
    int width = frame.size[1], height = frame.size[0];
    

    // Define the codec and create VideoWriter
    VideoWriter writer("datasetTest.avi",
                       VideoWriter::fourcc('M','J','P','G'),
                       fps,
                       Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Could not open the output video file" << std::endl;
    }

    // Define the codec and create VideoWriter
    VideoWriter maskWriter("datasetTestMask.avi",
                       VideoWriter::fourcc('M','J','P','G'),
                       fps,
                       Size(width, height));

    if (!maskWriter.isOpened()) {
        std::cerr << "Could not open the output video file" << std::endl;
    }

    cuda::GpuMat d_keypoints;
    Mat keypoints, status;
    cv::Scalar color;
    thrust::device_vector<Centroid> d_centroids;
    thrust::host_vector<Centroid> h_centroids;
    thrust::device_vector<DataPoint> d_dataPoints;
    thrust::host_vector<DataPoint> h_dataPoints;
    std::pair<std::vector<cv::Rect>, std::vector<cv::Rect>> boxes;

    while(true){
        if (i >= maxFrames) break;
        if(i < 300){
            i++;
            continue;
        }
        std::cout<< "i: " << i << std::endl;
        frame = cv::imread(framePaths[i]);
        frame.copyTo(frameToShow);
        d_keypoints = kpe.getUnclusteredKeypoints(frame);
        d_cumulativeStatus.download(status);
        d_keypoints.download(keypoints);

        if(true){//i % fps == 0) {
            cl.clusterize(frame, d_keypoints);
            boxes = cl.getBoxes();
        } else {
            cv::cuda::GpuMat d_frame;
            d_frame.upload(frame);
            cl.inheritClusters(d_cumulativeStatus, d_keypoints, d_frame);
        }
        h_centroids = cl.getCentroids();
        h_dataPoints = cl.getDatapoints();

        // Draw corners
        for (int j = 0; j < keypoints.cols; j++) {
            cv::Point2f pt = keypoints.at<Point2f>(0,j);
            DataPoint dp = h_dataPoints[j];
            
            cv::Scalar color = getColorByClass(dp.classId);
            cv::circle(frameToShow, pt, 2, color, -1);
        }

        //draw centroids
        for (Centroid c : h_centroids) {
            cv::Point2i pt = Point2i(c.x * frame.size().width, c.y * frame.size().height);
            cv::Scalar color = getColorByClass(c.classId);
            cv::rectangle(frameToShow, pt, Point2i(pt.x + 2, pt.y + 2), color, 1); 
        }

        for(cv::Rect box : boxes.first){
            cv::rectangle(frameToShow, Point2i(box.x, box.y), Point2i(box.x + box.width, box.y + box.height), cv::Scalar(0,0,0), 1); 
        }

        for(cv::Rect box : boxes.second){
            cv::rectangle(frameToShow, Point2i(box.x, box.y), Point2i(box.x + box.width, box.y + box.height), cv::Scalar(255,255,255), 1); 
        }

        // Write the frame to video
        writer.write(frameToShow);

        // Write mask to video
        d_mask.download(mask);
        cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
        maskWriter.write(mask);

        i++;
    }

    writer.release();*/
}
    