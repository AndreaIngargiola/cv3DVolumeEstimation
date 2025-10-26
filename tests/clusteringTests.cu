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

    std::vector<cv::Rect> boundingBoxes = yh.getBBOfPeople(image);
    
    // Save results in an image
    cv::RNG rng(cv::getTickCount());
    cv::Scalar brightColor(
        50 + rng.uniform(0, 206),
        50 + rng.uniform(0, 206),
        50 + rng.uniform(0, 206)
    );

    for (cv::Rect box : boundingBoxes) {
        rectangle(frameToShow, box, brightColor, 2);
    }

    cv::imwrite("testYOLO.png", frameToShow);
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
        cv::Scalar color;
        switch(dp.classId) {
            case -2: {
                color = Scalar(0,0,0); //TO CLUSTERIZE -> BLACK
                break;
            }
            case -1: {
                color = Scalar(255,255,255); //TO DELETE -> WHITE
                break;
            }
            
            case 0: {
                color = Scalar(0,0,255);  //CLASS 0 -> RED
                break;
            }

            case 1: {
                color = Scalar(0,255,0); //CLASS 1 -> GREEN
                break;
            }

            default: {
                color = Scalar(255,0,0); //CLASS 2...K -> BLUE
            }
        }
        cv::circle(frameToShowCl, pt, 6, color, -1); 
    }


    thrust::device_vector<Centroid> d_centroids = cl.getCentroids();
    std::vector<Centroid> h_cent(d_centroids.size());
    thrust::copy(d_centroids.begin(), d_centroids.end(), h_cent.begin());
    
    for(Centroid cent : h_cent) {
        cv::Point2i pt = Point2i(cent.x * image.size().width, cent.y * image.size().height);
        cv::Scalar color;
        switch(cent.classId) {
            case -2: {
                color = Scalar(0,0,0); //TO CLUSTERIZE -> BLACK
                break;
            }
            case -1: {
                color = Scalar(255,255,255); //TO DELETE -> WHITE
                break;
            }
            
            case 0: {
                color = Scalar(0,0,255);  //CLASS 0 -> RED
                break;
            }

            case 1: {
                color = Scalar(0,255,0); //CLASS 1 -> GREEN
                break;
            }

            default: {
                color = Scalar(255,0,0); //CLASS 2...K -> BLUE
            }
        }
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
    std::vector<cv::Rect> boxes;

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
            
            cv::Scalar color;
            switch(dp.classId) {
                case -2: {
                    color = Scalar(0,0,0); //TO CLUSTERIZE -> BLACK
                    break;
                }
                case -1: {
                    color = Scalar(255,255,255); //TO DELETE -> WHITE
                    break;
                }
                
                case 0: {
                    color = Scalar(0,0,255);  //CLASS 0 -> RED
                    break;
                }

                case 1: {
                    color = Scalar(0,255,0); //CLASS 1 -> GREEN
                    break;
                }

                default: {
                    color = Scalar(255,0,0); //CLASS 2...K -> BLUE
                }
            }
            cv::circle(frameToShow, pt, 6, color, -1);
        }

        //draw centroids
        for (Centroid c : h_centroids) {
            cv::Point2i pt = Point2i(c.x * frame.size().width, c.y * frame.size().height);
            cv::Scalar color;
            switch(c.classId) {
                case -2: {
                    color = Scalar(0,0,0); //TO CLUSTERIZE -> BLACK
                    break;
                }
                case -1: {
                    color = Scalar(255,255,255); //TO DELETE -> WHITE
                    break;
                }
                
                case 0: {
                    color = Scalar(0,0,255);  //CLASS 0 -> RED
                    break;
                }

                case 1: {
                    color = Scalar(0,255,0); //CLASS 1 -> GREEN
                    break;
                }

                default: {
                    color = Scalar(255,0,0); //CLASS 2...K -> BLUE
                }
            }
            cv::rectangle(frameToShow, pt, Point2i(pt.x + 15, pt.y + 15), color, 6); 
        }

        for(cv::Rect box : boxes){
            cv::rectangle(frameToShow, Point2i(box.x, box.y), Point2i(box.x + box.width, box.y + box.height), cv::Scalar(0,0,0), 6); 
        }
       
        // Write the frame to video
        writer.write(frameToShow);
        i++;
    }

    writer.release();
}