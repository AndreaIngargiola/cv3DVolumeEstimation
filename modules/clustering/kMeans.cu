#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <filesystem>
#include <clustering.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace cv::cuda;
namespace fs = std::filesystem;

// === GPU Kernel ===
__global__
void parseMatIntoEnstablishedClustersKernel(
    const float* keypoints, int kpSize,
    int rows, int cols, int step,
    DataPoint* src,
    const float* status,
    DataPoint* dst)
{
    // One thread per keypoint (a keypoint is represented as two consecutive values of the GpuMat keypoints)
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int elem = threadId * 2;
    if (elem >= kpSize) return;

    // If status[elem] == 0, the keypoint is lost in the tracking, so its DataPoint must die.
    DataPoint dp;
    if(status[elem] == 0) {
        dp.classId = -1;
        dst[elem] = dp;
        return;
    }

    const float x = keypoints[elem];
    const float y = keypoints[elem + 1];

    // Get and normalize keypoint position
    dp.features[0] = x / cols;   // x
    dp.features[1] = y / rows;   // y

    dp.classId = src[elem].classId; //inherit class
    dst[elem] = dp;
    return;
}

__global__
void parseMatIntoKeypointsKernel(
    const float* keypoints, int kpSize, 
    const uchar3* frame, int rows, int cols, int step, 
    Detection* boundingBoxes, int k,
    DataPoint* dst)
{
    // One thread per keypoint (a keypoint is represented as two consecutive values of the GpuMat keypoints)
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int elem = threadId * 2;
    if (elem >= kpSize) return;

    // Get pointer to this row (handle padding)
    //const float* kpPtr = (const float*)((const unsigned char*)keypoints + elem);

    DataPoint dp;
    const float x = keypoints[elem];//kpPtr[0];
    const float y = keypoints[elem + 1];//kpPtr[1];

    // Get and normalize keypoint position
    dp.features[0] = x / cols;   // x
    dp.features[1] = y / rows;   // y

    // Get and normalize HSV values from the keypoint position in the frame
    const int x_img = round(x);
    const int y_img = round(y);

    const uchar3* row = (const uchar3*)((const uchar*)((const void*)frame) + y_img * step);
    uchar3 pix = row[x_img];

    dp.features[2] = pix.x / 180.0f;   // H
    dp.features[3] = pix.y / 255.0f;   // S
    dp.features[4] = pix.z / 255.0f;   // V

    // Check if a keypoint belong to one, more or no bounding boxes  
    
    // classId = -1 => no class assigned, to remove
    // classId = {0...k} => one class assigned
    // classId = -2 => multiple class assigned, to cluster
    dp.classId = -1;
    Detection bb;
    for(int i = 0; i < k; ++i) {
        bb = boundingBoxes[i];
        if( x >= bb.left && 
            x <= bb.left + bb.w &&
            y >= bb.top &&
            y <= bb.top + bb.h) 
        {
            if(dp.classId == -1) {
                dp.classId = bb.classId;
                dp.isSupervised = 1; //true
            } else {
                dp.classId = -2;
                dp.isSupervised = 0; //false
            }
        }
    }

    dst[threadId] = dp;
}

Clusterer::Clusterer(
    YOLOHelper& boxFinder, 
    const float threshold, 
    const int frameSetSize):
        boxFinder(boxFinder), 
        frameSetSize(frameSetSize){

}

void Clusterer::clusterize(cv::Mat frame, cv::cuda::GpuMat d_unclusteredKP) {
    // 1. Convert frame (BGR) -> HSV on GPU
    cv::cuda::GpuMat d_bgr, d_hsv;
    d_bgr.upload(frame);
    cv::cuda::cvtColor(d_bgr, d_hsv, cv::COLOR_BGR2HSV);

    // Ensure data is continuous and 8UC3
    CV_Assert(d_hsv.type() == CV_8UC3);
    const int rows = d_hsv.rows;
    const int cols = d_hsv.cols;
    const int step = d_hsv.step;

    // 2. Prepare YOLO bounding boxes on device
    vector<cv::Rect> boundingBoxes = this->boxFinder.getBBOfPeople(frame);
    thrust::host_vector<Detection> h_dets;
    
    for (int i = 0; i < boundingBoxes.size(); ++i) {
        Rect bb = boundingBoxes[i];

        Detection det;
        det.top = bb.y;
        det.left = bb.x;
        det.w = bb.width;
        det.h = bb.height;
        det.classId = i; // classId = nth cluster
        det.score = -1;  //unused

        h_dets.push_back(det);
    }

    int k_boxes = static_cast<int>(h_dets.size());
    thrust::device_vector<Detection> d_dets = h_dets;

    // 3. Allocate output keypoints vector
    int kpSize = d_unclusteredKP.rows * d_unclusteredKP.cols * d_unclusteredKP.channels();
    int numKeypoints = kpSize / 2;  // since each keypoint = 2 floats (x, y)
    this->d_keypoints.resize(numKeypoints);

     // 4. Configure CUDA grid
    const int threadsPerBlock = 256;
    const int numBlocks = (numKeypoints + threadsPerBlock - 1) / threadsPerBlock;

    // 5. Run kernel to get the initial supervised and unsupervised DataPoints to clusterize
    parseMatIntoKeypointsKernel<<<numBlocks, threadsPerBlock>>>(
        reinterpret_cast<const float*>(d_unclusteredKP.ptr<float>()), kpSize,
        reinterpret_cast<const uchar3*>(d_hsv.ptr<uchar3>()), rows, cols, step,
        thrust::raw_pointer_cast(d_dets.data()), k_boxes,
        thrust::raw_pointer_cast(d_keypoints.data())
    );

    cudaDeviceSynchronize();

    // 6. Compute starting centroids
    thrust::device_vector<DataPoint> d_tmpKP = this->d_keypoints;
    using AccumTuple = thrust::tuple<float,float,float,float,float,float>;  // <x, y, h, s, v, countClusterMembers>

    auto end_it = thrust::remove_if(
        d_tmpKP.begin(), d_tmpKP.end(),
        [] __device__ (const DataPoint& p) {
            return p.classId < 0;  // remove invalid
        }
    );

    d_tmpKP.erase(end_it, d_tmpKP.end());


    thrust::device_vector<int> keys(d_tmpKP.size());
    thrust::device_vector<AccumTuple> values(d_tmpKP.size());

    thrust::transform(
        d_tmpKP.begin(), d_tmpKP.end(),
        keys.begin(),
        [] __device__ (const DataPoint& p) { return p.classId; }
    );

    thrust::transform(
        d_tmpKP.begin(), d_tmpKP.end(),
        values.begin(),
        [] __device__ (const DataPoint& p) {
            return AccumTuple{
                p.features[0], p.features[1], p.features[2],
                p.features[3], p.features[4], 1.0f
            };
        }
    );

    auto reduce_op = [] __host__ __device__ (const AccumTuple& a, const AccumTuple& b) {
        return AccumTuple{
            thrust::get<0>(a) + thrust::get<0>(b),
            thrust::get<1>(a) + thrust::get<1>(b),
            thrust::get<2>(a) + thrust::get<2>(b),
            thrust::get<3>(a) + thrust::get<3>(b),
            thrust::get<4>(a) + thrust::get<4>(b),
            thrust::get<5>(a) + thrust::get<5>(b)
        };
    };

    thrust::device_vector<int> unique_keys(d_tmpKP.size());
    thrust::device_vector<AccumTuple> reduced_vals(d_tmpKP.size());
    
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    
    auto new_end = thrust::reduce_by_key(
        keys.begin(),
        keys.end(),
        values.begin(),
        unique_keys.begin(),
        reduced_vals.begin(),
        thrust::equal_to<int>(),
        reduce_op
    );
/*
    thrust::host_vector<int> h_uniq = unique_keys;
    for (int i = 0; i < h_uniq.size(); i++){
        std::cout << h_uniq[i] << " ";
    }
    std::cout << std::endl;
    */

    int n_groups = new_end.first - unique_keys.begin();
    cout << "N_GROUPS =  " << n_groups << endl;
    this->d_centroids.resize(n_groups);

    thrust::for_each_n(
        thrust::make_counting_iterator(0),
        n_groups,
        [centroids_ptr = thrust::raw_pointer_cast(d_centroids.data()),
        keys_ptr = thrust::raw_pointer_cast(unique_keys.data()),
        sums_ptr = thrust::raw_pointer_cast(reduced_vals.data())] __device__ (int i)
        {
            int cid = keys_ptr[i];
            if (cid < 0) return;
            auto s = sums_ptr[i];
            float count = thrust::get<5>(s);
            if (count > 0) {
                centroids_ptr[cid].x = thrust::get<0>(s) / count;
                centroids_ptr[cid].y = thrust::get<1>(s) / count;
                centroids_ptr[cid].h = thrust::get<2>(s) / count;
                centroids_ptr[cid].s = thrust::get<3>(s) / count;
                centroids_ptr[cid].v = thrust::get<4>(s) / count;
                centroids_ptr[cid].classId = i;
               
                centroids_ptr[cid].lastUpdateDistance = MAXFLOAT;
            }
        }
    );
}

thrust::device_vector<DataPoint> Clusterer::getDatapoints() {
    return this->d_keypoints;
}

thrust::device_vector<Centroid> Clusterer::getCentroids() {
    return this->d_centroids;
}

void Clusterer::inheritClusters(GpuMat d_statusKP, GpuMat d_unclusteredKP) {

}