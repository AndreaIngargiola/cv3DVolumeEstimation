#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <filesystem>
#include <clustering.hpp>
/*
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
    const float shiftThreshold, 
    const int frameSetSize):
        boxFinder(boxFinder), 
        frameSetSize(frameSetSize),
        shiftThreshold(shiftThreshold){

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
    pair<vector<Rect>, vector<Rect>> bbs = this->boxFinder.getBBOfPeople(frame);
    this->boxes = bbs.first;
    this->boxesHeads = bbs.second;
    vector<cv::Rect> boundingBoxes = boxes;
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
    float sumSquaredShift = this->updateCentroids();

    // 7. Perform K-means clustering on the non-supervised keypoints
    while (sumSquaredShift > this->shiftThreshold) {
        this->updateKeypoints();
        sumSquaredShift = this->updateCentroids();
    }

    this->isFirstIteration = true;
}

float Clusterer::updateCentroids(){
    if (!this->isFirstIteration) this->d_oldCentroids = this->d_centroids;

    thrust::device_vector<DataPoint> d_tmpKP = this->d_keypoints;
    using AccumTuple = thrust::tuple<float,float,float,float,float,float>;  // <x, y, h, s, v, countClusterMembers>

    //filter out non assigned DataPoints (de-facto filter out all non-supervised)
    auto end_it = thrust::remove_if(
        d_tmpKP.begin(), d_tmpKP.end(),
        [] __device__ (const DataPoint& p) {
            return p.classId < 0;  // remove invalid
        }
    );

    d_tmpKP.erase(end_it, d_tmpKP.end());

    // Get keys and values from DataPoints (keys = classId) to sum reduce
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

    thrust::device_vector<int> unique_keys(d_tmpKP.size());
    thrust::device_vector<AccumTuple> reduced_vals(d_tmpKP.size());
    
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    
    // Sum reduce the DataPoints array by classId
    auto new_end = thrust::reduce_by_key(
        keys.begin(),
        keys.end(),
        values.begin(),
        unique_keys.begin(),
        reduced_vals.begin(),
        thrust::equal_to<int>(),
        [] __host__ __device__ (const AccumTuple& a, const AccumTuple& b) {
            return AccumTuple{
                thrust::get<0>(a) + thrust::get<0>(b),
                thrust::get<1>(a) + thrust::get<1>(b),
                thrust::get<2>(a) + thrust::get<2>(b),
                thrust::get<3>(a) + thrust::get<3>(b),
                thrust::get<4>(a) + thrust::get<4>(b),
                thrust::get<5>(a) + thrust::get<5>(b)
            };
        }
    );

    this->k = new_end.first - unique_keys.begin();
    this->d_centroids.resize(this->k);

    // Perform the means on the summed features of each one of the k centroids
    thrust::for_each_n(
        thrust::make_counting_iterator(0),
        this->k,
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
            }
        }
    );

    if(this->isFirstIteration) {
        this->isFirstIteration = false;
        return MAXFLOAT;
    }
    // 5. Compute convergence: mean squared centroid shift
    thrust::device_vector<float> diffs(this->k);

    thrust::transform(
        this->d_centroids.begin(), this->d_centroids.end(),
        this->d_oldCentroids.begin(),
        diffs.begin(),
        [] __device__ (const Centroid& a, const Centroid& b) {
            float dx = a.x - b.x;
            float dy = a.y - b.y;
            float dh = a.h - b.h;
            float ds = a.s - b.s;
            float dv = a.v - b.v;
            return dx*dx + dy*dy + dh*dh + ds*ds + dv*dv;
        }
    );

    float sum = thrust::reduce(diffs.begin(), diffs.end(), 0.0f, thrust::plus<float>());
    return sum / this->k; // av
}

thrust::device_vector<DataPoint> Clusterer::getDatapoints() {
    return this->d_keypoints;
}

thrust::device_vector<Centroid> Clusterer::getCentroids() {
    return this->d_centroids;
}

void Clusterer::inheritClusters(GpuMat d_statusKP, GpuMat d_unclusteredKP, GpuMat d_frame) {
    const int rows = d_frame.rows;
    const int cols = d_frame.cols;
    const int step = d_frame.step;

    int kpSize = d_unclusteredKP.rows * d_unclusteredKP.cols * d_unclusteredKP.channels();
    int numKeypoints = kpSize / 2;  // since each keypoint = 2 floats (x, y)
    this->d_keypoints.resize(numKeypoints);
    
    // 4. Configure CUDA grid
    const int threadsPerBlock = 256;
    const int numBlocks = (numKeypoints + threadsPerBlock - 1) / threadsPerBlock;

    // 5. Run kernel to get the initial supervised and unsupervised DataPoints to clusterize
    parseMatIntoEnstablishedClustersKernel<<<numBlocks, threadsPerBlock>>>(
        reinterpret_cast<const float*>(d_unclusteredKP.ptr<float>()), kpSize,
        rows, cols, step,
        thrust::raw_pointer_cast(this->d_keypoints.data()),
        reinterpret_cast<const float*>(d_statusKP.ptr<float>()),
        thrust::raw_pointer_cast(this->d_keypoints.data())
    );

    cudaDeviceSynchronize();
}

struct AssignToNearestCentroid {
    const Centroid* centroids;
    int k;

    AssignToNearestCentroid(const Centroid* c, int k)
        : centroids(c), k(k) {}

    __device__ void operator()(DataPoint& p) const {
        if (p.isSupervised) return; // skip supervised points
        if (p.classId == -1) return; // skip dead keypoints

        float minDist = MAXFLOAT;
        int bestId = -1;

        for (int i = 0; i < k; ++i) {
            const Centroid& c = centroids[i];
            float dx = p.features[0] - c.x;
            float dy = p.features[1] - c.y;
            float dh = p.features[2] - c.h;
            float ds = p.features[3] - c.s;
            float dv = p.features[4] - c.v;
            float dist = dx*dx + dy*dy + dh*dh + ds*ds + dv*dv;
            if (dist < minDist) {
                minDist = dist;
                bestId = c.classId;
            }
        }
        p.classId = bestId;
    }
};

void Clusterer::updateKeypoints(){
    thrust::for_each(
        this->d_keypoints.begin(), this->d_keypoints.end(),
        AssignToNearestCentroid(
            thrust::raw_pointer_cast(this->d_centroids.data()), this->d_centroids.size())
    );
}

pair<std::vector<cv::Rect>, std::vector<cv::Rect>> Clusterer::getBoxes() {
    return pair(this->boxes, this->boxesHeads);
}
*/