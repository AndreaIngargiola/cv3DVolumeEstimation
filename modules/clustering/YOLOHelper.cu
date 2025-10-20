#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <clustering.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace cv::cuda;
namespace fs = std::filesystem;

// === GPU Kernel ===
__global__
void parseRowIntoDetectionKernel(const float* src, Detection* dst,
                                    int rows, int cols, size_t step, float confThreshold,
                                int pad_x, int pad_y, float scale, int frame_w, int frame_h)
{
    // One thread per row (YOLO detection candidate)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    // Get pointer to this row (handle padding)
    const float* rowPtr = (const float*)((const unsigned char*)src + row * step);

    // Read objectness
    float obj_conf = rowPtr[4];
    if (obj_conf < confThreshold) {
        dst[row] = {0,0,0,0,0,-1};
        return;
    }

    // Find best class
    int bestClass = -1;
    float bestClassScore = 0;
    for (int i = 5; i < cols; i++) {
        float score = rowPtr[i];
        if (score > bestClassScore) {
            bestClassScore = score;
            bestClass = i - 5;
        }
    }

    float conf = obj_conf * bestClassScore;
    if (conf < confThreshold) {
        dst[row] = {0,0,0,0,0,-1};
        return;
    }

    // Fill detection struct (undo letterboxing in the meantime)
    Detection det;

    float x = ((rowPtr[0] - pad_x) / scale);
    float y = ((rowPtr[1] - pad_y) / scale);
    float width = (rowPtr[2] / scale);
    float height = (rowPtr[3] / scale);

    int left = int(x - width / 2);
    int top  = int(y - height / 2);

    // Clip to image bounds
    det.left = max(0, left);
    det.top = max(0, top);
    det.w = min((float) width, (float)(frame_w- left));
    det.h = min((float) height, (float)(frame_h - top));
   
    det.score = conf;
    det.classId = bestClass;

    dst[row] = det;
}

struct FilterPerson {
    float conf;
    __device__ bool operator()(const Detection& det) const {
        return det.score < conf || det.classId != 0;
    }
};

YOLOHelper::YOLOHelper( const std::string modelPath, 
                        const cv::Size frameSize, 
                        const cv::Size targetSize,
                        const float confThreshold,
                        const float nmsThreshold)
                            :targetSize(targetSize),
                            frameSize(frameSize),
                            confThreshold(confThreshold),
                            nmsThreshold(nmsThreshold){
    
    // Load model and set it for GPU execution
    net = readNetFromONNX(modelPath);
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);

    // Compute variables for letterboxing
    this->scale = std::min(targetSize.width / (float)frameSize.width,
                           targetSize.height / (float)frameSize.height);

    this->unpaddedSize = Size(int(round(frameSize.width * scale)), int(round(frameSize.height * scale)));

    pad_x = (targetSize.width - unpaddedSize.width) / 2;
    pad_y = (targetSize.height - unpaddedSize.height) / 2;
}

vector<Rect> YOLOHelper::getBBOfPeople(Mat& frame) {
    
    // Adapt the image to the dimensions of YOLO input, creating the blob after letterboxing
    Mat padded = this->letterbox(frame); //preserve aspect-ratio of the image while fitting to the size of YOLO input
    Mat blob;
    blobFromImage(padded, blob, 1/255.0, this->targetSize, Scalar(), true, false);

    // Run YOLO on the selected image
    net.setInput(blob);
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Reshape output to [numDetections x numFeatures] instead of [1 x numDetections x numFeatures]
    Mat outMat = outputs[0].reshape(1, outputs[0].size[1]);

    // Parse detections
    vector<Rect> boxes;
    vector<float> confidences;

    parseDetections(outMat, boxes, confidences);

    // Perform Non Maximum Suppression to merge overlapping bounding boxes probably belonging to the same person
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    vector<Rect> finalBoxes;
    for (int idx : indices) {
        finalBoxes.push_back(boxes[idx]);
    }
    return finalBoxes;
}

Mat YOLOHelper::letterbox(const Mat& d_src) {
    Mat d_resized, d_padded;
    cv::resize(d_src, d_resized, this->unpaddedSize);
    cv::copyMakeBorder(d_resized, d_padded,
        pad_y, targetSize.height - unpaddedSize.height - pad_y,
        pad_x, targetSize.width - unpaddedSize.width - pad_x,
        BORDER_CONSTANT, Scalar(114, 114, 114));
    return d_padded;
}

void YOLOHelper::parseDetections(Mat& h_out, vector<Rect>& boxes, vector<float>& confidences){
    // Convert YOLO outputs in an array of Detection to parse them
    int numDetections = h_out.rows;
    int numFeatures = h_out.cols;

    GpuMat d_out;
    d_out.upload(h_out);
   
    size_t step = d_out.step; // necessary to deal with padded GpuMat (that are not saved with continous memory in GPU)

    // Allocate device memory
    thrust::device_vector<Detection> d_detections(numDetections);
    const float* d_gpuMatPtr = reinterpret_cast<const float*>(d_out.data);

    int threads = 256;
    int blocks = (numDetections + threads - 1) / threads;

    // Perform conversion on GPU where each thread process one row of the matrix and converts it to a Detection
    parseRowIntoDetectionKernel<<<blocks, threads>>>(
        d_gpuMatPtr, 
        thrust::raw_pointer_cast(d_detections.data()),
        numDetections, 
        numFeatures, 
        step, 
        confThreshold,
        pad_x, pad_y, scale,
        this->frameSize.width, this->frameSize.height
    );
    cudaDeviceSynchronize();

    // While still on GPU, filter out all non persons (COCO class 0)
    auto end = thrust::remove_if(
        d_detections.begin(), d_detections.end(), FilterPerson{confThreshold}
    );
    d_detections.erase(end, d_detections.end());

    // Copy data (very small array of valid people Detections) to CPU
    vector<Detection> h_detections(d_detections.size());
    thrust::copy(d_detections.begin(), d_detections.end(), h_detections.begin());

    // Extract rectangles and confidences from Detections to perform NMS
    boxes.resize(d_detections.size());
 
    // Transform Detection -> cv::Rect
    std::transform(
        h_detections.begin(),
        h_detections.end(),
        boxes.begin(),
        [](const Detection& d) {
            return cv::Rect(d.left, d.top, (int)d.w, (int)d.h);
        }
    );

    // Transform Detection -> float (confidence)
    confidences.resize(h_detections.size());
    std::transform(
        h_detections.begin(), 
        h_detections.end(), 
        confidences.begin(), 
        [] (const Detection d) {
            return d.score;}
    );
}