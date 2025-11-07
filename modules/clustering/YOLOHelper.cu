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

// device helper
__device__ __forceinline__ float sigmoidf_dev(float x){
    return 1.f / (1.f + expf(-x));
}

__global__
void decodeAllScalesKernel(
    // per-scale arrays (flattened pointers)
    float** d_out_ptrs,           // device pointer to array of float* (per-scale)
    int*    d_H, int* d_W,        // per-scale H and W arrays (length = nScales)
    int*    d_A,                  // per-scale anchors count (typically all 3)
    int*    d_D,                  // per-scale values per anchor (7)
    const float* d_anchors,       // anchors array flattened: [scale0_a0_w, scale0_a0_h, scale0_a1_w, ...]
    const int   maxAnchorsPerScale,
    const int   nScales,
    const int   inputWidth,       // network input size (e.g. 640)
    const float confThresh,
    Detection* d_outDetections   // preallocated device buffer (size >= totalPreds)
) {
    // mapping: blockIdx.z -> scale index
    int s = blockIdx.z; // scale index 0..nScales-1
    
    if (s >= nScales) return;

    // per-scale params
    int H = d_H[s];
    int W = d_W[s];

    int A = d_A[s];     // anchors per scale (3)
    
    int anchorIdx = threadIdx.y;
    if(anchorIdx >= A) return;

    int D = d_D[s];     // values per anchor (7)
    float stride = float(inputWidth / W); // stride (should be integer)

    // global thread index inside a scale for anchor+cell
    int matrixIdx = blockIdx.x * blockDim.x + threadIdx.x; // cells 
    int totalCells = H * W;
    if (matrixIdx >= totalCells) return;
    

    int outIdx = blockIdx.y *(blockDim.x * gridDim.x * gridDim.z) + blockIdx.z * gridDim.x * blockDim.x + matrixIdx; //global thread id in the grid
    
    int anchorId = blockIdx.y;  //matrixIdx % A;
    if (anchorId >= A) return;
    
    int cellId = matrixIdx;// / A;    // 0..(H*W-1)
    int i = cellId / W;              // row
    int j = cellId % W;              // col

    // load pointer to this scale's data
    const float* scale_ptr = d_out_ptrs[s]; // pointer to scale's base in device mem
    // layout: element index = (((a * H + i) * W + j) * D) + d
    int base = (((anchorId * H + i) * W + j) * D);

    // read raw predictions:
    float tx = scale_ptr[base + 0];
    float ty = scale_ptr[base + 1];
    float tw = scale_ptr[base + 2];
    float th = scale_ptr[base + 3];
    float to = scale_ptr[base + 4]; // objectness logit
    float c_person = scale_ptr[base + 5];
    float c_head   = scale_ptr[base + 6];

    // convert logits -> probabilities (YOLO typically uses sigmoid)
    float objProb = sigmoidf_dev(to);
    float p_person = sigmoidf_dev(c_person);
    float p_head  = sigmoidf_dev(c_head);

    // compute final per-class scores
    float score_person = objProb * p_person;
    float score_head = objProb * p_head;

    // get anchors for this scale+anchor
    // anchors are packed as: for s in [0..nScales-1]: a in [0..A-1]: {aw,ah}
    int anchor_offset = s * maxAnchorsPerScale * 2 + anchorId * 2;
    float aw = d_anchors[anchor_offset + 0];
    float ah = d_anchors[anchor_offset + 1];
   
    // decode centers and sizes (YOLOv5-like decode using sigmoid-based variant)
    float bx = ( (sigmoidf_dev(tx) * 2.f - 0.5f) + float(j) ) * stride; // center x in pixels
    float by = ( (sigmoidf_dev(ty) * 2.f - 0.5f) + float(i) ) * stride; // center y in pixels
    float bw = powf(sigmoidf_dev(tw) * 2.f, 2.f) * aw; // width in pixels (anchor in pixels)
    float bh = powf(sigmoidf_dev(th) * 2.f, 2.f) * ah; // height in pixels
   
    // check person class

    float bestScore = score_person;
    int bestClass = 0;

    if (score_head > bestScore) {
        bestScore = score_head;
        bestClass = 1;
    }

    if (bestScore >= confThresh) {
        // append detection
        Detection d;
        d.left = bx - bw * 0.5f;
        d.top = by - bh * 0.5f;
        d.w = bw;
        d.h = bh;
        d.score = bestScore;
        d.classId = bestClass;
        d_outDetections[outIdx] = d;
    }
}


void scaleCoords(const cv::Size& from, const cv::Size& to, cv::Rect& box, int padY) 
{
   
    // Only height scaling, width preserved
    float gain = (float)to.height / from.height;

    // padX always zero in your setup
    int padX = 0;

    // padY must be taken from letterbox, NOT computed again
    

    // Remove padding
    box.x = (box.x - padX);
    box.y = (box.y - padY);

    // Undo scaling
    //box.x /= gain;
    //box.y *= gain;
    //box.width  *= gain;
    //box.height /= gain;

    // Clip to original image
    box.x = std::max(0, std::min(box.x, from.width - 1));
    box.y = std::max(0, std::min(box.y, from.height - 1));
    box.width  = std::min(box.width, from.width  - box.x);
    box.height = std::min(box.height, from.height - box.y);
}

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

int flag = 1;
pair<std::vector<cv::Rect>, std::vector<cv::Rect>> YOLOHelper::getBBOfPeople(Mat& frame) {
    
    // Adapt the image to the dimensions of YOLO input, creating the blob after letterboxing
    Mat padded = this->letterbox(frame); //preserve aspect-ratio of the image while fitting to the size of YOLO input
    if(flag) {
        flag=0;
        cout << padded.type() << " "<< CV_8UC3 << endl;
    }
    Mat blob;
    blobFromImage(padded, blob, 1/255.0, this->targetSize, Scalar(), true, false);

    // Run YOLO on the selected image
    net.setInput(blob);
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    
    std::vector<int> anchors_per_scale = {3,3,3};    // number of anchors per scale (usually 3)

    // Flatten all outputs pointers to device
    std::vector<float*> h_scale_ptrs(3);
    std::vector<GpuMatND> d_outputs;
    d_outputs.resize(outputs.size());
    std::vector<int> h_H, h_W, h_A, h_D;
    std::vector<float> h_stride;

    for(size_t s=0; s<outputs.size(); ++s) {
        if (d_outputs[s].empty()) {
            vector<int> dims = {outputs[s].size[0], outputs[s].size[1], outputs[s].size[2], outputs[s].size[3], outputs[s].size[4]};
            d_outputs[s] = GpuMatND(dims, outputs[s].type());
        }
        d_outputs[s].upload(outputs[s]);
        vector<int> idx = {0,0,0};
        h_scale_ptrs[s] = reinterpret_cast<float*>(d_outputs[s].createGpuMatHeader(idx, Range(0,1), Range(0,7)).data);
        h_H.push_back(outputs[s].size[2]);  // height
        h_W.push_back(outputs[s].size[3]);  // width
        h_A.push_back(anchors_per_scale[s]);
    }

    std::vector<float> anchors = {
         30.0f,61.0f,   62.0f,45.0f,    59.0f,119.0f ,  // P4/16 -> 40x40
         116.0f,90.0f,  156.0f,198.0f,  373.0f,326.0f,  // P5/32 -> 20x20
         10.0f,13.0f,   16.0f,30.0f,    33.0f,23.0f     // P3/8  -> 80x80
    };

    std::vector<int> values_per_anchor = {7,7,7}; 
    
    // Copy to device
    thrust::device_vector<float*> d_scale_ptrs(3);
    d_scale_ptrs = h_scale_ptrs;
    thrust::device_vector<int> d_H = h_H;
    thrust::device_vector<int> d_W = h_W;
    thrust::device_vector<int> d_A = h_A;
    thrust::device_vector<int> d_D = values_per_anchor;
    thrust::device_vector<float> d_stride = h_stride;
    thrust::device_vector<float> d_anchors = anchors;

    // Prepare output
    int total_bboxes = 0;
    int maxLenght = 0;
    for(Mat m : outputs){
        maxLenght = max(maxLenght, m.size[2]); // assume square matrixes
    }
    total_bboxes = outputs.size() * maxLenght * maxLenght * anchors_per_scale[0]; // assume all scales have the same number of anchors (total__bboxes = nscale * maxLenght^2 * anchors)
    thrust::device_vector<Detection> d_detections(total_bboxes);

    // Define grid
    int threads = 256;
    int maxTotalPerScale = 0;
    for (int s = 0; s < outputs.size(); ++s) maxTotalPerScale = max(maxTotalPerScale, outputs[s].size[2] * outputs[s].size[3]);
    int blocksX = (maxTotalPerScale + threads - 1) / threads;
    dim3 gridDim(blocksX, h_A[0], outputs.size());
    dim3 blockDim(threads, 1, 1);

    decodeAllScalesKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_scale_ptrs.data()),           // device pointer to array of float* (per-scale)
        thrust::raw_pointer_cast(d_H.data()), thrust::raw_pointer_cast(d_W.data()),               // per-scale H and W arrays (length = nScales)
        thrust::raw_pointer_cast(d_A.data()),                  // per-scale anchors count (typically all 3)
        thrust::raw_pointer_cast(d_D.data()),                  // per-scale values per anchor (7)
        thrust::raw_pointer_cast(d_anchors.data()),       // anchors array flattened: [scale0_a0_w, scale0_a0_h, scale0_a1_w, ...]
        3,
        3,
        640,       // network input size (e.g. 640)
        this->confThreshold,
        thrust::raw_pointer_cast(d_detections.data())   // preallocated device buffer (size >= totalPreds)
    );
    
    cudaDeviceSynchronize();

    for(auto mat : d_outputs) mat.release();

    auto end_it = thrust::remove_if(
        d_detections.begin(), d_detections.end(),
        [] __device__ (const Detection d) {
            return d.classId < 0;
        }
    );
    d_detections.erase(end_it, d_detections.end());

    thrust::host_vector<Detection> h_detections = d_detections;
    d_detections.clear();
    d_detections.shrink_to_fit();

    std::vector<cv::Rect> boxes;
    if(h_detections.size() == 0){
        cout << "return with 0 elements " <<endl;
        return pair(boxes, boxes);
    }
    std::vector<float> scores;

    // Extract rectangles and confidences from Detections to perform NMS
 
    // Transform Detection -> cv::Rect
    boxes.resize(h_detections.size());
    std::transform(
        h_detections.begin(),
        h_detections.end(),
        boxes.begin(),
        [](const Detection& d) {
            return cv::Rect(d.left, d.top, (int)d.w, (int)d.h);
        }
    );

    // Transform Detection -> float (confidence)
    scores.resize(h_detections.size());
    std::transform(
        h_detections.begin(), 
        h_detections.end(), 
        scores.begin(), 
        [] (const Detection d) {
            return d.score;}
    );

    // Perform Non Maximum Suppression to merge overlapping bounding boxes probably belonging to the same person
    vector<int> indices;
    NMSBoxes(boxes, scores, confThreshold, nmsThreshold, indices);

    vector<Rect> finalBoxesPeople;
    for (int idx : indices) {
        if(h_detections[idx].classId != 0) continue; //keep only persons
        Rect box = boxes[idx];
        scaleCoords(targetSize, frameSize, box, this->pad_y);
        finalBoxesPeople.push_back(box);
    }
    vector<Rect> finalBoxesHeads;
    for (int idx : indices) {
        if(h_detections[idx].classId != 1) continue; //keep only heads
        Rect box = boxes[idx];
        scaleCoords(targetSize, frameSize, box, this->pad_y);
        finalBoxesHeads.push_back(box);
    }

    return std::pair(finalBoxesPeople, finalBoxesHeads);
}

Mat YOLOHelper::letterbox(const Mat& src) {
    Mat  padded;
    cv::copyMakeBorder(src, padded,
        pad_y, targetSize.height - unpaddedSize.height - pad_y,
        pad_x, targetSize.width - unpaddedSize.width - pad_x,
        BORDER_CONSTANT, Scalar(114, 114, 114));
    return padded;
}