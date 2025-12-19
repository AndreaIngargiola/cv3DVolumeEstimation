#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <filesystem>
#include <clustering.hpp>
#include <customMath.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace cv::cuda;
namespace fs = std::filesystem;

EMClusterer::EMClusterer(YOLOHelper yh, Mat halfPersonPlane, int halfPersonZ, Mat P)
    :boxFinder(yh), 
    halfPersonPlane(halfPersonPlane),
    halfPersonZ(halfPersonZ),
    P(P) {};


cv::Rect scaleRect(const cv::Rect& r, float s)
{
    // center coordinates
    float cx = r.x + 0.5f * r.width;
    float cy = r.y + 0.5f * r.height;

    int newW = static_cast<int>(r.width  * s);
    int newH = static_cast<int>(r.height * s);

    int newX = static_cast<int>(cx - 0.5f * newW);
    int newY = static_cast<int>(cy - 0.5f * newH);

    return cv::Rect(newX, newY, newW, newH);
}

std::pair<thrust::host_vector<Cluster>, std::vector<DataPoint>> EMClusterer::clusterize(GpuMat keypoints, Mat frame) {
    this->ew.clear();
    this->eh.clear();
    this->ellipses.clear();
    this->mu.clear();
    this->oldMu.clear();
    this->pi.clear();
    this->pts.clear();

    this->img_w = frame.size().width;
    this->img_h = frame.size().height;
    
    this->boundingBoxes = this->boxFinder.getBBOfPeople(frame);
    
    for(int i = 0; i < boundingBoxes.first.size(); i++ ){
        this->boundingBoxes.first[i] = scaleRect(this->boundingBoxes.first[i], 1.25f);
    }

    for(int i = 0; i < boundingBoxes.second.size(); i++ ){
        this->boundingBoxes.second[i] = scaleRect(this->boundingBoxes.second[i], 1.25f);
    }
    int numDetections = this->boundingBoxes.first.size();
    this->k = numDetections * 2;
    keypoints.copyTo(this->kp);

    this->initializeGaussians();

    const int N = pts.size();
    const int K = mu.size();

    // responsibilities r[n][k]
    std::vector<std::vector<float>> r(N, std::vector<float>(K));
    for(int i = 0; i < this->maxIter; ++i){
        this->oldMu = this->mu;
        
        this->EStep(r, N, K);
        this->MStep(r, N, K);

        bool converged = true;

        for (int k = 0; k < K; ++k)
        {
            double dx = mu[k].x - oldMu[k].x;
            double dy = mu[k].y - oldMu[k].y;
            double dist2 = dx*dx + dy*dy;

            if (dist2 > DELTA*DELTA)
            {
                converged = false;
                break;
            }
        }
        if (converged) break;   // EM converged early
    }

    vector<Rect> boxes;

    for (int i = 0; i < this->k; i++) {

        int x = round((img_w - mu[i].x) - ew[i]);
        int y = round((img_h - mu[i].y) - eh[i]);
        int w = round(2.0 * ew[i]);
        int h = round(2.0 * eh[i]);

        boxes.emplace_back(x, y, w, h);
    }

    // NMS parameters
    float scoreThreshold = 0.05f;   
    float nmsThreshold   = 0.5f;   // paper-like value

    std::vector<int> keptIndices;

    cv::dnn::NMSBoxes(
        boxes,
        this->pi,
        scoreThreshold,
        nmsThreshold,
        keptIndices
    );

    this->ellipses.clear();
    unordered_map<int, int> classes;
    int classId = 0;

    for(int idx : keptIndices){
        Cluster cl;
        cl.eh = this->eh[idx];
        cl.ew = this->ew[idx];
        cl.mu = this->mu[idx];
        cl.pi = this->pi[idx];
        classes.emplace(idx, classId);
        classId++;
        this->ellipses.push_back(cl);
    }

    int ptIdx = 0;
    for(int i = 0; i < this->datapoints.size(); i++) {
        if(this->datapoints[i].classId == -2) continue;
        vector<float> conf = r[ptIdx];
        ptIdx++;
        float bestScore = -1;
        int bestClass = -2;
        for(int idx : keptIndices){
            if(conf[idx] > bestScore) {
                bestScore = conf[idx];
                bestClass = classes.at(idx);
            }
        }
        this->datapoints[i].classId = bestClass;
    }
    
    return pair(this->ellipses, this->datapoints);
}

void EMClusterer::initializeGaussians(){
    
    // 1. Download keypoints from GPU
    
    cv::Mat h_kpts;
    this->kp.download(h_kpts);
    int N = h_kpts.cols;
    cout << "N: " << N << endl;
    this->datapoints.resize(N);
    int goodKPCounter = 0;

    // 1.5 Filter out non-person keypoints
    for (int i = 0; i < N; i++) {
        cv::Point2f pt = h_kpts.at<Point2f>(0,i);
        Point2f ptInRightSystem = Point2f(this->img_w - pt.x, this->img_h - pt.y);
        datapoints[i].features[0] = ptInRightSystem.x;
        datapoints[i].features[1] = ptInRightSystem.y;
        datapoints[i].classId = -2;

        if(pt.x < 0 || pt.y < 0) continue; // discard dead keypoints
        bool added = false;

        for (Rect r : this->boundingBoxes.first) {
            if(r.contains(pt)) {    // Add only keypoints that are contained in a bounding box
                pts.push_back(ptInRightSystem);
                datapoints[i].classId = -1;
                goodKPCounter++;
                added = true;
                break;
            }
        }

        if(!added) {
            for (Rect r : this->boundingBoxes.second) {
                if(r.contains(pt)) {
                    pts.push_back(ptInRightSystem);
                    datapoints[i].classId = -1;
                    goodKPCounter++;
                    break;
                }
            }
        }
    }

    N = goodKPCounter;

    // 2. Compute density of each keypoint (number of neighbors in R)
    //    As in the paper: radius R ≈ 8 px. (Fig. 5, Sec. 3.2)
    
    float R = 4.0f;
    float R2 = R * R;

    std::vector<int> density(N, 0);

    for (int i = 0; i < N; ++i) {
        float xi = pts[i].x;
        float yi = pts[i].y;

        int count = 0;

        // brute-force count (paper uses exact same logic)
        for (int j = 0; j < N; ++j) {
            float dx = pts[j].x - xi;
            float dy = pts[j].y - yi;
            if (dx*dx + dy*dy <= R2)
                count++;
        }

        density[i] = count;
    }

    // 3. Sort points by density (descending)
    
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;

    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        return density[a] > density[b];
    });

    // 4. Select initial cluster centers
    //    – pick highest-density points
    //    – enforce spacing ≥ 8 px (same R)
    //    – stop when K *2 centers found
    
    this->mu.clear();

    Mat inverseHom = this->halfPersonPlane.inv();

    //Rect halfPersonPlaneLimits(Point2i(-84000, -59200), Point2i(57200, 63600));
    Rect halfPersonPlaneLimits(Point2i(-4200, -2960), Point2i(2860, 3180)); 
    for (int t = 0; t < N && (int)this->mu.size() < this->k; ++t) {

        int id = idx[t];
        float px = pts[id].x;
        float py = pts[id].y;

        //if the centroid candidate is out of bounds in the 3d world, discard it
        if(!halfPersonPlaneLimits.contains(customMath::projectOnImgFromPlane(Point2d(px, py), inverseHom))) continue;

        // maintain minimum spacing like the paper (≈ 4px)
        bool farEnough = true;
        for (int s = 0; s < (int)this->mu.size(); ++s) {
            float dx = px - this->mu[s].x;
            float dy = py - this->mu[s].y;
            if (dx*dx + dy*dy < R2) {
                farEnough = false;
                break;
            }
        }

        if (farEnough) {
            this->mu.push_back(cv::Point2f(px, py));
        }
    }

    this->k = this->mu.size();

    
    float personAspectRatio = 0.33f;
    float priorInitialValue = 1.f / this->k;
    
    for(auto centroid: this->mu){
        Point2d centroidOnPlane = customMath::projectOnImgFromPlane(Point2d(centroid.x, centroid.y), inverseHom);
        Point3d centroidOn3d = Point3d(centroidOnPlane.x, centroidOnPlane.y, this->halfPersonZ);
        
        Point3d topOn3d = centroidOn3d + Point3d(0,0,900);
        Point2d topOnImg = customMath::projectOnImgFrom3D(topOn3d, this->P);
        
        float eh = topOnImg.y - centroid.y;
        float ew = eh * personAspectRatio;
        
        this->eh.push_back(eh);
        this->ew.push_back(ew);
        this->pi.push_back(priorInitialValue);
    }

    // If density is low or points overlap, K might not be fully reached.
    // The paper notes this is acceptable; EM will prune useless clusters.
}

 void EMClusterer::EStep(std::vector<std::vector<float>>& r, int N, int K){
    for (int n = 0; n < N; ++n)
    {
        float denom = 0.0f;

        for (int k = 0; k < K; ++k)
        {
            // Compute ellipse area A_e  (= π * eh * ew)
            float Ae = CV_PI * ew[k] * eh[k];

            // Compute Σ from c * ellipse axes:
            //      σx = c * ew[k]
            //      σy = c * eh[k]
            float sx = c * ew[k];
            float sy = c * eh[k];

            float invNorm = 1.0f / (2.0f * CV_PI * sx * sy);

            // Check if point is inside ellipse
            float dx = pts[n].x - mu[k].x;
            float dy = pts[n].y - mu[k].y;

            bool inside =
                (dx * dx) / (ew[k] * ew[k]) +
                (dy * dy) / (eh[k] * eh[k]) <= 1.0f;

            //  CASE 1: inside ellipse → uniform: m / Ae
            //  CASE 2: outside ellipse → full Gaussian
            //
            //  (EXACTLY as written in Equation (8)
            float h;
            if (inside)
            {
                h = m / Ae;
            }
            else
            {
                float ex = -0.5f * (dx * dx) / (sx * sx)
                           -0.5f * (dy * dy) / (sy * sy);
                h = std::exp(ex) * invNorm;
            }

            float val = pi[k] * h;
            r[n][k] = val;
            denom += val;
        }

        // normalize responsibilities
        if (denom > 0)
        {
            for (int k = 0; k < K; ++k)
                r[n][k] /= denom;
        }
    }
}

 void EMClusterer::MStep(std::vector<std::vector<float>>& r, int N, int K){
    for (int k = 0; k < K; ++k)
    {
        // Nk = effective cluster population
        float Nk = 0.0f;

        for (int n = 0; n < N; ++n)
            Nk += r[n][k];

        // update prior
        this->pi[k] = Nk / float(N);

        // update center μ_k (the only parameter that moves)
        float sumx = 0.0f, sumy = 0.0f;

        for (int n = 0; n < N; ++n)
        {
            sumx += r[n][k] * pts[n].x;
            sumy += r[n][k] * pts[n].y;
        }

        if (Nk > 1e-6f)
        {
            mu[k].x = sumx / Nk;
            mu[k].y = sumy / Nk;
        }
    }
}