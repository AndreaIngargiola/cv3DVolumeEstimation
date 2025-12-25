#include <customMath.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vision.hpp>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace cv::cuda;
namespace fs = std::filesystem;


TridimensionalReconstructor::TridimensionalReconstructor(Mat K, Mat R, Mat t, Homographer hom, float zDimensionInCm) 
    : K(K), R(R), t(t), zDimensionInCm(zDimensionInCm) {

    hom.computeHomographies(200, 300);
    this->zPlanes = hom.getHomographies();
};

__global__ void mark(const DataPoint* pts, int* keep, int N,
                     float xl, float xh, float yl, float yh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        keep[i] =
            (pts[i].x >= xl && pts[i].x <= xh &&
             pts[i].y >= yl && pts[i].y <= yh);
}

void removeOutliers(vector<DataPoint>& kp) {
    DataPoint* d_points;   // N points on device
    int N = kp.size();
    cudaMalloc(&d_points, N * sizeof(DataPoint));
    cudaMemcpy(d_points, kp.data(), N * sizeof(DataPoint), cudaMemcpyHostToDevice);

    float p_low = 0.05f, p_high = 1.f - p_low;

    // 1. Find values interval of the histograms
    thrust::host_vector<DataPoint> h_thrust_points = kp;
    thrust::device_vector<DataPoint> d_thrust_points = h_thrust_points;

    // 1.1 Find minx and maxx
    float miny =
        thrust::transform_reduce(
            d_thrust_points.begin(), d_thrust_points.end(),
            [] __device__ (const DataPoint& p) { return p.y; },
            FLT_MAX,
            thrust::minimum<float>()
        );

    float maxy =
        thrust::transform_reduce(
            d_thrust_points.begin(), d_thrust_points.end(),
            [] __device__ (const DataPoint& p) { return p.y; },
            FLT_MIN,
            thrust::maximum<float>()
        );

    float minx =
        thrust::transform_reduce(
            d_thrust_points.begin(), d_thrust_points.end(),
            [] __device__ (const DataPoint& p) { return p.x; },
            FLT_MAX,
            thrust::minimum<float>()
        );

    float maxx =
        thrust::transform_reduce(
            d_thrust_points.begin(), d_thrust_points.end(),
            [] __device__ (const DataPoint& p) { return p.x; },
            FLT_MIN,
            thrust::maximum<float>()
        );

    int nbins_x = int(maxx - minx);
    int nbins_y = int(maxy - miny);

    int *d_histx, *d_histy;
    cudaMalloc(&d_histx, nbins_x * sizeof(int));
    cudaMalloc(&d_histy, nbins_y * sizeof(int));
    cudaMemset(d_histx, 0, nbins_x * sizeof(int));
    cudaMemset(d_histy, 0, nbins_y * sizeof(int));

    // 2. Calc histograms via CUB
    // 2.1 planning for x and y histograms
    size_t temp_x = 0, temp_y = 0;

    cub::DeviceHistogram::HistogramEven(
        nullptr, temp_x,
        thrust::make_transform_iterator(d_points, [] __device__ (DataPoint p){ return p.x; }),
        d_histx,
        nbins_x + 1,
        minx, maxx,
        N
    );

    cub::DeviceHistogram::HistogramEven(
        nullptr, temp_y,
        thrust::make_transform_iterator(d_points, [] __device__ (DataPoint p){ return p.y; }),
        d_histy,
        nbins_y + 1,
        miny, maxy,
        N
    );

    // 2.2 allocation and execution of histograms calculation
    size_t temp_bytes = max(temp_x, temp_y);
    void* d_temp = nullptr;
    cudaMalloc(&d_temp, temp_bytes);
    vector<int> hx(nbins_x), hy(nbins_y);

    cub::DeviceHistogram::HistogramEven(
        d_temp, temp_bytes,
        thrust::make_transform_iterator(d_points, [] __device__ (DataPoint p){ return p.x; }),
        d_histx,
        nbins_x + 1,
        minx, maxx,
        N
    );
    cudaMemcpy(hx.data(), d_histx, nbins_x * sizeof(int), cudaMemcpyDeviceToHost);

    cub::DeviceHistogram::HistogramEven(
        d_temp, temp_bytes,
        thrust::make_transform_iterator(d_points, [] __device__ (DataPoint p){ return p.y; }),
        d_histy,
        nbins_y + 1,
        miny, maxy,
        N
    );
    cudaMemcpy(hy.data(), d_histy, nbins_y * sizeof(int), cudaMemcpyDeviceToHost);

    // 3. find what are the 5th (x_lo and y_lo) and the 95th (x_hi and y_hi) percentile values (of both x and y)
    int lo = int(p_low * N), hi = int(p_high * N);
    float x_lo = minx, x_hi = maxx, y_lo = miny, y_hi = maxy;

    for (int i = 0, c = 0; i < nbins_x; ++i) {
        c += hx[i];
        if (c >= lo) { x_lo = minx + i; break; }
    }
    for (int i = 0, c = 0; i < nbins_x; ++i) {
        c += hx[i];
        if (c >= hi) { x_hi = minx + i; break; }
    }
    for (int i = 0, c = 0; i < nbins_y; ++i) {
        c += hy[i];
        if (c >= lo) { y_lo = miny + i; break; }
    }
    for (int i = 0, c = 0; i < nbins_y; ++i) {
        c += hy[i];
        if (c >= hi) { y_hi = miny + i; break; }
    }

    // 4. Flag the outliers with a custom kernel
    int* d_keep;
    DataPoint* d_filtered;
    int* d_count;

    cudaMalloc(&d_keep, N * sizeof(int));
    cudaMalloc(&d_filtered, N * sizeof(DataPoint));
    cudaMalloc(&d_count, sizeof(int));

    mark<<<(N+255)/256,256>>>(d_points, d_keep, N, x_lo, x_hi, y_lo, y_hi);

    cub::DeviceSelect::Flagged(
        nullptr, temp_bytes,
        d_points, d_keep,
        d_filtered, d_count,
        N
    );
    cudaFree(d_temp);
    cudaMalloc(&d_temp, temp_bytes);

    cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        d_points, d_keep,
        d_filtered, d_count,
        N
    );
    
    int num_inliers;
    cudaMemcpy(&num_inliers, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    kp.clear();
    kp.resize(num_inliers);
    cudaMemcpy(kp.data(), d_filtered, num_inliers * sizeof(DataPoint), cudaMemcpyDeviceToHost);
    
    cudaFree(d_filtered);
    cudaFree(d_keep);
    cudaFree(d_points);
    cudaFree(d_temp);
    cudaFree(d_count);
}

void TridimensionalReconstructor::computeClusterDim(vector<DataPoint> kp, Point3d& base, Point3d& top, float& height, float& width){
    //removeOutliers(kp);
    thrust::host_vector<DataPoint> points = kp;
    thrust::device_vector<DataPoint> d_points = points;
    
    double m0_y =
        thrust::transform_reduce(
            d_points.begin(), d_points.end(),
            [] __device__ (const DataPoint& p) { return p.y; },
            FLT_MAX,
            thrust::minimum<float>()
        );

    double mh_y =
        thrust::transform_reduce(
            d_points.begin(), d_points.end(),
            [] __device__ (const DataPoint& p) { return p.y; },
            FLT_MIN,
            thrust::maximum<float>()
        );

    double sum =
        thrust::transform_reduce(
            d_points.begin(), d_points.end(),
            [] __device__ (const DataPoint& p) { return p.x; },
            0.0f,
            thrust::plus<float>()
        );

    double m_x = sum / d_points.size();

    Point2d m0(m_x, m0_y);
    Point2d mh(m_x, mh_y);
    
    // Homogeneous image points
    cv::Mat m0_h = (cv::Mat_<double>(3,1) << m0.x, m0.y, 1.0);
    cv::Mat mh_h = (cv::Mat_<double>(3,1) << mh.x, mh.y, 1.0);

    
    // Full projection matrix P_complete = K [R | t]
    cv::Mat Rt;
    cv::hconcat(R, t, Rt);          // 3x4
    cv::Mat P_complete = K * Rt;    // 3x4

    // Split P_complete
    cv::Mat P = P_complete.colRange(0, 3).clone();   // 3x3
    cv::Mat p_sg = P_complete.col(3).clone();        // 3x1

    
    // Camera center C = -P^{-1} * p_sg
    cv::Mat P_inv = P.inv();
    cv::Mat C = -P_inv * p_sg;   // 3x1

    // Backproject rays
    cv::Mat D0 = P_inv * m0_h;   // feet ray direction
    cv::Mat Dh = P_inv * mh_h;   // head ray direction

    // Intersect feet ray with ground plane Z = 0
    double lambda0 = -C.at<double>(2,0) / D0.at<double>(2,0);
    cv::Mat M0 = C + lambda0 * D0;

    // World vertical direction
    cv::Mat Dv = (cv::Mat_<double>(3,1) << 0.0, 0.0, 1.0);

    
    // Solve intersection of:
    //   Lh = C + λ Dh
    //   Lv = M0 + μ Dv
    //
    //   Dh * λ - Dv * μ = M0 - C
    cv::Mat A;
    cv::hconcat(Dh, -Dv, A);   // 3x2

    cv::Mat b = M0 - C;        // 3x1

    cv::Mat sol;
    cv::solve(A, b, sol, cv::DECOMP_SVD);

    double lambda = sol.at<double>(0,0);
    // double mu     = sol.at<double>(1,0); // not strictly needed

    
    // Head 3D position
    cv::Mat Mh = C + lambda * Dh;

    // Height
    height = cv::norm(Mh - M0) * this->zDimensionInCm;
    base = Point3d(M0.at<double>(0), M0.at<double>(1), M0.at<double>(2));
    top = Point3d(Mh.at<double>(0), Mh.at<double>(1), Mh.at<double>(2));
    //cout << "height = " << height/10 << " cm" << endl;
    //cout << "M0: " << M0 << " MH: " << Mh << endl;
}


struct Homography {
    double h[9];
};

struct ProjDist {
    double dist;
    double x, y, z;
};

struct MinMax3 {
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
};

Cuboid TridimensionalReconstructor::computeCluster3d(std::vector<DataPoint> kp){
    Point3d base;
    Point3d top;
    float height;
    float width;

    removeOutliers(kp);
    computeClusterDim(kp, base, top, height, width);

    std::vector<double> zVals;
    std::vector<Homography> Hs;

    for (const auto& kv : zPlanes) {
        zVals.push_back(static_cast<double>(kv.first));

        Homography H;
        const cv::Mat& m = kv.second.inv();
        for (int i = 0; i < 9; ++i)
            H.h[i] = m.at<double>(i / 3, i % 3);

        Hs.push_back(H);
    }

    thrust::device_vector<DataPoint> d_kp = kp;
    thrust::device_vector<double> d_zVals = zVals;
    thrust::device_vector<Homography> d_Hs = Hs;

    int N = kp.size();
    int Z = zVals.size();
    int total = N * Z;
    thrust::device_vector<ProjDist> d_out(total);

    auto projector =
        [kps   = thrust::raw_pointer_cast(d_kp.data()),
        zVals = thrust::raw_pointer_cast(d_zVals.data()),
        Hs    = thrust::raw_pointer_cast(d_Hs.data()),
        Z,
        xs = base.x, ys = base.y, z0 = base.z, z1 = top.z]
        __device__ (int idx) -> ProjDist
    {
        int i = idx / Z;
        int j = idx % Z;

        float u = kps[i].x;
        float v = kps[i].y;

        const double* H = Hs[j].h;

        double Xw = H[0]*u + H[1]*v + H[2];
        double Yw = H[3]*u + H[4]*v + H[5];
        double W  = H[6]*u + H[7]*v + H[8];

        Xw /= W;
        Yw /= W;

        double Zw = zVals[j];

        double zc = fminf(fmaxf(Zw, z0), z1);

        double dx = Xw - xs;
        double dy = Yw - ys;
        double dz = Zw - zc;

        return { sqrt(dx*dx + dy*dy + dz*dz), Xw, Yw, Zw };
    };

    thrust::transform(
        thrust::device,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(total),
        d_out.begin(),
        projector
    );

    constexpr float MIN_D = 300.0f;   // millimeters
    constexpr float eps   = 1e-4f;

    double zTop = top.z;

    auto end_it = thrust::remove_if(
        thrust::device,
        d_out.begin(),
        d_out.end(),
        [zTop] __device__ (const ProjDist& p) {
            return (p.z > zTop) && (p.dist < MIN_D - eps);
        }
    );

    d_out.erase(end_it, d_out.end());
    cout << "projections reduced from " << total;
    total = d_out.size();
    cout << " to " << total << endl;

    thrust::device_vector<int> d_keys(total);

    thrust::transform(
        thrust::device,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(total),
        d_keys.begin(),
        [Z] __device__ (int idx) {
            return idx / Z;
        }
    );

    thrust::device_vector<int> d_outKeys(N);
    thrust::device_vector<ProjDist> d_best(N);

    thrust::reduce_by_key(
        thrust::device,
        d_keys.begin(), d_keys.end(),
        d_out.begin(),
        d_outKeys.begin(),
        d_best.begin(),
        thrust::equal_to<int>(),
        [] __device__ (const ProjDist& a, const ProjDist& b) {
            return (a.dist < b.dist) ? a : b;
        }
    );

    auto make_minmax =
    [] __device__ (const ProjDist& p) {
        return MinMax3{
            p.x, p.x,
            p.y, p.y,
            p.z, p.z
        };
    };

    auto reduce_minmax =
    [] __device__ (const MinMax3& a, const MinMax3& b) {
        return MinMax3{
            fminf(a.xmin, b.xmin), fmaxf(a.xmax, b.xmax),
            fminf(a.ymin, b.ymin), fmaxf(a.ymax, b.ymax),
            fminf(a.zmin, b.zmin), fmaxf(a.zmax, b.zmax)
        };
    };

    MinMax3 init{
        DBL_MAX, -DBL_MAX,
        DBL_MAX, -DBL_MAX,
        DBL_MAX, -DBL_MAX
    };

    MinMax3 result = thrust::transform_reduce(
        thrust::device,
        d_best.begin(),
        d_best.end(),
        make_minmax,
        init,
        reduce_minmax
    );

    vector<Point3d> clusterPts;
    thrust::host_vector<ProjDist> h_best = d_best;
    for(ProjDist p : h_best) clusterPts.emplace_back(Point3d(p.x, p.y, p.z));
    
    Cuboid c;
    c.origin = Point3d(result.xmin, result.ymin, result.zmin);
    c.dx = result.xmax - result.xmin;
    c.dy = result.ymax - result.ymin;
    c.dz = result.zmax - result.zmin;
    c.pts = clusterPts;
    return c;
}

std::vector<Cuboid> TridimensionalReconstructor::get3DBoundingBoxes(cv::Mat frame, std::vector<DataPoint> kp){
    
    thrust::device_vector<DataPoint> d_points = kp;

    // 1. Remove all unclustered keypoints
    auto end_it = thrust::remove_if(
        thrust::device,
        d_points.begin(),
        d_points.end(),
        [] __device__ (const DataPoint& p) {
            return p.classId < 0;
        }
    );

    d_points.erase(end_it, d_points.end());


    // 2. Group by classId
    thrust::sort_by_key(
        thrust::device,
        d_points.begin(),                // keys: classId inside struct
        d_points.end(),
        d_points.begin(),
        [] __device__ (const DataPoint& a, const DataPoint& b) {
            return a.classId < b.classId;
        }
    );

    thrust::device_vector<int> d_keys(d_points.size());

    thrust::transform(
        thrust::device,
        d_points.begin(),
        d_points.end(),
        d_keys.begin(),
        [] __device__ (const DataPoint& p) {
            return p.classId;
        }
    );

    thrust::device_vector<int> d_uniqueKeys;
    thrust::device_vector<int> d_counts;

    d_uniqueKeys.resize(d_points.size());
    d_counts.resize(d_points.size());

    auto end = thrust::reduce_by_key(
        thrust::device,
        d_keys.begin(), d_keys.end(),
        thrust::make_constant_iterator(1),
        d_uniqueKeys.begin(),
        d_counts.begin()
    );

    thrust::host_vector<int> h_uniqueKeys = d_uniqueKeys;
    thrust::host_vector<int> h_counts = d_counts;
    thrust::host_vector<DataPoint> h_points = d_points;

    int numClasses = end.first - d_uniqueKeys.begin();
    
    std::unordered_map<int, std::vector<DataPoint>> grouped;

    size_t offset = 0;
    for (int i = 0; i < numClasses; ++i) {
        int cls = h_uniqueKeys[i];
        int cnt = h_counts[i];
        grouped[cls].assign(
            h_points.begin() + offset,
            h_points.begin() + offset + cnt
        );
        offset += cnt;
    }

    // 3. Compute 3D bounding boxes for each cluster
    std::vector<Cuboid> boundingBoxes3d;
    for(auto kv : grouped) {
        vector<DataPoint> cluster = kv.second;
        Cuboid c = this->computeCluster3d(cluster);
        c.classId = kv.first;
        boundingBoxes3d.push_back(c);
    }

    return boundingBoxes3d;
}