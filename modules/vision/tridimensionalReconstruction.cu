#include <customMath.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vision.hpp>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace cv::cuda;
namespace fs = std::filesystem;


TridimensionalReconstructor::TridimensionalReconstructor(Mat K, Mat R, Mat t, Homographer hom, float zDimensionInCm) 
    : K(K), R(R), t(t), 
    hom(hom), 
    zDimensionInCm(zDimensionInCm)
    {};


#include <cub/cub.cuh>
#include <cuda_runtime.h>

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

    cout << "allocated " << kp.at(0).x << endl;
    float p_low = 0.05f, p_high = 1.f - p_low;

    // 1. Find values interval of the histograms
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

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


    std::cout << "minx = " << minx << ", maxx = " << maxx << "\n";
    std::cout << "miny = " << miny << ", maxy = " << maxy << "\n";
    
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
    cudaFree(d_temp);

    // 2.2 allocation and execution of histograms calculation
    temp_bytes = max(temp_x, temp_y);
    cudaMalloc(&d_temp, temp_bytes);

    cub::DeviceHistogram::HistogramEven(
        d_temp, temp_bytes,
        thrust::make_transform_iterator(d_points, [] __device__ (DataPoint p){ return p.x; }),
        d_histx,
        nbins_x + 1,
        minx, maxx,
        N
    );

    cub::DeviceHistogram::HistogramEven(
        d_temp, temp_bytes,
        thrust::make_transform_iterator(d_points, [] __device__ (DataPoint p){ return p.y; }),
        d_histy,
        nbins_y + 1,
        miny, maxy,
        N
    );

    // 3. find what are the 5 (x_lo and y_lo) and the 95th (x_hi and y_hi) percentile values (of both x and y)
    std::vector<int> hx(nbins_x), hy(nbins_y);
    cudaMemcpy(hx.data(), d_histx, nbins_x * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hy.data(), d_histy, nbins_y * sizeof(int), cudaMemcpyDeviceToHost);

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

    cout << "original kp was " << N << " reduced to " << num_inliers << endl;
}

void TridimensionalReconstructor::computeClusterDim(vector<DataPoint> kp){
    removeOutliers(kp);
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
    int height = cv::norm(Mh - M0);
    cout << "height = " << height/10 << " cm" << endl;
    cout << "M0: " << M0 << " MH: " << Mh << endl;
}