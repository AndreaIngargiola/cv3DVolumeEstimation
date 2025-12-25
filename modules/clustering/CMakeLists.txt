find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 52)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

add_library(clustering_module 
    YOLOHelper.cu
    kMeans.cu
    EM.cu
    tridimensionalReconstruction.cu
)

target_include_directories(clustering_module PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(clustering_module
    ${OpenCV_LIBS}
)
