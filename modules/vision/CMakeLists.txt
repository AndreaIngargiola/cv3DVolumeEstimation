add_library(vision_module 
    calibration.cpp
    homography.cpp
    tridimensionalReconstruction.cu
)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 52)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")


include_directories(
    ${CUDA_INCLUDE_DIRS}
)

link_directories(
    ${CUDA_LIBRARY_DIRS}
)

target_include_directories(vision_module PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(vision_module
    ${OpenCV_LIBS}
)
