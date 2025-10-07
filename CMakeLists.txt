cmake_minimum_required(VERSION 3.18)
project(cv3DVolumeEstimation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

find_package(CUDA REQUIRED)

# Add submodules
add_subdirectory(modules/customMath)
add_subdirectory(modules/vision)
add_subdirectory(modules/segmentation)
add_subdirectory(modules/clustering)
add_subdirectory(tests)

add_executable(my_app
    src/main.cu
    src/utils.cpp
)

target_link_libraries(my_app
    ${OpenCV_LIBS}
    cuda
    cudart
    customMath_module
    vision_module
    segmentation_module
    clustering_module
)

target_include_directories(my_app PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    customMath_module
    vision_module
    segmentation_module
    clustering_module
)

set_target_properties(my_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
