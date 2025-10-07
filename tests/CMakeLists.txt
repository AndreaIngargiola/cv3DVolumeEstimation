enable_testing()

find_package(GTest REQUIRED)
find_package(CUDA REQUIRED)

add_executable(runTests
    testUtils.cpp
    segmentationTests.cpp
    clusteringTests.cu
)

include_directories(
    ${CUDA_INCLUDE_DIRS}
)

link_directories(
    ${CUDA_LIBRARY_DIRS}
)

target_link_libraries(runTests
    PRIVATE
    customMath_module
    vision_module
    segmentation_module
    clustering_module
    GTest::GTest
    GTest::Main
)

add_test(NAME RunAllTests COMMAND runTests)
