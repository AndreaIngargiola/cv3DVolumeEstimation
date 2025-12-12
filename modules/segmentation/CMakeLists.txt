add_library(segmentation_module 
    keypointsExtraction.cu
)

target_include_directories(segmentation_module PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(segmentation_module
    ${OpenCV_LIBS}
)
