add_library(vision_module 
    calibration.cpp
)

target_include_directories(vision_module PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(vision_module
    ${OpenCV_LIBS}
)
