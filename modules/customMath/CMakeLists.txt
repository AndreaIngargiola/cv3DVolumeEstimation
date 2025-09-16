add_library(customMath_module 
    homogeneousConversion.cpp
    axisInversion.cpp
    projection.cpp
)

target_include_directories(customMath_module PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(customMath_module
    ${OpenCV_LIBS}
)
