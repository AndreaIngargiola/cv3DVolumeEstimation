enable_testing()

find_package(GTest REQUIRED)

add_executable(runTests
    testUtils.cpp
)

target_link_libraries(runTests
    PRIVATE
    customMath_module
    vision_module
    segmentation_module
    GTest::GTest
    GTest::Main
)

add_test(NAME RunAllTests COMMAND runTests)
