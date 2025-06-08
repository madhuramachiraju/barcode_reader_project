cmake_minimum_required(VERSION 3.10)
project(barcode_reader)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# ZXing
set(ZXING_INCLUDE_DIRS /usr/local/include)
set(ZXING_LIBRARY_DIRS /usr/local/lib)
set(ZXING_LIBRARIES ZXing)

# libdmtx
find_library(DMTX_LIBRARY dmtx REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ZXING_INCLUDE_DIRS}
    /opt/homebrew/Cellar/opencv/4.11.0_1/include/opencv4
    /opt/homebrew/Cellar/libdmtx/0.7.8/include
    /usr/local/include
)

# Link directories
link_directories(
    ${OpenCV_LIBRARY_DIRS}
    ${ZXING_LIBRARY_DIRS}
    /opt/homebrew/Cellar/opencv/4.11.0_1/lib
    /opt/homebrew/Cellar/libdmtx/0.7.8/lib
    /usr/local/lib
)

# Add the shared library
add_library(barcode_scanner_lib SHARED barcode_scanner_lib.cpp)

# Link the shared library with necessary libraries
target_link_libraries(barcode_scanner_lib
    ${OpenCV_LIBS}
    ${ZXING_LIBRARIES}
    ${DMTX_LIBRARY}
)

# Build target
add_executable(barcode_reader scan_main.cpp)

# Link executable with the shared library
target_link_libraries(barcode_reader
    barcode_scanner_lib
)

# Set RPATH
set_target_properties(barcode_reader PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN:/usr/local/lib:/opt/homebrew/Cellar/opencv/4.11.0_1/lib:/opt/homebrew/Cellar/libdmtx/0.7.8/lib:${CMAKE_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
) 