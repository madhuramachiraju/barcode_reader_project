cmake_minimum_required(VERSION 3.10)
project(barcode_reader)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# ZBar
set(ZBAR_INCLUDE_DIR /opt/homebrew/include)
set(ZBAR_LIBRARY /opt/homebrew/lib/libzbar.dylib)
include_directories(${ZBAR_INCLUDE_DIR})
link_directories(/opt/homebrew/lib)

# ZXing
set(ZXING_INCLUDE_DIRS /usr/local/include)
set(ZXING_LIBRARY_DIRS /usr/local/lib)
set(ZXING_LIBRARIES ZXing)

# libdmtx
find_library(DMTX_LIBRARY dmtx REQUIRED)

# ZBar
find_library(ZBAR_LIBRARY zbar REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ZXING_INCLUDE_DIRS}
    /opt/homebrew/Cellar/opencv/4.11.0_1/include/opencv4
    /opt/homebrew/Cellar/libdmtx/0.7.8/include
    /usr/local/include
    /opt/homebrew/include
)

# Link directories
link_directories(
    ${OpenCV_LIBRARY_DIRS}
    ${ZXING_LIBRARY_DIRS}
    /opt/homebrew/Cellar/opencv/4.11.0_1/lib
    /opt/homebrew/Cellar/libdmtx/0.7.8/lib
    /usr/local/lib
)

add_executable(barcode_reader main.cpp)

# Link executable with the shared library
target_link_libraries(barcode_reader
    ${OpenCV_LIBS}
    ${ZBAR_LIBRARY}
    ${ZXING_LIBRARIES}
    ${DMTX_LIBRARY}
)

# Set RPATH
set_target_properties(barcode_reader PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN:/usr/local/lib:/opt/homebrew/Cellar/opencv/4.11.0_1/lib:/opt/homebrew/Cellar/libdmtx/0.7.8/lib:${CMAKE_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)