FROM ubuntu:22.04

# Install all dependencies in a single layer
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libdmtx-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source files
COPY barcode_scanner_lib.cpp .
COPY barcode_scanner_lib.h .

# Build the shared library directly
RUN g++ -std=c++17 -fPIC -I. -shared barcode_scanner_lib.cpp \
    -o libbarcode_reader.so \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -ldmtx 