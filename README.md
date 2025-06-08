# Professional Barcode Reader

A professional-grade barcode reader inspired by Scandit SDK architecture, built with ZXing and libdmtx.

## Features

- Supports multiple barcode formats (1D and 2D)
- High-performance barcode detection
- Visual feedback with highlighted barcodes
- Professional architecture similar to Scandit SDK

## Dependencies

- OpenCV 4.11.0 or later
- ZXing library
- libdmtx
- CMake 3.10 or later
- C++17 compatible compiler

## Installation

### macOS

1. Install dependencies using Homebrew:
```bash
brew install opencv
brew install zxing-cpp
brew install libdmtx
```

2. Build the project:
```bash
mkdir build
cd build
cmake ..
make
```

## Usage

Run the barcode reader with an image file:
```bash
DYLD_LIBRARY_PATH="/usr/local/lib:." ./barcode_reader <path_to_image>
```

Example:
```bash
DYLD_LIBRARY_PATH="/usr/local/lib:." ./barcode_reader test_image.jpg
```

## Project Structure

- `barcode_scanner_lib.h`: Header file containing class definitions
- `barcode_scanner_lib.cpp`: Implementation of the barcode scanner library
- `scan_main.cpp`: Main application file
- `CMakeLists.txt`: Build configuration

## Building from Source

1. Clone the repository
2. Install dependencies
3. Create build directory:
```bash
mkdir build
cd build
```

4. Configure and build:
```bash
cmake ..
make
```

## Notes

- The project uses dynamic libraries (.dylib on macOS)
- Make sure all dependencies are properly installed
- The DYLD_LIBRARY_PATH environment variable is required to run the executable

## License

[Add your license information here] 