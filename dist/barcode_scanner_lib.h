#ifndef BARCODE_SCANNER_LIB_H
#define BARCODE_SCANNER_LIB_H

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <map>

#include <opencv2/opencv.hpp>

#include <ZXing/ZXingCpp.h>
#include <ZXing/ReadBarcode.h>
#include <ZXing/Flags.h>

extern "C" {
#include <dmtx.h>
}

// Scandit-style enums and structures (duplicate from scan_main.cpp for now, will remove from main later)
enum ScanStatus {
    SCAN_SUCCESS = 0,
    SCAN_NO_CODES_FOUND = 1,
    SCAN_PROCESSING_ERROR = 2,
    SCAN_INVALID_IMAGE = 3
};

enum ScanPreset {
    PRESET_SINGLE_FRAME_MODE,
    PRESET_REALTIME_MODE
};

// Define SymbologyType enum
enum class SymbologyType {
    None,
    Code128,
    Code39,
    Code93,
    EAN,
    EAN13,
    EAN8,
    UPCA,
    UPCE,
    DataMatrix,
    QRCode,
    PDF417,
    Aztec
};

struct BarcodeResult {
    std::string data;
    std::string symbology_name;
    SymbologyType symbology;
    cv::Rect location;
    double confidence;
    bool is_color_inverted;
    std::string format_details;  // Additional format-specific details
    std::string error_correction;  // Error correction level if applicable
    std::string raw_data;  // Raw data before parsing
};

struct ImageDescription {
    int width;
    int height;
    int channels;
    int row_bytes;
    size_t memory_size;
    cv::Mat image_data;
};

// Scandit-style scanner settings class
class BarcodeScannerSettings {
private:
    std::map<SymbologyType, bool> enabled_symbologies;
    std::map<SymbologyType, bool> color_inverted;
    int max_codes_per_frame;
    bool search_whole_image;
    bool try_harder_mode;

public:
    BarcodeScannerSettings();
    void setSymbologyEnabled(SymbologyType symbology, bool enabled);
    void setColorInvertedEnabled(SymbologyType symbology, bool enabled);
    void setMaxCodesPerFrame(int max_codes);
    void setSearchWholeImage(bool search);
    void setTryHarderMode(bool try_harder);
    std::set<SymbologyType> getEnabledSymbologies() const;
    bool isColorInverted(SymbologyType symbology) const;
    int getMaxCodesPerFrame() const;
    bool getSearchWholeImage() const;
    bool getTryHarderMode() const;
    bool isSymbologyEnabled(SymbologyType symbology) const;
};

// Scandit-style recognition context
class RecognitionContext {
private:
    bool frame_sequence_started;
    bool initialized;
    
public:
    RecognitionContext();
    ~RecognitionContext();
    bool startNewFrameSequence();
    void endFrameSequence();
    bool isFrameSequenceStarted() const;
    bool isInitialized() const;
};

// Scandit-style barcode scanner
class BarcodeScanner {
public:
    BarcodeScanner(std::shared_ptr<RecognitionContext> ctx, std::shared_ptr<BarcodeScannerSettings> sett);
    bool waitForSetupCompleted();
    ScanStatus processFrame(const ImageDescription& image_desc);
    const std::vector<BarcodeResult>& getLastScanResults() const;

private:
    SymbologyType convertZXingFormat(ZXing::BarcodeFormat format); // Needs ZXing::BarcodeFormat declared
    ZXing::BarcodeFormats createZXingFormats(const std::set<SymbologyType>& enabled_symbologies); // Needs ZXing::BarcodeFormats declared
    std::vector<BarcodeResult> processWithColorInversion(const cv::Mat& image);
    std::vector<BarcodeResult> processImage(const cv::Mat& image, bool is_inverted = false);
    std::vector<BarcodeResult> processDataMatrix(const cv::Mat& image, bool is_inverted = false); // Needs libdmtx types
    std::string parseGTIN(const std::string& data);
    std::string parseQRCode(const std::string& data);
    std::string parseDataMatrix(const std::string& data);
    std::string getSymbologyName(SymbologyType symbology) const;

    std::shared_ptr<RecognitionContext> context;
    std::shared_ptr<BarcodeScannerSettings> settings;
    std::vector<BarcodeResult> last_scan_results;
    bool setup_completed;
};

// Scandit-style factory functions
std::shared_ptr<RecognitionContext> createRecognitionContext();
std::shared_ptr<BarcodeScannerSettings> createScannerSettings(ScanPreset preset = PRESET_SINGLE_FRAME_MODE);
void configureScannerForShippingLabels(std::shared_ptr<BarcodeScannerSettings> settings);
ImageDescription createImageDescription(const cv::Mat& opencv_image);

#endif // BARCODE_SCANNER_LIB_H 