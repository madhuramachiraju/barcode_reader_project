#include "barcode_scanner_lib.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <ZXing/ZXingCpp.h>
#include <ZXing/ReadBarcode.h>
#include <ZXing/Flags.h>

extern "C" {
#include <dmtx.h>
}

// Implementations for BarcodeScannerSettings class
BarcodeScannerSettings::BarcodeScannerSettings() : max_codes_per_frame(1), search_whole_image(false), try_harder_mode(false) {
    // Initialize all symbologies to false
    for (int i = 0; i < static_cast<int>(SymbologyType::Aztec) + 1; ++i) {
        enabled_symbologies[static_cast<SymbologyType>(i)] = false;
        color_inverted[static_cast<SymbologyType>(i)] = false;
    }
}

void BarcodeScannerSettings::setSymbologyEnabled(SymbologyType symbology, bool enabled) {
    enabled_symbologies[symbology] = enabled;
}

void BarcodeScannerSettings::setColorInvertedEnabled(SymbologyType symbology, bool enabled) {
    color_inverted[symbology] = enabled;
}

void BarcodeScannerSettings::setMaxCodesPerFrame(int max_codes) {
    max_codes_per_frame = max_codes;
}

void BarcodeScannerSettings::setSearchWholeImage(bool search) {
    search_whole_image = search;
}

void BarcodeScannerSettings::setTryHarderMode(bool try_harder) {
    try_harder_mode = try_harder;
}

std::set<SymbologyType> BarcodeScannerSettings::getEnabledSymbologies() const {
    std::set<SymbologyType> enabled;
    for (const auto& pair : enabled_symbologies) {
        if (pair.second) {
            enabled.insert(pair.first);
        }
    }
    return enabled;
}

bool BarcodeScannerSettings::isColorInverted(SymbologyType symbology) const {
    auto it = color_inverted.find(symbology);
    return it != color_inverted.end() && it->second;
}

int BarcodeScannerSettings::getMaxCodesPerFrame() const {
    return max_codes_per_frame;
}

bool BarcodeScannerSettings::getSearchWholeImage() const {
    return search_whole_image;
}

bool BarcodeScannerSettings::getTryHarderMode() const {
    return try_harder_mode;
}

bool BarcodeScannerSettings::isSymbologyEnabled(SymbologyType symbology) const {
    auto it = enabled_symbologies.find(symbology);
    return it != enabled_symbologies.end() && it->second;
}

// Implementations for RecognitionContext class
RecognitionContext::RecognitionContext() {
    frame_sequence_started = false;
    initialized = true;
    std::cout << "Recognition context created successfully" << std::endl;
}

RecognitionContext::~RecognitionContext() {
    if (frame_sequence_started) {
        endFrameSequence();
    }
    std::cout << "Recognition context released" << std::endl;
}

bool RecognitionContext::startNewFrameSequence() {
    if (!initialized) return false;
    
    frame_sequence_started = true;
    std::cout << "New frame sequence started" << std::endl;
    return true;
}

void RecognitionContext::endFrameSequence() {
    if (frame_sequence_started) {
        frame_sequence_started = false;
        std::cout << "Frame sequence ended" << std::endl;
    }
}

bool RecognitionContext::isFrameSequenceStarted() const {
    return frame_sequence_started;
}

bool RecognitionContext::isInitialized() const {
    return initialized;
}

// Implementations for BarcodeScanner class
BarcodeScanner::BarcodeScanner(std::shared_ptr<RecognitionContext> ctx, std::shared_ptr<BarcodeScannerSettings> sett) 
    : context(ctx), settings(sett), setup_completed(false) {
    
    if (!context || !context->isInitialized()) {
        throw std::runtime_error("Invalid recognition context");
    }
    
    setup_completed = true;
    std::cout << "Barcode scanner created successfully" << std::endl;
}

bool BarcodeScanner::waitForSetupCompleted() {
    std::cout << "Scanner setup completed" << std::endl;
    return setup_completed;
}

ScanStatus BarcodeScanner::processFrame(const ImageDescription& image_desc) {
    if (!context->isFrameSequenceStarted()) {
        std::cout << "Error: Frame sequence not started" << std::endl;
        return SCAN_PROCESSING_ERROR;
    }
    
    if (image_desc.image_data.empty()) {
        std::cout << "Error: Invalid image data" << std::endl;
        return SCAN_INVALID_IMAGE;
    }
    
    std::cout << "Processing frame: " << image_desc.width << "x" << image_desc.height 
             << " (" << image_desc.channels << " channels)" << std::endl;
    
    // Clear previous results
    last_scan_results.clear();
    
    // Convert to grayscale if needed
    cv::Mat gray_image;
    if (image_desc.channels == 3) {
        cv::cvtColor(image_desc.image_data, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image_desc.image_data.clone();
    }
    
    // Process with potential color inversion
    last_scan_results = processWithColorInversion(gray_image);
    
    std::cout << "Scanning completed. Found " << last_scan_results.size() << " barcode(s)" << std::endl;
    
    return last_scan_results.empty() ? SCAN_NO_CODES_FOUND : SCAN_SUCCESS;
}

const std::vector<BarcodeResult>& BarcodeScanner::getLastScanResults() const {
    return last_scan_results;
}

SymbologyType BarcodeScanner::convertZXingFormat(ZXing::BarcodeFormat format) {
    switch (format) {
        case ZXing::BarcodeFormat::QRCode:
            return SymbologyType::QRCode;
        case ZXing::BarcodeFormat::DataMatrix:
            return SymbologyType::DataMatrix;
        case ZXing::BarcodeFormat::Aztec:
            return SymbologyType::Aztec;
        case ZXing::BarcodeFormat::PDF417:
            return SymbologyType::PDF417;
        case ZXing::BarcodeFormat::EAN13:
            return SymbologyType::EAN13;
        case ZXing::BarcodeFormat::EAN8:
            return SymbologyType::EAN8;
        case ZXing::BarcodeFormat::UPCA:
            return SymbologyType::UPCA;
        case ZXing::BarcodeFormat::UPCE:
            return SymbologyType::UPCE;
        case ZXing::BarcodeFormat::Code39:
            return SymbologyType::Code39;
        case ZXing::BarcodeFormat::Code93:
            return SymbologyType::Code93;
        case ZXing::BarcodeFormat::Code128:
            return SymbologyType::Code128;
        default:
            return SymbologyType::None;
    }
}

ZXing::BarcodeFormats BarcodeScanner::createZXingFormats(const std::set<SymbologyType>& enabled_symbologies) {
    ZXing::BarcodeFormats formats;
    for (const auto& symbology : enabled_symbologies) {
        switch (symbology) {
            case SymbologyType::QRCode:
                formats |= ZXing::BarcodeFormat::QRCode;
                break;
            case SymbologyType::DataMatrix:
                formats |= ZXing::BarcodeFormat::DataMatrix;
                break;
            case SymbologyType::Aztec:
                formats |= ZXing::BarcodeFormat::Aztec;
                break;
            case SymbologyType::PDF417:
                formats |= ZXing::BarcodeFormat::PDF417;
                break;
            case SymbologyType::EAN13:
                formats |= ZXing::BarcodeFormat::EAN13;
                break;
            case SymbologyType::EAN8:
                formats |= ZXing::BarcodeFormat::EAN8;
                break;
            case SymbologyType::UPCA:
                formats |= ZXing::BarcodeFormat::UPCA;
                break;
            case SymbologyType::UPCE:
                formats |= ZXing::BarcodeFormat::UPCE;
                break;
            case SymbologyType::Code39:
                formats |= ZXing::BarcodeFormat::Code39;
                break;
            case SymbologyType::Code93:
                formats |= ZXing::BarcodeFormat::Code93;
                break;
            case SymbologyType::Code128:
                formats |= ZXing::BarcodeFormat::Code128;
                break;
            default:
                break;
        }
    }
    return formats;
}

std::vector<BarcodeResult> BarcodeScanner::processWithColorInversion(const cv::Mat& image) {
    std::vector<BarcodeResult> results;
    
    // Process normal image
    auto normal_results = processImage(image, false);
    results.insert(results.end(), normal_results.begin(), normal_results.end());
    
    // Process inverted image if any symbology has color inversion enabled
    bool any_color_inversion = false;
    for (int i = 0; i < static_cast<int>(SymbologyType::Aztec) + 1; ++i) {
        if (settings->isColorInverted(static_cast<SymbologyType>(i))) {
            any_color_inversion = true;
            break;
        }
    }
    
    if (any_color_inversion) {
        cv::Mat inverted;
        cv::bitwise_not(image, inverted);
        auto inverted_results = processImage(inverted, true);
        results.insert(results.end(), inverted_results.begin(), inverted_results.end());
    }
    
    return results;
}

std::vector<BarcodeResult> BarcodeScanner::processImage(const cv::Mat& image, bool is_inverted) {
    std::vector<BarcodeResult> results;
    
    // ZXing processing
    ZXing::ImageView view(image.data, image.cols, image.rows, ZXing::ImageFormat::Lum);
    
    ZXing::ReaderOptions options;
    options.setTryHarder(settings->getTryHarderMode());
    options.setTryRotate(true);
    options.setMaxNumberOfSymbols(settings->getMaxCodesPerFrame());
    options.setFormats(createZXingFormats(settings->getEnabledSymbologies()));
    
    auto barcodes = ZXing::ReadBarcodes(view, options);
    
    for (const auto& barcode : barcodes) {
        if (barcode.isValid() && !barcode.text().empty()) {
            BarcodeResult result;
            result.data = barcode.text();
            result.symbology = convertZXingFormat(barcode.format());
            result.symbology_name = getSymbologyName(result.symbology);
            result.is_color_inverted = is_inverted;
            result.confidence = 1.0; // ZXing doesn't provide confidence
            
            // Parse format-specific details
            switch (result.symbology) {
                case SymbologyType::EAN13:
                case SymbologyType::EAN8:
                case SymbologyType::UPCA:
                    result.format_details = parseGTIN(result.data);
                    break;
                case SymbologyType::QRCode:
                    result.format_details = parseQRCode(result.data);
                    break;
                case SymbologyType::DataMatrix:
                    result.format_details = parseDataMatrix(result.data);
                    break;
                default:
                    result.format_details = "Standard format";
            }
            
            // Get location if available
            try {
                auto position = barcode.position();
                result.location = cv::Rect(
                    position.topLeft().x,
                    position.topLeft().y,
                    position.bottomRight().x - position.topLeft().x,
                    position.bottomRight().y - position.topLeft().y
                );
            } catch (...) {
                // Fallback if position is not available
                result.location = cv::Rect(0, 0, image.cols, image.rows);
            }
            
            results.push_back(result);
        }
    }
    
    // libdmtx processing for DataMatrix (if enabled)
    if (settings->isSymbologyEnabled(SymbologyType::DataMatrix)) {
        auto dm_results = processDataMatrix(image, is_inverted);
        results.insert(results.end(), dm_results.begin(), dm_results.end());
    }
    
    return results;
}

std::vector<BarcodeResult> BarcodeScanner::processDataMatrix(const cv::Mat& image, bool is_inverted) {
    std::vector<BarcodeResult> results;
    
    DmtxImage* img = dmtxImageCreate(image.data, image.cols, image.rows, DmtxPack8bppK);
    if (!img) return results;
    
    DmtxDecode* dec = dmtxDecodeCreate(img, 1);
    if (!dec) {
        dmtxImageDestroy(&img);
        return results;
    }
    
    DmtxTime timeout = dmtxTimeAdd(dmtxTimeNow(), 2000); // 2 second timeout
    
    for (int i = 0; i < 10; i++) {
        DmtxRegion* reg = dmtxRegionFindNext(dec, &timeout);
        if (!reg) break;
        
        DmtxMessage* msg = dmtxDecodeMatrixRegion(dec, reg, DmtxUndefined);
        if (msg && msg->output != nullptr && msg->outputSize > 0) {
            BarcodeResult result;
            result.data = std::string(reinterpret_cast<char*>(msg->output), msg->outputSize);
            result.symbology = SymbologyType::DataMatrix;
            result.symbology_name = "DataMatrix";
            result.is_color_inverted = is_inverted;
            result.confidence = 1.0;
            result.location = cv::Rect(0, 0, image.cols, image.rows);
            
            results.push_back(result);
            dmtxMessageDestroy(&msg);
        }
        
        dmtxRegionDestroy(&reg);
    }
    
    dmtxDecodeDestroy(&dec);
    dmtxImageDestroy(&img);
    
    return results;
}

std::string BarcodeScanner::parseGTIN(const std::string& data) {
    if (data.length() < 8) return "Invalid GTIN";
    
    std::stringstream ss;
    ss << "GTIN: " << data << "\n";
    
    // Check digit validation
    int sum = 0;
    for (size_t i = 0; i < data.length() - 1; i++) {
        int digit = data[i] - '0';
        sum += (i % 2 == 0) ? digit * 3 : digit;
    }
    int checkDigit = (10 - (sum % 10)) % 10;
    
    ss << "Check Digit: " << checkDigit << "\n";
    ss << "Valid: " << (checkDigit == (data.back() - '0') ? "Yes" : "No");
    
    return ss.str();
}

std::string BarcodeScanner::parseQRCode(const std::string& data) {
    std::stringstream ss;
    ss << "QR Code Data:\n";
    
    // Try to detect if it's a URL
    if (data.find("http://") == 0 || data.find("https://") == 0) {
        ss << "Type: URL\n";
        ss << "URL: " << data;
    }
    // Try to detect if it's a vCard
    else if (data.find("BEGIN:VCARD") != std::string::npos) {
        ss << "Type: vCard\n";
        // Parse vCard fields
        std::istringstream iss(data);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.find("FN:") == 0) ss << "Name: " << line.substr(3) << "\n";
            else if (line.find("TEL:") == 0) ss << "Phone: " << line.substr(4) << "\n";
            else if (line.find("EMAIL:") == 0) ss << "Email: " << line.substr(6) << "\n";
        }
    }
    // Try to detect if it's a WiFi configuration
    else if (data.find("WIFI:") == 0) {
        ss << "Type: WiFi Configuration\n";
        // Parse WiFi fields
        std::istringstream iss(data);
        std::string line;
        while (std::getline(iss, line, ';')) {
            if (line.find("S:") == 0) ss << "SSID: " << line.substr(2) << "\n";
            else if (line.find("T:") == 0) ss << "Security: " << line.substr(2) << "\n";
            else if (line.find("P:") == 0) ss << "Password: " << line.substr(2) << "\n";
        }
    }
    else {
        ss << "Type: Text\n";
        ss << "Content: " << data;
    }
    
    return ss.str();
}

std::string BarcodeScanner::parseDataMatrix(const std::string& data) {
    std::stringstream ss;
    ss << "DataMatrix Content:\n";
    
    // Try to detect if it's a GS1 format
    if (data.find("(01)") == 0 || data.find("(10)") == 0 || data.find("(21)") == 0) {
        ss << "Type: GS1\n";
        // Parse GS1 fields
        std::istringstream iss(data);
        std::string line;
        while (std::getline(iss, line, '(')) {
            if (line.empty()) continue;
            size_t end = line.find(')');
            if (end != std::string::npos) {
                std::string ai = line.substr(0, end);
                std::string value = line.substr(end + 1);
                ss << "AI " << ai << ": " << value << "\n";
            }
        }
    }
    else {
        ss << "Type: Raw Data\n";
        ss << "Content: " << data;
    }
    
    return ss.str();
}

std::string BarcodeScanner::getSymbologyName(SymbologyType symbology) const {
    switch (symbology) {
        case SymbologyType::QRCode: return "QR";
        case SymbologyType::DataMatrix: return "DataMatrix";
        case SymbologyType::Aztec: return "Aztec";
        case SymbologyType::PDF417: return "PDF417";
        case SymbologyType::EAN: return "EAN";
        case SymbologyType::Code39: return "Code39";
        case SymbologyType::Code93: return "Code93";
        case SymbologyType::Code128: return "Code128";
        case SymbologyType::UPCA: return "UPCA";
        case SymbologyType::UPCE: return "UPCE";
        case SymbologyType::EAN8: return "EAN8";
        case SymbologyType::EAN13: return "EAN13";
        default: return "Unknown";
    }
}

// Implementations for factory functions
std::shared_ptr<RecognitionContext> createRecognitionContext() {
    return std::make_shared<RecognitionContext>();
}

std::shared_ptr<BarcodeScannerSettings> createScannerSettings(ScanPreset preset) {
    return std::make_shared<BarcodeScannerSettings>();
}

void configureScannerForShippingLabels(std::shared_ptr<BarcodeScannerSettings> settings) {
    std::cout << "\n=== CONFIGURING SCANNER FOR SHIPPING LABELS ===" << std::endl;
    
    // Enable symbologies commonly found on shipping labels
    settings->setSymbologyEnabled(SymbologyType::Code128, true);
    settings->setSymbologyEnabled(SymbologyType::Code39, true);
    settings->setSymbologyEnabled(SymbologyType::EAN, true);
    settings->setSymbologyEnabled(SymbologyType::DataMatrix, true);
    settings->setSymbologyEnabled(SymbologyType::QRCode, true);
    
    // Enable color inversion for problematic barcodes
    settings->setColorInvertedEnabled(SymbologyType::Code128, true);
    settings->setColorInvertedEnabled(SymbologyType::EAN, true);
    
    // Configure for single frame processing
    settings->setMaxCodesPerFrame(10);
    settings->setSearchWholeImage(true);
    settings->setTryHarderMode(true);
    
    std::cout << "Scanner configured for shipping label processing" << std::endl;
}

ImageDescription createImageDescription(const cv::Mat& opencv_image) {
    ImageDescription desc;
    desc.width = opencv_image.cols;
    desc.height = opencv_image.rows;
    desc.channels = opencv_image.channels();
    desc.row_bytes = desc.channels * desc.width;
    desc.memory_size = desc.width * desc.height * desc.channels;
    desc.image_data = opencv_image.clone();
    
    std::cout << "Image description created: " << desc.width << "x" << desc.height 
             << " (" << desc.channels << " channels, " << desc.memory_size << " bytes)" << std::endl;
    
    return desc;
} 