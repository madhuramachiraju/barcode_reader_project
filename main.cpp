// Professional barcode reader inspired by Scandit SDK architecture
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>  // for fastNlMeansDenoising
#include <zbar.h>
#include <dmtx.h>
#include <ZXing/ReadBarcode.h>
#include <ZXing/Barcode.h>
#include <ZXing/ReaderOptions.h>
#include <ZXing/ImageView.h>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <map>

using namespace std;
using namespace cv;
using namespace ZXing;

// Scandit-style enums and structures
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

enum SymbologyType {
    SYMBOLOGY_CODE128,
    SYMBOLOGY_CODE39,
    SYMBOLOGY_EAN13,
    SYMBOLOGY_EAN8,
    SYMBOLOGY_UPCA,
    SYMBOLOGY_DATAMATRIX,
    SYMBOLOGY_QR_CODE,
    SYMBOLOGY_PDF417
};

struct BarcodeResult {
    string data;
    string symbology_name;
    SymbologyType symbology;
    Rect location;
    double confidence;
    bool is_color_inverted;
};

struct ImageDescription {
    int width;
    int height;
    int channels;
    int row_bytes;
    size_t memory_size;
    Mat image_data;
};

// Scandit-style scanner settings class
class BarcodeScannerSettings {
private:
    map<SymbologyType, bool> enabled_symbologies;
    map<SymbologyType, bool> color_inverted_enabled;
    bool search_whole_image;
    int max_codes_per_frame;
    bool try_harder_mode;
    ScanPreset preset_mode;

public:
    BarcodeScannerSettings(ScanPreset preset) {
        preset_mode = preset;
        search_whole_image = true;
        max_codes_per_frame = 10;
        try_harder_mode = true;
        
        // Initialize all symbologies as disabled
        enabled_symbologies[SYMBOLOGY_CODE128] = false;
        enabled_symbologies[SYMBOLOGY_CODE39] = false;
        enabled_symbologies[SYMBOLOGY_EAN13] = false;
        enabled_symbologies[SYMBOLOGY_EAN8] = false;
        enabled_symbologies[SYMBOLOGY_UPCA] = false;
        enabled_symbologies[SYMBOLOGY_DATAMATRIX] = false;
        enabled_symbologies[SYMBOLOGY_QR_CODE] = false;
        enabled_symbologies[SYMBOLOGY_PDF417] = false;
        
        // Initialize color inversion settings
        for (auto& pair : enabled_symbologies) {
            color_inverted_enabled[pair.first] = false;
        }
        
        std::cout << "Scanner settings created with preset: " 
                 << (preset == PRESET_SINGLE_FRAME_MODE ? "SINGLE_FRAME_MODE" : "REALTIME_MODE") << std::endl;
    }
    
    void setSymbologyEnabled(SymbologyType symbology, bool enabled) {
        enabled_symbologies[symbology] = enabled;
        string name = getSymbologyName(symbology);
        std::cout << "Symbology " << name << " " << (enabled ? "ENABLED" : "DISABLED") << std::endl;
    }
    
    void setColorInvertedEnabled(SymbologyType symbology, bool enabled) {
        color_inverted_enabled[symbology] = enabled;
        string name = getSymbologyName(symbology);
        cout << "Color inversion for " << name << " " << (enabled ? "ENABLED" : "DISABLED") << endl;
    }
    
    void setMaxCodesPerFrame(int max_codes) {
        max_codes_per_frame = max_codes;
        cout << "Max codes per frame set to: " << max_codes << endl;
    }
    
    void setSearchWholeImage(bool search_whole) {
        search_whole_image = search_whole;
        cout << "Search whole image: " << (search_whole ? "ENABLED" : "DISABLED") << endl;
    }
    
    void setTryHarderMode(bool try_harder) {
        try_harder_mode = try_harder;
        cout << "Try harder mode: " << (try_harder ? "ENABLED" : "DISABLED") << endl;
    }
    
    bool isSymbologyEnabled(SymbologyType symbology) const {
        auto it = enabled_symbologies.find(symbology);
        return it != enabled_symbologies.end() && it->second;
    }
    
    bool isColorInvertedEnabled(SymbologyType symbology) const {
        auto it = color_inverted_enabled.find(symbology);
        return it != color_inverted_enabled.end() && it->second;
    }
    
    int getMaxCodesPerFrame() const { return max_codes_per_frame; }
    bool getSearchWholeImage() const { return search_whole_image; }
    bool getTryHarderMode() const { return try_harder_mode; }
    ScanPreset getPresetMode() const { return preset_mode; }
    
    string getSymbologyName(SymbologyType symbology) const {
        switch (symbology) {
            case SYMBOLOGY_CODE128: return "Code128";
            case SYMBOLOGY_CODE39: return "Code39";
            case SYMBOLOGY_EAN13: return "EAN13";
            case SYMBOLOGY_EAN8: return "EAN8";
            case SYMBOLOGY_UPCA: return "UPCA";
            case SYMBOLOGY_DATAMATRIX: return "DataMatrix";
            case SYMBOLOGY_QR_CODE: return "QR";
            case SYMBOLOGY_PDF417: return "PDF417";
            default: return "Unknown";
        }
    }
};

// Scandit-style recognition context
class RecognitionContext {
private:
    bool frame_sequence_started;
    bool initialized;
    
public:
    RecognitionContext() {
        frame_sequence_started = false;
        initialized = true;
        std::cout << "Recognition context created successfully" << std::endl;
    }
    
    ~RecognitionContext() {
        if (frame_sequence_started) {
            endFrameSequence();
        }
        std::cout << "Recognition context released" << std::endl;
    }
    
    bool startNewFrameSequence() {
        if (!initialized) return false;
        
        frame_sequence_started = true;
        std::cout << "New frame sequence started" << std::endl;
        return true;
    }
    
    void endFrameSequence() {
        if (frame_sequence_started) {
            frame_sequence_started = false;
            std::cout << "Frame sequence ended" << std::endl;
        }
    }
    
    bool isFrameSequenceStarted() const {
        return frame_sequence_started;
    }
    
    bool isInitialized() const {
        return initialized;
    }
};

// Scandit-style barcode scanner
class BarcodeScanner {
private:
    shared_ptr<RecognitionContext> context;
    shared_ptr<BarcodeScannerSettings> settings;
    vector<BarcodeResult> last_scan_results;
    bool setup_completed;
    
    // Convert ZXing format to our symbology type
    SymbologyType convertZXingFormat(BarcodeFormat format) {
        switch (format) {
            case ZXing::BarcodeFormat::Code128: return SYMBOLOGY_CODE128;
            case ZXing::BarcodeFormat::Code39: return SYMBOLOGY_CODE39;
            case ZXing::BarcodeFormat::EAN13: return SYMBOLOGY_EAN13;
            case ZXing::BarcodeFormat::EAN8: return SYMBOLOGY_EAN8;
            case ZXing::BarcodeFormat::UPCA: return SYMBOLOGY_UPCA;
            case ZXing::BarcodeFormat::DataMatrix: return SYMBOLOGY_DATAMATRIX;
            case ZXing::BarcodeFormat::QRCode: return SYMBOLOGY_QR_CODE;
            case ZXing::BarcodeFormat::PDF417: return SYMBOLOGY_PDF417;
            default: return SYMBOLOGY_CODE128; // fallback
        }
    }
    
    // Create ZXing barcode formats from enabled symbologies
    BarcodeFormats createZXingFormats() {
        BarcodeFormats formats = ZXing::BarcodeFormat::None;
        
        cout << "\n=== ENABLING ZXING FORMATS ===" << endl;
        
        if (settings->isSymbologyEnabled(SYMBOLOGY_CODE128)) {
            formats |= ZXing::BarcodeFormat::Code128;
            cout << "✓ Code128 enabled" << endl;
        }
        if (settings->isSymbologyEnabled(SYMBOLOGY_CODE39)) {
            formats |= ZXing::BarcodeFormat::Code39;
            cout << "✓ Code39 enabled" << endl;
        }
        if (settings->isSymbologyEnabled(SYMBOLOGY_EAN13)) {
            formats |= ZXing::BarcodeFormat::EAN13;
            cout << "✓ EAN13 enabled" << endl;
        }
        if (settings->isSymbologyEnabled(SYMBOLOGY_EAN8)) {
            formats |= ZXing::BarcodeFormat::EAN8;
            cout << "✓ EAN8 enabled" << endl;
        }
        if (settings->isSymbologyEnabled(SYMBOLOGY_UPCA)) {
            formats |= ZXing::BarcodeFormat::UPCA;
            cout << "✓ UPCA enabled" << endl;
        }
        if (settings->isSymbologyEnabled(SYMBOLOGY_DATAMATRIX)) {
            formats |= ZXing::BarcodeFormat::DataMatrix;
            cout << "✓ DataMatrix enabled" << endl;
        }
        if (settings->isSymbologyEnabled(SYMBOLOGY_QR_CODE)) {
            formats |= ZXing::BarcodeFormat::QRCode;
            cout << "✓ QR Code enabled" << endl;
        }
        if (settings->isSymbologyEnabled(SYMBOLOGY_PDF417)) {
            formats |= ZXing::BarcodeFormat::PDF417;
            cout << "✓ PDF417 enabled" << endl;
        }
        
        cout << "ZXing formats configured: " << (formats == ZXing::BarcodeFormat::None ? "NONE" : "MULTIPLE") << endl;
        
        return formats;
    }
    
    // Process with color inversion if enabled
    vector<BarcodeResult> processWithColorInversion(const Mat& image) {
        vector<BarcodeResult> results;
        
        // Process normal image
        auto normal_results = processImage(image, false);
        results.insert(results.end(), normal_results.begin(), normal_results.end());
        
        // Process inverted image if any symbology has color inversion enabled
        bool any_color_inversion = false;
        for (int i = SYMBOLOGY_CODE128; i <= SYMBOLOGY_PDF417; i++) {
            if (settings->isColorInvertedEnabled(static_cast<SymbologyType>(i))) {
                any_color_inversion = true;
                break;
            }
        }
        
        if (any_color_inversion) {
            Mat inverted;
            bitwise_not(image, inverted);
            auto inverted_results = processImage(inverted, true);
            results.insert(results.end(), inverted_results.begin(), inverted_results.end());
        }
        
        return results;
    }
    
    // Core image processing function
    vector<BarcodeResult> processImage(const Mat& image, bool is_inverted = false) {
        vector<BarcodeResult> results;
        
        // Preprocess image for better low-resolution barcode detection
        Mat processed_image = image.clone();
        
        // Convert to grayscale if needed
        if (processed_image.channels() == 3) {
            cvtColor(processed_image, processed_image, COLOR_BGR2GRAY);
        }
        
        // Apply advanced preprocessing for low-resolution images
        // 1. Upscale image using bicubic interpolation
        Mat upscaled;
        resize(processed_image, upscaled, cv::Size(), 2.0, 2.0, INTER_CUBIC);
        
        // 2. Apply adaptive histogram equalization for better contrast
        Mat clahe_output;
        Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(upscaled, clahe_output);
        
        // 3. Denoise while preserving edges
        Mat denoised;
        fastNlMeansDenoising(clahe_output, denoised, 10, 7, 21);
        
        // 4. Apply unsharp masking for edge enhancement
        Mat gaussian_blur;
        GaussianBlur(denoised, gaussian_blur, cv::Size(0, 0), 3);
        Mat unsharp_mask = denoised - gaussian_blur;
        Mat sharpened = denoised + 0.7 * unsharp_mask;
        
        // 5. Apply adaptive thresholding
        Mat binary;
        adaptiveThreshold(sharpened, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 5);
        
        // 6. Apply morphological operations to clean up noise and enhance barcode patterns
        Mat morph_kernel = getStructuringElement(MORPH_RECT, cv::Size(3, 3));
        Mat cleaned;
        morphologyEx(binary, cleaned, MORPH_CLOSE, morph_kernel);
        
        // Try multiple scales for barcode detection
        vector<double> scales = {1.0, 1.5, 2.0};
        for (double scale : scales) {
            Mat scaled;
            if (scale != 1.0) {
                resize(cleaned, scaled, cv::Size(), scale, scale, INTER_LINEAR);
            } else {
                scaled = cleaned;
            }
            
            // ZXing processing with enhanced image
            ZXing::ImageView view(scaled.data, scaled.cols, scaled.rows, ZXing::ImageFormat::Lum);
            
            ZXing::ReaderOptions options;
            options.setTryHarder(settings->getTryHarderMode());
            options.setTryRotate(true);
            options.setMaxNumberOfSymbols(settings->getMaxCodesPerFrame());
            options.setFormats(createZXingFormats());
            
            auto barcodes = ZXing::ReadBarcodes(view, options);
            
            cout << "\n=== ZXING BARCODE DETECTION (Scale: " << scale << ") ===" << endl;
            cout << "ZXing found " << barcodes.size() << " barcode(s)" << endl;
            
            for (const auto& barcode : barcodes) {
                cout << "ZXing barcode: format=" << static_cast<int>(barcode.format()) 
                     << ", valid=" << barcode.isValid() 
                     << ", text='" << barcode.text() << "'" << endl;
                 
                if (barcode.isValid() && !barcode.text().empty()) {
                    BarcodeResult result;
                    result.data = barcode.text();
                    result.symbology = convertZXingFormat(barcode.format());
                    result.symbology_name = settings->getSymbologyName(result.symbology);
                    result.is_color_inverted = is_inverted;
                    result.confidence = 1.0; // ZXing doesn't provide confidence
                    
                    cout << "Processing ZXing barcode: " << result.symbology_name << " - " << result.data << endl;
                    
                    // Get location if available
                    try {
                        auto position = barcode.position();
                        cout << "ZXing barcode position retrieved successfully" << endl;
                        result.location = Rect(
                            position.topLeft().x,
                            position.topLeft().y,
                            position.bottomRight().x - position.topLeft().x,
                            position.bottomRight().y - position.topLeft().y
                        );
                        cout << "Location: (" << result.location.x << "," << result.location.y 
                             << ") " << result.location.width << "x" << result.location.height << endl;
                    } catch (const std::exception& e) {
                        cout << "Failed to get ZXing barcode position: " << e.what() << endl;
                        result.location = Rect(0, 0, image.cols, image.rows);
                    }
                    
                    results.push_back(result);
                    cout << "Added ZXing barcode to results: " << result.symbology_name << endl;
                } else {
                    cout << "Skipping invalid or empty ZXing barcode" << endl;
                }
            }
        }
        
        // libdmtx processing for DataMatrix (if enabled)
        if (settings->isSymbologyEnabled(SYMBOLOGY_DATAMATRIX)) {
            auto dm_results = processDataMatrix(image, is_inverted);
            results.insert(results.end(), dm_results.begin(), dm_results.end());
        }
        
        // ZBar 1D barcode detection (if any 1D symbologies are enabled)
        bool any_1d_enabled = settings->isSymbologyEnabled(SYMBOLOGY_CODE128) ||
                             settings->isSymbologyEnabled(SYMBOLOGY_CODE39) ||
                             settings->isSymbologyEnabled(SYMBOLOGY_EAN13) ||
                             settings->isSymbologyEnabled(SYMBOLOGY_EAN8) ||
                             settings->isSymbologyEnabled(SYMBOLOGY_UPCA);
        if (any_1d_enabled) {
            auto zbar_results = processZBar1D(image, is_inverted);
            results.insert(results.end(), zbar_results.begin(), zbar_results.end());
        }
        
        return results;
    }
    
    // DataMatrix processing using libdmtx
    vector<BarcodeResult> processDataMatrix(const Mat& image, bool is_inverted = false) {
        vector<BarcodeResult> results;
        
        DmtxImage* img = dmtxImageCreate(image.data, image.cols, image.rows, DmtxPack8bppK);
        if (!img) return results;
        
        DmtxDecode* dec = dmtxDecodeCreate(img, 1);
        if (!dec) {
            dmtxImageDestroy(&img);
            return results;
        }
        
        DmtxTime timeout = dmtxTimeAdd(dmtxTimeNow(), 2000); // 2 second timeout
        
        for (int i = 0; i < settings->getMaxCodesPerFrame(); i++) {
            DmtxRegion* reg = dmtxRegionFindNext(dec, &timeout);
            if (!reg) break;
            
            DmtxMessage* msg = dmtxDecodeMatrixRegion(dec, reg, DmtxUndefined);
            if (msg && msg->output != nullptr && msg->outputSize > 0) {
                BarcodeResult result;
                result.data = string(reinterpret_cast<char*>(msg->output), msg->outputSize);
                result.symbology = SYMBOLOGY_DATAMATRIX;
                result.symbology_name = "DataMatrix";
                result.is_color_inverted = is_inverted;
                result.confidence = 1.0;
                
                // Get the actual barcode location from libdmtx using boundMin and boundMax
                result.location = Rect(
                    reg->boundMin.X,
                    reg->boundMin.Y,
                    reg->boundMax.X - reg->boundMin.X,
                    reg->boundMax.Y - reg->boundMin.Y
                );
                cout << "DataMatrix location: (" << result.location.x << "," << result.location.y 
                     << ") " << result.location.width << "x" << result.location.height << endl;
                
                results.push_back(result);
                dmtxMessageDestroy(&msg);
            }
            
            dmtxRegionDestroy(&reg);
        }
        
        dmtxDecodeDestroy(&dec);
        dmtxImageDestroy(&img);
        
        return results;
    }
    
    // ZBar processing for 1D barcodes
    vector<BarcodeResult> processZBar1D(const Mat& image, bool is_inverted = false) {
        vector<BarcodeResult> results;
        
        // Create ZBar scanner
        zbar::ImageScanner scanner;
        scanner.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);
        
        // Convert image to grayscale if needed
        Mat gray_image;
        if (image.channels() == 3) {
            cvtColor(image, gray_image, COLOR_BGR2GRAY);
        } else {
            gray_image = image.clone();
        }
        
        // Create ZBar image
        zbar::Image zbar_image(gray_image.cols, gray_image.rows, "Y800", gray_image.data, gray_image.cols * gray_image.rows);
        
        // Scan for barcodes
        int n = scanner.scan(zbar_image);
        if (n > 0) {
            for (zbar::Image::SymbolIterator symbol = zbar_image.symbol_begin(); symbol != zbar_image.symbol_end(); ++symbol) {
                BarcodeResult result;
                result.data = symbol->get_data();
                result.is_color_inverted = is_inverted;
                result.confidence = 1.0;
                
                // Convert ZBar format to our symbology type
                string zbar_type = symbol->get_type_name();
                if (zbar_type == "CODE-128") {
                    result.symbology = SYMBOLOGY_CODE128;
                    result.symbology_name = "Code128";
                } else if (zbar_type == "CODE-39") {
                    result.symbology = SYMBOLOGY_CODE39;
                    result.symbology_name = "Code39";
                } else if (zbar_type == "EAN-13") {
                    result.symbology = SYMBOLOGY_EAN13;
                    result.symbology_name = "EAN13";
                } else if (zbar_type == "EAN-8") {
                    result.symbology = SYMBOLOGY_EAN8;
                    result.symbology_name = "EAN8";
                } else if (zbar_type == "UPC-A") {
                    result.symbology = SYMBOLOGY_UPCA;
                    result.symbology_name = "UPCA";
                } else {
                    continue;
                }
                // Get location from ZBar symbol
                vector<cv::Point> points;
                for (int i = 0; i < symbol->get_location_size(); i++) {
                    points.push_back(cv::Point(symbol->get_location_x(i), symbol->get_location_y(i)));
                }
                if (points.size() >= 4) {
                    int minX = points[0].x, maxX = points[0].x;
                    int minY = points[0].y, maxY = points[0].y;
                    for (const auto& point : points) {
                        minX = min(minX, point.x);
                        maxX = max(maxX, point.x);
                        minY = min(minY, point.y);
                        maxY = max(maxY, point.y);
                    }
                    result.location = Rect(minX, minY, maxX - minX, maxY - minY);
                } else {
                    result.location = Rect(0, 0, image.cols, image.rows);
                }
                results.push_back(result);
            }
        }
        return results;
    }
    
public:
    BarcodeScanner(shared_ptr<RecognitionContext> ctx, shared_ptr<BarcodeScannerSettings> sett) 
        : context(ctx), settings(sett), setup_completed(false) {
        
        if (!context || !context->isInitialized()) {
            throw runtime_error("Invalid recognition context");
        }
        
        setup_completed = true;
        std::cout << "Barcode scanner created successfully" << std::endl;
    }
    
    bool waitForSetupCompleted() {
        std::cout << "Scanner setup completed" << std::endl;
        return setup_completed;
    }
    
    ScanStatus processFrame(const ImageDescription& image_desc, Mat& output_image_with_overlay) {
        if (!context->isFrameSequenceStarted()) {
            cout << "Error: Frame sequence not started" << endl;
            return SCAN_PROCESSING_ERROR;
        }
        
        if (image_desc.image_data.empty()) {
            cout << "Error: Invalid image data" << endl;
            return SCAN_INVALID_IMAGE;
        }
        
        cout << "Processing frame: " << image_desc.width << "x" << image_desc.height 
             << " (" << image_desc.channels << " channels)" << endl;
        
        // Clear previous results
        last_scan_results.clear();
        
        // Create output image for overlay (copy original)
        output_image_with_overlay = image_desc.image_data.clone();
        
        // Convert to grayscale for processing
        Mat gray_image;
        if (image_desc.channels == 3) {
            cvtColor(image_desc.image_data, gray_image, COLOR_BGR2GRAY);
        } else {
            gray_image = image_desc.image_data.clone();
            // Convert grayscale to color for overlay
            cvtColor(gray_image, output_image_with_overlay, COLOR_GRAY2BGR);
        }
        
        // Process with potential color inversion
        last_scan_results = processWithColorInversion(gray_image);
        
        // Draw overlays on the output image
        drawBarcodeOverlays(output_image_with_overlay, last_scan_results);
        
        cout << "Scanning completed. Found " << last_scan_results.size() << " barcode(s)" << endl;
        
        return last_scan_results.empty() ? SCAN_NO_CODES_FOUND : SCAN_SUCCESS;
    }
    
    const vector<BarcodeResult>& getLastScanResults() const {
        return last_scan_results;
    }
    
private:
    // Professional Scandit-style overlay drawing
    void drawBarcodeOverlays(Mat& image, const vector<BarcodeResult>& results) {
        cout << "\n=== DRAWING BARCODE OVERLAYS ===" << endl;
        
        if (results.empty()) {
            cout << "No results to draw overlays for" << endl;
            return;
        }
        
        for (size_t i = 0; i < results.size(); i++) {
            const auto& barcode = results[i];
            
            // Validate barcode location
            if (barcode.location.width <= 0 || barcode.location.height <= 0) {
                cout << "Skipping barcode " << i + 1 << " with invalid location" << endl;
                continue;
            }
            
            // Ensure location is within image bounds
            if (barcode.location.x < 0 || barcode.location.y < 0 || 
                barcode.location.x + barcode.location.width > image.cols ||
                barcode.location.y + barcode.location.height > image.rows) {
                cout << "Skipping barcode " << i + 1 << " with out-of-bounds location" << endl;
                continue;
            }
            
            // Choose color based on symbology type
            Scalar overlay_color;
            Scalar text_color = Scalar(255, 255, 255); // White text
            string prefix;
            
            if (barcode.symbology == SYMBOLOGY_DATAMATRIX || barcode.symbology == SYMBOLOGY_QR_CODE) {
                overlay_color = Scalar(255, 100, 0);  // Orange for 2D codes
                prefix = "2D: ";
            } else {
                overlay_color = Scalar(0, 255, 0);     // Green for 1D codes  
                prefix = "1D: ";
            }
            
            // Adjust for color inversion
            if (barcode.is_color_inverted) {
                overlay_color = Scalar(255, 0, 255);   // Magenta for inverted
                prefix += "[INV] ";
            }
            
            try {
                // Draw bounding rectangle
                rectangle(image, barcode.location, overlay_color, 3);
                
                // Draw corner markers (Scandit-style)
                int corner_size = 15;
                Point tl = barcode.location.tl();
                Point br = barcode.location.br();
                
                // Top-left corner
                line(image, tl, Point(tl.x + corner_size, tl.y), overlay_color, 5);
                line(image, tl, Point(tl.x, tl.y + corner_size), overlay_color, 5);
                
                // Top-right corner
                line(image, Point(br.x, tl.y), Point(br.x - corner_size, tl.y), overlay_color, 5);
                line(image, Point(br.x, tl.y), Point(br.x, tl.y + corner_size), overlay_color, 5);
                
                // Bottom-left corner
                line(image, Point(tl.x, br.y), Point(tl.x + corner_size, br.y), overlay_color, 5);
                line(image, Point(tl.x, br.y), Point(tl.x, br.y - corner_size), overlay_color, 5);
                
                // Bottom-right corner
                line(image, br, Point(br.x - corner_size, br.y), overlay_color, 5);
                line(image, br, Point(br.x, br.y - corner_size), overlay_color, 5);
                
                // Prepare barcode data text
                string display_text = prefix + barcode.symbology_name + ": " + barcode.data;
                
                // Truncate long text
                if (display_text.length() > 30) {
                    display_text = display_text.substr(0, 27) + "...";
                }
                
                // Calculate text size and position
                int font_face = FONT_HERSHEY_SIMPLEX;
                double font_scale = 0.7;
                int font_thickness = 2;
                int baseline = 0;
                
                cv::Size text_size = getTextSize(display_text, font_face, font_scale, font_thickness, &baseline);
                
                // Position text above the barcode, or below if near top
                Point text_position;
                if (barcode.location.y > text_size.height + 10) {
                    text_position = Point(barcode.location.x, barcode.location.y - 10);
                } else {
                    text_position = Point(barcode.location.x, barcode.location.y + barcode.location.height + text_size.height + 10);
                }
                
                // Ensure text stays within image bounds
                text_position.x = max(0, min(text_position.x, image.cols - text_size.width));
                text_position.y = max(text_size.height, min(text_position.y, image.rows - 10));
                
                // Draw text background rectangle with bounds checking
                Rect text_bg_rect(
                    max(0, text_position.x - 5),
                    max(0, text_position.y - text_size.height - 5),
                    min(text_size.width + 10, image.cols - max(0, text_position.x - 5)),
                    min(text_size.height + 10, image.rows - max(0, text_position.y - text_size.height - 5))
                );
                
                // Only draw background if rectangle is valid
                if (text_bg_rect.width > 0 && text_bg_rect.height > 0 && 
                    text_bg_rect.x + text_bg_rect.width <= image.cols && 
                    text_bg_rect.y + text_bg_rect.height <= image.rows) {
                    
                    // Semi-transparent background
                    Mat text_bg = image(text_bg_rect);
                    Mat colored_bg(text_bg.size(), text_bg.type(), overlay_color);
                    addWeighted(text_bg, 0.7, colored_bg, 0.3, 0, text_bg);
                }
                
                // Draw the text
                putText(image, display_text, text_position, font_face, font_scale, text_color, font_thickness);
                
                // Draw barcode number circle
                Point circle_center(barcode.location.x - 20, barcode.location.y - 20);
                circle_center.x = max(25, min(circle_center.x, image.cols - 25));
                circle_center.y = max(25, min(circle_center.y, image.rows - 25));
                
                circle(image, circle_center, 20, overlay_color, FILLED);
                circle(image, circle_center, 20, Scalar(0, 0, 0), 2);
                
                string number_text = to_string(i + 1);
                cv::Size num_size = getTextSize(number_text, FONT_HERSHEY_SIMPLEX, 0.8, 2, nullptr);
                Point num_pos(circle_center.x - num_size.width/2, circle_center.y + num_size.height/2);
                putText(image, number_text, num_pos, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
                
                cout << "Drew overlay for barcode " << i + 1 << ": " << barcode.symbology_name 
                     << " at (" << barcode.location.x << "," << barcode.location.y << ")" << endl;
                     
            } catch (const std::exception& e) {
                cout << "Error drawing overlay for barcode " << i + 1 << ": " << e.what() << endl;
                continue;
            }
        }
        
        // Draw header with scan summary
        try {
            drawScanSummaryHeader(image, results);
        } catch (const std::exception& e) {
            cout << "Error drawing scan summary header: " << e.what() << endl;
        }
        
        cout << "Overlay drawing completed for " << results.size() << " barcode(s)" << endl;
    }
    
    void drawScanSummaryHeader(Mat& image, const vector<BarcodeResult>& results) {
        // Count barcode types
        int count_1d = 0, count_2d = 0, count_inverted = 0;
        for (const auto& result : results) {
            if (result.symbology == SYMBOLOGY_DATAMATRIX || result.symbology == SYMBOLOGY_QR_CODE) {
                count_2d++;
            } else {
                count_1d++;
            }
            if (result.is_color_inverted) {
                count_inverted++;
            }
        }
        
        // Draw header background
        Rect header_rect(0, 0, image.cols, 80);
        Mat header_bg = image(header_rect);
        Mat dark_bg(header_bg.size(), header_bg.type(), Scalar(40, 40, 40));
        addWeighted(header_bg, 0.3, dark_bg, 0.7, 0, header_bg);
        
        // Draw header text
        string summary = "SCANDIT-STYLE SCANNER | Found: " + to_string(results.size()) + " codes";
        putText(image, summary, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        
        string details = "1D: " + to_string(count_1d) + " | 2D: " + to_string(count_2d) + 
                        " | Inverted: " + to_string(count_inverted);
        putText(image, details, Point(10, 55), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);
    }
};

// Scandit-style factory functions
shared_ptr<RecognitionContext> createRecognitionContext() {
    return make_shared<RecognitionContext>();
}

shared_ptr<BarcodeScannerSettings> createScannerSettings(ScanPreset preset = PRESET_SINGLE_FRAME_MODE) {
    return make_shared<BarcodeScannerSettings>(preset);
}

void configureScannerForShippingLabels(shared_ptr<BarcodeScannerSettings> settings) {
    cout << "\n=== CONFIGURING SCANNER FOR SHIPPING LABELS ===" << endl;
    
    // Enable symbologies commonly found on shipping labels
    settings->setSymbologyEnabled(SYMBOLOGY_CODE128, true);
    settings->setSymbologyEnabled(SYMBOLOGY_CODE39, true);
    settings->setSymbologyEnabled(SYMBOLOGY_EAN13, true);
    settings->setSymbologyEnabled(SYMBOLOGY_EAN8, true);
    settings->setSymbologyEnabled(SYMBOLOGY_DATAMATRIX, true);
    settings->setSymbologyEnabled(SYMBOLOGY_QR_CODE, true);
    
    // Enable color inversion for problematic barcodes
    settings->setColorInvertedEnabled(SYMBOLOGY_CODE128, true);
    settings->setColorInvertedEnabled(SYMBOLOGY_EAN13, true);
    
    // Configure for single frame processing
    settings->setMaxCodesPerFrame(10);
    settings->setSearchWholeImage(true);
    settings->setTryHarderMode(true);
    
    cout << "Scanner configured for shipping label processing" << endl;
}

void configureScannerForLowResolution(std::shared_ptr<BarcodeScannerSettings> settings) {
    std::cout << "\n=== CONFIGURING SCANNER FOR LOW RESOLUTION BARCODES ===" << std::endl;
    
    // Enable all supported symbologies
    settings->setSymbologyEnabled(SYMBOLOGY_CODE128, true);
    settings->setSymbologyEnabled(SYMBOLOGY_CODE39, true);
    settings->setSymbologyEnabled(SYMBOLOGY_EAN13, true);
    settings->setSymbologyEnabled(SYMBOLOGY_EAN8, true);
    settings->setSymbologyEnabled(SYMBOLOGY_UPCA, true);
    settings->setSymbologyEnabled(SYMBOLOGY_DATAMATRIX, true);
    settings->setSymbologyEnabled(SYMBOLOGY_QR_CODE, true);
    
    // Enable color inversion for all symbologies
    settings->setColorInvertedEnabled(SYMBOLOGY_CODE128, true);
    settings->setColorInvertedEnabled(SYMBOLOGY_CODE39, true);
    settings->setColorInvertedEnabled(SYMBOLOGY_EAN13, true);
    settings->setColorInvertedEnabled(SYMBOLOGY_EAN8, true);
    settings->setColorInvertedEnabled(SYMBOLOGY_UPCA, true);
    settings->setColorInvertedEnabled(SYMBOLOGY_DATAMATRIX, true);
    settings->setColorInvertedEnabled(SYMBOLOGY_QR_CODE, true);
    
    // Configure for maximum detection capability
    settings->setMaxCodesPerFrame(20);  // Increase max codes per frame
    settings->setSearchWholeImage(true); // Search entire image
    settings->setTryHarderMode(true);    // Enable try harder mode
    
    std::cout << "Scanner configured for low resolution barcode detection" << std::endl;
}

ImageDescription createImageDescription(const Mat& opencv_image) {
    ImageDescription desc;
    desc.width = opencv_image.cols;
    desc.height = opencv_image.rows;
    desc.channels = opencv_image.channels();
    desc.row_bytes = desc.channels * desc.width;
    desc.memory_size = desc.width * desc.height * desc.channels;
    desc.image_data = opencv_image.clone();
    
    cout << "Image description created: " << desc.width << "x" << desc.height 
         << " (" << desc.channels << " channels, " << desc.memory_size << " bytes)" << endl;
    
    return desc;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        std::cout << "Professional barcode scanner for low resolution images" << std::endl;
        return 1;
    }
    
    string image_path = argv[1];
    
    std::cout << "=== PROFESSIONAL BARCODE SCANNER FOR LOW RESOLUTION IMAGES ===" << std::endl;
    std::cout << "Enhanced preprocessing and multi-scale detection" << std::endl;
    std::cout << "Version 2.0" << std::endl;
    std::cout << "Processing: " << image_path << std::endl;
    
    try {
        // Step 1: Create recognition context
        auto recognition_context = std::make_shared<RecognitionContext>();
        if (!recognition_context) {
            std::cout << "Could not create recognition context!" << std::endl;
            return 1;
        }
        
        // Step 2: Create and configure scanner settings
        auto scanner_settings = std::make_shared<BarcodeScannerSettings>(PRESET_SINGLE_FRAME_MODE);
        if (!scanner_settings) {
            std::cout << "Could not create scanner settings!" << std::endl;
            return 1;
        }
        
        configureScannerForLowResolution(scanner_settings);  // Use low resolution optimized settings
        
        // Step 3: Create barcode scanner
        auto scanner = std::make_shared<BarcodeScanner>(recognition_context, scanner_settings);
        
        // Step 4: Wait for scanner setup
        if (!scanner->waitForSetupCompleted()) {
            cout << "Barcode scanner setup failed!" << endl;
            return 1;
        }
        
        // Step 5: Load image
        Mat opencv_image = imread(image_path, IMREAD_COLOR);
        if (opencv_image.empty()) {
            cout << "Could not read the image: " << image_path << endl;
            return 1;
        }
        
        // Step 6: Create image description
        ImageDescription image_desc = createImageDescription(opencv_image);
        
        // Step 7: Start frame sequence
        if (!recognition_context->startNewFrameSequence()) {
            cout << "Could not start frame sequence!" << endl;
            return 1;
        }
        
        // Step 8: Process the frame with overlay generation
        Mat output_image_with_overlay;
        ScanStatus result = scanner->processFrame(image_desc, output_image_with_overlay);
        
        // Step 9: Handle results and save output image
        cout << "\n=== SCAN RESULTS ===" << endl;
        
        try {
            if (result == SCAN_SUCCESS) {
                const auto& results = scanner->getLastScanResults();
                
                cout << "Successfully found " << results.size() << " barcode(s):" << endl;
                
                for (size_t i = 0; i < results.size(); i++) {
                    const auto& barcode = results[i];
                    cout << "\nBarcode " << i + 1 << ":" << endl;
                    cout << "  Data: " << barcode.data << endl;
                    cout << "  Symbology: " << barcode.symbology_name << endl;
                    cout << "  Location: (" << barcode.location.x << "," << barcode.location.y 
                         << ") " << barcode.location.width << "x" << barcode.location.height << endl;
                    cout << "  Color Inverted: " << (barcode.is_color_inverted ? "Yes" : "No") << endl;
                    cout << "  Confidence: " << barcode.confidence << endl;
                }
                
                // Separate 1D and 2D results
                vector<BarcodeResult> code1d, code2d;
                for (const auto& result : results) {
                    if (result.symbology == SYMBOLOGY_DATAMATRIX || result.symbology == SYMBOLOGY_QR_CODE) {
                        code2d.push_back(result);
                    } else {
                        code1d.push_back(result);
                    }
                }
                
                cout << "\n📊 SUMMARY:" << endl;
                cout << "1D Barcodes found: " << code1d.size() << endl;
                cout << "2D Barcodes found: " << code2d.size() << endl;
                
                // Save the output image with overlays
                string output_filename = "scandit_style_output.jpg";
                if (imwrite(output_filename, output_image_with_overlay)) {
                    cout << "\n💾 Output image with overlays saved: " << output_filename << endl;
                    cout << "Image size: " << output_image_with_overlay.cols << "x" << output_image_with_overlay.rows << endl;
                } else {
                    cout << "\n❌ Failed to save output image" << endl;
                }
                
            } else {
                // Even if no barcodes found, save the image for debugging
                string debug_filename = "scandit_style_debug.jpg";
                if (!output_image_with_overlay.empty()) {
                    imwrite(debug_filename, output_image_with_overlay);
                    cout << "Debug image saved: " << debug_filename << endl;
                }
                
                switch (result) {
                    case SCAN_NO_CODES_FOUND:
                        cout << "No barcodes found in the image" << endl;
                        break;
                    case SCAN_PROCESSING_ERROR:
                        cout << "Processing error occurred" << endl;
                        break;
                    case SCAN_INVALID_IMAGE:
                        cout << "Invalid image data" << endl;
                        break;
                    default:
                        cout << "Unknown error occurred" << endl;
                }
            }
        } catch (const std::exception& e) {
            cout << "Error processing scan results: " << e.what() << endl;
            cout << "Attempting to save debug image..." << endl;
            
            string debug_filename = "scandit_style_error_debug.jpg";
            if (!output_image_with_overlay.empty()) {
                imwrite(debug_filename, output_image_with_overlay);
                cout << "Error debug image saved: " << debug_filename << endl;
            }
        }
        
        // Step 10: End frame sequence
        recognition_context->endFrameSequence();
        
        cout << "\n✅ Processing completed successfully" << endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 