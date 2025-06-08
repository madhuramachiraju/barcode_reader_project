// Professional barcode reader inspired by Scandit SDK architecture
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iomanip>

#include "barcode_scanner_lib.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        std::cout << "Professional barcode scanner inspired by Scandit SDK" << std::endl;
        return 1;
    }
    
    std::string image_path = argv[1];
    
    std::cout << "=== PROFESSIONAL BARCODE SCANNER ===" << std::endl;
    std::cout << "Scandit-inspired architecture with ZXing + libdmtx" << std::endl;
    std::cout << "Version 1.0" << std::endl;
    std::cout << "Processing: " << image_path << std::endl;
    
    try {
        // Step 1: Create recognition context (like Scandit)
        auto recognition_context = createRecognitionContext();
        if (!recognition_context) {
            std::cout << "Could not create recognition context!" << std::endl;
            return 1;
        }
        
        // Step 2: Create and configure scanner settings
        auto scanner_settings = createScannerSettings();
        if (!scanner_settings) {
            std::cout << "Could not create scanner settings!" << std::endl;
            return 1;
        }
        
        configureScannerForShippingLabels(scanner_settings);
        
        // Step 3: Create barcode scanner
        auto scanner = std::make_shared<BarcodeScanner>(recognition_context, scanner_settings);
        
        // Step 4: Wait for scanner setup
        if (!scanner->waitForSetupCompleted()) {
            std::cout << "Barcode scanner setup failed!" << std::endl;
            return 1;
        }
        
        // Step 5: Load image
        cv::Mat opencv_image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (opencv_image.empty()) {
            std::cout << "Could not read the image: " << image_path << "\n";
            return 1;
        }
        
        // Step 6: Create image description
        ImageDescription image_desc = createImageDescription(opencv_image);
        
        // Step 7: Start frame sequence
        if (!recognition_context->startNewFrameSequence()) {
            std::cout << "Could not start frame sequence!" << std::endl;
            return 1;
        }
        
        // Step 8: Process the frame
        ScanStatus result = scanner->processFrame(image_desc);
        
        // Step 9: Handle results
        std::cout << "\n=== SCAN RESULTS ===" << std::endl;
        
        if (result == ScanStatus::SCAN_SUCCESS) {
            const auto& results = scanner->getLastScanResults();
            
            std::cout << "Successfully found " << results.size() << " barcode(s):" << std::endl;
            
            for (size_t i = 0; i < results.size(); i++) {
                const auto& barcode = results[i];
                std::cout << "\n📦 Barcode " << i + 1 << ":" << std::endl;
                std::cout << "  Type: " << barcode.symbology_name << std::endl;
                std::cout << "  Data: " << barcode.data << std::endl;
                std::cout << "  Format Details:\n" << barcode.format_details << std::endl;
                std::cout << "  Location: (" << barcode.location.x << "," << barcode.location.y 
                         << ") " << barcode.location.width << "x" << barcode.location.height << std::endl;
                std::cout << "  Color Inverted: " << (barcode.is_color_inverted ? "Yes" : "No") << std::endl;
                std::cout << "  Confidence: " << std::fixed << std::setprecision(2) << barcode.confidence << std::endl;

                // Print barcode location for debugging
                std::cout << "  DEBUG - Barcode Location: x=" << barcode.location.x 
                          << ", y=" << barcode.location.y 
                          << ", width=" << barcode.location.width 
                          << ", height=" << barcode.location.height << std::endl;
            }
            
            // Separate 1D and 2D results
            std::vector<BarcodeResult> code1d, code2d;
            for (const auto& result : results) {
                // Note: SymbologyType::EAN and SymbologyType::UPCA/UPCE are considered 1D barcodes for this purpose
                if (result.symbology == SymbologyType::DataMatrix || 
                    result.symbology == SymbologyType::QRCode ||
                    result.symbology == SymbologyType::Aztec ||
                    result.symbology == SymbologyType::PDF417) {
                    code2d.push_back(result);
                } else {
                    code1d.push_back(result);
                }
            }
            
            std::cout << "\n📊 SUMMARY:" << std::endl;
            std::cout << "1D Barcodes found: " << code1d.size() << std::endl;
            std::cout << "2D Barcodes found: " << code2d.size() << std::endl;
            
            // Display the image with barcode locations
            cv::Mat display_image = opencv_image.clone();
            for (const auto& barcode : results) {
                cv::Scalar color;
                if (barcode.symbology == SymbologyType::DataMatrix || barcode.symbology == SymbologyType::QRCode || barcode.symbology == SymbologyType::Aztec || barcode.symbology == SymbologyType::PDF417) {
                    // 2D Barcodes - Bright green rectangle and text
                    color = cv::Scalar(0, 255, 0); // Bright green
                    cv::rectangle(display_image, barcode.location, color, 4); // Increased thickness
                    
                    // Add text with background for better visibility
                    std::string label = barcode.symbology_name;
                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
                    cv::Point text_origin(barcode.location.x, barcode.location.y - 20);
                    
                    // Draw text background
                    cv::rectangle(display_image, 
                                cv::Rect(text_origin.x, text_origin.y - text_size.height,
                                        text_size.width, text_size.height + baseline),
                                cv::Scalar(0, 0, 0), -1);
                    
                    // Draw text
                    cv::putText(display_image, label, text_origin,
                               cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
                } else {
                    // 1D Barcodes - Red arrow and text
                    color = cv::Scalar(0, 0, 255); // Red
                    
                    // Calculate center of the 1D barcode for the arrow target
                    cv::Point barcode_center(barcode.location.x + barcode.location.width / 2, 
                                             barcode.location.y + barcode.location.height / 2);
                    
                    // Define start point for the arrow (e.g., 50 pixels above the barcode's top-left corner)
                    cv::Point arrow_start(barcode.location.x, barcode.location.y - 50);

                    // Draw an arrow pointing to the 1D barcode
                    cv::arrowedLine(display_image, arrow_start, barcode_center, color, 3, cv::LINE_8, 0, 0.1); // Increased thickness, adjusted tip length
                    
                    // Add text with background for better visibility near the arrow start
                    std::string label = barcode.symbology_name + ": " + barcode.data.substr(0, std::min((int)barcode.data.length(), 20)) + "...";
                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline); // Slightly smaller font for 1D labels
                    cv::Point text_origin(arrow_start.x, arrow_start.y - 10); // Position text above the arrow start
                    
                    // Draw text background
                    cv::rectangle(display_image, 
                                cv::Rect(text_origin.x, text_origin.y - text_size.height,
                                        text_size.width, text_size.height + baseline),
                                cv::Scalar(0, 0, 0), -1);
                    
                    // Draw text
                    cv::putText(display_image, label, text_origin,
                               cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
                }
            }
            
            // Show the image
            cv::imshow("Barcode Detection Results", display_image);
            cv::waitKey(0); // Keep window open indefinitely until a key is pressed
            cv::destroyAllWindows();

            // Save the image with drawn barcodes to a file
            cv::imwrite("barcode_results.jpg", display_image);
            
        } else {
            switch (result) {
                case ScanStatus::SCAN_NO_CODES_FOUND:
                    std::cout << "No barcodes found in the image" << std::endl;
                    break;
                case ScanStatus::SCAN_PROCESSING_ERROR:
                    std::cout << "Processing error occurred" << std::endl;
                    break;
                case ScanStatus::SCAN_INVALID_IMAGE:
                    std::cout << "Invalid image data" << std::endl;
                    break;
                default:
                    std::cout << "Unknown error occurred" << std::endl;
            }
        }
        
        // Step 10: End frame sequence
        recognition_context->endFrameSequence();
        
        std::cout << "\n✅ Processing completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 