#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "CourtLinePixelDetector.h"
#include "CourtLineCandidateDetector.h"
#include "TennisCourtFitter.h"
#include "TimeMeasurement.h"

// Keep the released detector stages; replace only midpoint decoding with an image input.
int main(int argc, char** argv) {
    if (argc != 3) return 1;
    cv::setNumThreads(2);
    TimeMeasurement::debug = false;
    CourtLinePixelDetector::debug = false;
    CourtLineCandidateDetector::debug = false;
    TennisCourtFitter::debug = false;
    const auto started = std::chrono::steady_clock::now();
    cv::Mat frame = cv::imread(argv[1]);
    if (frame.empty()) return 2;
    CourtLinePixelDetector pixels;
    CourtLineCandidateDetector lines;
    TennisCourtFitter fitter;
    const auto loaded = std::chrono::steady_clock::now();
    try {
        cv::Mat binary = pixels.run(frame);
        auto candidates = lines.run(binary, frame);
        auto court = fitter.run(candidates, binary, frame);
        const auto finished = std::chrono::steady_clock::now();
        court.writeToFile(argv[2]);
        std::cout << std::setprecision(12) << "BENCH_TIMING "
                  << std::chrono::duration<double>(loaded - started).count() << " "
                  << std::chrono::duration<double>(finished - loaded).count() << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 3;
    }
    return 0;
}
