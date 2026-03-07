#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cstring>
#include "constants.cuh"
#include "preprocess.h"

static void printUsage(const char *prog) {
    std::cerr << "Usage: " << prog << " [--threads N] [--benchmarking]\n"
              << "  --threads N      Number of worker threads (default: hardware concurrency)\n"
              << "  --benchmarking   Skip writing the output CSV file\n";
}

int main(int argc, char *argv[]) {
    // ── Parse CLI arguments ─────────────────────────────────────────────
    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;  // fallback
    bool benchmarking = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                numThreads = static_cast<unsigned>(std::stoul(argv[++i]));
                if (numThreads == 0) numThreads = 1;
            } else {
                printUsage(argv[0]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--benchmarking") == 0) {
            benchmarking = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    auto start_global = std::chrono::steady_clock::now();

    // ── Stage 1: Read input data from disk ──────────────────────────────
    std::string filename = DEFAULT_INPUT_FILE;
    std::vector<std::string> seqs;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return 1;
    }
    std::string line;
    while (std::getline(infile, line)) {
        seqs.push_back(line);
    }
    
    auto elapsed_1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_global);
    std::cout << "Reading data from disk finished in: " << elapsed_1.count() << "us" << std::endl;

    // ── Stage 2: Tokenization (CPU, multithreaded) ──────────────────────
    auto start_compute = std::chrono::steady_clock::now();
    Preprocessor preprocessor;
    std::cout << "Using " << numThreads << " thread(s) for tokenization" << std::endl;
    auto b_tokenized = preprocessor.preprocessBatchMT(seqs, MAX_TOKENS, numThreads);
    auto start_write = std::chrono::steady_clock::now();
    auto elapsed_compute = std::chrono::duration_cast<std::chrono::microseconds>(start_write - start_compute);
    std::cout << "CPU Tokenization finished in: " << elapsed_compute.count() << "us" << std::endl;
    std::cout << "Tokenizing + Indexing finished in: " << elapsed_compute.count() << "us" << std::endl;
    std::cout << "End-to-end (excl. I/O) finished in: " << elapsed_compute.count() << "us" << std::endl;

    // ── Stage 3: Write output to CSV ────────────────────────────────────
    if (!benchmarking) {
        std::string out_filename = "output/output_cpu.csv";
        std::ofstream ofs(out_filename);
        if (!ofs.is_open()) {
            std::cerr << "Error: Cannot open output file " << out_filename << std::endl;
            return 1;
        }
        for (size_t i = 0; i < b_tokenized.size(); i++) {
            for (size_t j = 0; j < (size_t)MAX_TOKENS; j++) {
                if (j < b_tokenized[i].size()) {
                    ofs << static_cast<int>(b_tokenized[i][j]);
                } else {
                    ofs << 0;
                }
                if (j + 1 < (size_t)MAX_TOKENS) {
                    ofs << ",";
                }
            }
            ofs << "\n";
        }
        auto elapsed_write = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_write);
        std::cout << "Writing output to host finished in: " << elapsed_write.count() << "us" << std::endl;
    } else {
        std::cout << "Benchmarking mode: skipping output write.\n";
    }

    return 0;
}
