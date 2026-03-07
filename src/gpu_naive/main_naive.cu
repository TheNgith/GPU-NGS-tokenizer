/**
 * @file main_naive.cu
 * @brief Naive GPU tokenization pipeline driver (CUDA).
 *
 * Implements the same pipeline as main_naive.cpp but runs tokenization
 * on the GPU using the CUDA kernels extracted from
 * model_inference_GPU/src/inference.cu:
 *
 *   __device__  naive_trigramToNumber()
 *   __global__  naive_tokenizeAndEncodeBatch()
 *
 * Sentinel convention: '<' (begin) and '>' (end) — exactly as in
 * inference.cu's buildChar2Idx() — are injected into the flat buffer here.
 *
 * Output: output/output_naive.csv   (same shape as output_cpu.csv)
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include "constants.cuh"
#include "naive_preprocess.cuh"

static void printUsage(const char* prog)
{
    std::cerr << "Usage: " << prog << " [--threads N] [--benchmarking]\n"
              << "  --threads N      (threads flag accepted but ignored; naive pipeline uses GPU)\n"
              << "  --benchmarking   Skip writing the output CSV file\n";
}

// ── Local file reader that injects '<' / '>' sentinels ──────────────────────
// Mirrors read_file() from utils.cuh but uses '<'/'>' instead of 'B'/'E'
// so that the naive char2idx ('<'→0, '>'→5) works without modification.
static bool naive_read_file(
    const std::string&    filename,
    std::string&          allText,
    std::vector<int>&     offsets)
{
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << "\n";
        return false;
    }

    std::string line;
    int currentOffset = 0;

    while (std::getline(infile, line)) {
        offsets.push_back(currentOffset);
        allText += '<';                                       // start sentinel
        allText += line;
        allText += '>';                                       // end   sentinel
        currentOffset += static_cast<int>(line.size()) + 2;
    }
    offsets.push_back(currentOffset);  // trailing entry so offsets is n+1 long

    return true;
}

int main(int argc, char* argv[])
{
    bool benchmarking = false;
    // ── Parse CLI (mirrors main_cpu.cpp; --threads is accepted but ignored) ──
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) { ++i; }  // consume the value, ignore it
            else { printUsage(argv[0]); return 1; }
        } else if (std::strcmp(argv[i], "--benchmarking") == 0) {
            benchmarking = true;
        } else if (std::strcmp(argv[i], "--help") == 0 ||
                   std::strcmp(argv[i], "-h")     == 0) {
            printUsage(argv[0]); return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            printUsage(argv[0]); return 1;
        }
    }

    auto start_global = std::chrono::steady_clock::now();

    // ── Stage 1: Read input data from disk ──────────────────────────────────
    const std::string filename = DEFAULT_INPUT_FILE;

    std::string      allText;
    std::vector<int> offsets;

    if (!naive_read_file(filename, allText, offsets)) return 1;

    int num_sequences = static_cast<int>(offsets.size()) - 1;  // offsets is n+1 long

    auto elapsed_1 = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_global);
    std::cout << "Reading data from disk finished in: "
              << elapsed_1.count() << "us\n";

    // ── Stage 2: Build and upload lookup tables to GPU constant memory ──────
    auto start_e2e = std::chrono::steady_clock::now();
    naive_copy_tables_to_device();

    // ── Stage 3: Allocate GPU memory and copy data ──────────────────────────
    char* d_sequences;
    int*  d_offsets;
    int*  d_output;

    cudaMalloc(&d_sequences, allText.size());
    cudaMalloc(&d_offsets,   (num_sequences + 1) * sizeof(int));
    cudaMalloc(&d_output,    num_sequences * MAX_TOKENS * sizeof(int));

    cudaMemcpy(d_sequences, allText.data(), allText.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets,   offsets.data(), (num_sequences + 1) * sizeof(int), cudaMemcpyHostToDevice);

    auto elapsed_2 = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_global);
    std::cout << "Setting up data in GPU finished in: "
              << elapsed_2.count() - elapsed_1.count() << "us\n";

    // ── Stage 4: Launch tokenization kernel ─────────────────────────────────
    int threadsPerBlock = 256;
    int blocks = (num_sequences + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    naive_tokenizeAndEncodeBatch<<<blocks, threadsPerBlock>>>(
        d_sequences, d_offsets, d_output, num_sequences, MAX_TOKENS);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "Naive GPU kernel finished in: " << gpuTime * 1000 << "us\n";

    // ── Stage 5: Copy results back to host ──────────────────────────────────
    int* h_output = new int[num_sequences * MAX_TOKENS];
    cudaMemcpy(h_output, d_output, num_sequences * MAX_TOKENS * sizeof(int),
               cudaMemcpyDeviceToHost);

    auto elapsed_3 = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_global);
    std::cout << "Transferring output to host finished in: "
              << elapsed_3.count() - elapsed_2.count() - gpuTime * 1000 << "us\n";

    auto elapsed_e2e = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_e2e);
    std::cout << "End-to-end (excl. I/O) finished in: " << elapsed_e2e.count() << "us\n";

    // ── Independent experiment: raw D2H copy time for d_output ───────────────
    {
        int* h_raw_copy = new int[num_sequences * MAX_TOKENS];
        cudaEvent_t d2h_start, d2h_stop;
        cudaEventCreate(&d2h_start);
        cudaEventCreate(&d2h_stop);
        cudaEventRecord(d2h_start, 0);
        cudaMemcpy(h_raw_copy, d_output,
                   num_sequences * MAX_TOKENS * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaEventRecord(d2h_stop, 0);
        cudaEventSynchronize(d2h_stop);
        float d2hTime;
        cudaEventElapsedTime(&d2hTime, d2h_start, d2h_stop);
        std::cout << "[Experiment] D2H copy of int d_output: "
                  << d2hTime * 1000 << "us\n";
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_stop);
        delete[] h_raw_copy;
    }

    // ── Stage 6: Write output to CSV ─────────────────────────────────────────
    auto start_write = std::chrono::steady_clock::now();

    if (!benchmarking) {
        const std::string out_filename = "output/output_naive.csv";
        std::ofstream ofs(out_filename);
        if (!ofs.is_open()) {
            std::cerr << "Error: Cannot open output file " << out_filename << "\n";
            delete[] h_output;
            return 1;
        }

        for (int i = 0; i < num_sequences; ++i) {
            for (int j = 0; j < MAX_TOKENS; ++j) {
                ofs << h_output[i * MAX_TOKENS + j];
                if (j + 1 < MAX_TOKENS) ofs << ',';
            }
            ofs << '\n';
        }

        auto elapsed_write = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_write);
        std::cout << "Writing output to host finished in: "
                  << elapsed_write.count() << "us\n";
    } else {
        std::cout << "Benchmarking mode: skipping output write.\n";
    }

    // ── Cleanup ─────────────────────────────────────────────────────────────
    cudaFree(d_sequences);
    cudaFree(d_offsets);
    cudaFree(d_output);
    delete[] h_output;

    return 0;
}
