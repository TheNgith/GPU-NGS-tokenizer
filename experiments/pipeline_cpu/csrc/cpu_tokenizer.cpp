/**
 * @file cpu_tokenizer.cpp
 * @brief PyTorch C++ extension that wraps the CPU tokenization pipeline.
 *
 * Reads a text file, tokenizes via Preprocessor::preprocessBatchMT()
 * (from preprocess.h), and returns the result as a torch::Tensor on CPU.
 * Data stays in host memory — zero-copy handoff to PyTorch.
 */

#include <torch/extension.h>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <tuple>

// Project headers (copies in csrc/)
#include "constants.cuh"
#include "preprocess.h"

/**
 * Tokenize sequences from a text file on CPU.
 *
 * @param input_path  Path to the input file (one sequence per line).
 * @return A tuple containing:
 *         1. torch::Tensor of shape (num_seqs, MAX_TOKENS), dtype int64, on CPU.
 *         2. double representing the file I/O time in microseconds.
 *         3. double representing the tokenizing + indexing time in microseconds.
 */
std::tuple<torch::Tensor, double, double> tokenize_cpu(const std::string& input_path, int num_threads) {
    auto t0 = std::chrono::high_resolution_clock::now();
    // ── Read sequences ──────────────────────────────────────────────────
    std::vector<std::string> seqs;
    std::ifstream infile(input_path);
    TORCH_CHECK(infile.is_open(), "Cannot open file: ", input_path);
    std::string line;
    while (std::getline(infile, line)) {
        seqs.push_back(line);
    }
    infile.close();
    TORCH_CHECK(!seqs.empty(), "No sequences found in: ", input_path);

    auto t1 = std::chrono::high_resolution_clock::now();
    double io_time_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // ── Tokenize (multithreaded CPU) ────────────────────────────────────
    auto t_compute_start = std::chrono::high_resolution_clock::now();
    Preprocessor preprocessor;
    auto b_tokenized = preprocessor.preprocessBatchMT(seqs, MAX_TOKENS, num_threads);
    auto t_compute_end = std::chrono::high_resolution_clock::now();
    double compute_time_us = std::chrono::duration<double, std::micro>(t_compute_end - t_compute_start).count();

    const int64_t N = static_cast<int64_t>(b_tokenized.size());
    const int64_t L = MAX_TOKENS;

    // ── Build contiguous int64 tensor directly ──────────────────────────
    // Allocate a CPU tensor and fill it — data lives on host, no copy needed
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    torch::Tensor result = torch::zeros({N, L}, options);
    auto accessor = result.accessor<int64_t, 2>();

    for (int64_t i = 0; i < N; ++i) {
        const auto& row = b_tokenized[i];
        int64_t fill_len = std::min(static_cast<int64_t>(row.size()), L);
        for (int64_t j = 0; j < fill_len; ++j) {
            accessor[i][j] = static_cast<int64_t>(row[j]);
        }
        // Remaining positions stay 0 (padding) from torch::zeros
    }

    return std::make_tuple(result, io_time_us, compute_time_us);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tokenize", &tokenize_cpu,
          "Tokenize sequences from a text file on CPU.\n"
          "Args:\n"
          "  input_path (str): Path to the input file.\n"
          "  num_threads (int): Number of tokenization threads.\n"
          "Returns:\n"
          "  Tuple (Tensor, double, double) containing:\n"
          "    1. Tensor of shape (N, MAX_TOKENS), dtype int64, on CPU.\n"
          "    2. File I/O time in microseconds.\n"
          "    3. Tokenizing + indexing time in microseconds.",
          py::arg("input_path"),
          py::arg("num_threads") = 1);
}
