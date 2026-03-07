/**
 * @file naive_tokenizer.cu
 * @brief PyTorch CUDA extension that wraps the naive GPU tokenization kernel.
 *
 * Reads a text file, uploads data to GPU, runs the naive CUDA tokenization
 * kernel (from naive_preprocess.cuh), and returns the result as a
 * torch::Tensor on CUDA device.  Data stays on GPU — zero-copy handoff
 * to PyTorch.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <tuple>

// Project headers (copies in csrc/)
#include "constants.cuh"
#include "naive_preprocess.cuh"

// ─── Host-side file reader (with '<'/'>' sentinels) ─────────────────────────
static bool naive_read_file(
    const std::string& filename,
    std::string& allText,
    std::vector<int>& offsets)
{
    std::ifstream infile(filename);
    if (!infile.is_open()) return false;

    std::string line;
    int currentOffset = 0;

    while (std::getline(infile, line)) {
        offsets.push_back(currentOffset);
        allText += '<';
        allText += line;
        allText += '>';
        currentOffset += static_cast<int>(line.size()) + 2;
    }
    offsets.push_back(currentOffset);  // trailing entry so offsets is n+1 long
    return true;
}

/**
 * Tokenize sequences from a text file on GPU using the naive CUDA kernel.
 *
 * @param input_path  Path to the input file (one sequence per line).
 * @return std::tuple<torch::Tensor, double> containing:
 *         1. torch::Tensor of shape (num_seqs, MAX_TOKENS), dtype int64, on CUDA.
 *         2. double representing the file I/O time in microseconds.
 */
std::tuple<torch::Tensor, double> tokenize_naive_cuda(const std::string& input_path) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // ── Stage 1: Read data on host ──────────────────────────────────────
    std::string text;
    std::vector<int> offsets;
    TORCH_CHECK(naive_read_file(input_path, text, offsets),
                "Cannot open input file: ", input_path);

    int num_sequences = static_cast<int>(offsets.size()) - 1;
    TORCH_CHECK(num_sequences > 0, "No sequences found in: ", input_path);

    auto t1 = std::chrono::high_resolution_clock::now();
    double io_time_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // ── Stage 2: Upload lookup tables to GPU constant memory ────────────
    naive_copy_tables_to_device();

    // ── Stage 3: Allocate GPU memory and copy data ──────────────────────
    char* d_sequences;
    cudaMalloc(&d_sequences, text.size());
    cudaMemcpy(d_sequences, text.data(), text.size(), cudaMemcpyHostToDevice);

    int* d_offsets;
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
    cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    int* d_output;
    cudaMalloc(&d_output, num_sequences * MAX_TOKENS * sizeof(int));
    cudaMemset(d_output, 0, num_sequences * MAX_TOKENS * sizeof(int));

    // ── Stage 4: Launch kernel ──────────────────────────────────────────
    int threads_per_block = 256;
    int num_blocks = (num_sequences + threads_per_block - 1) / threads_per_block;
    naive_tokenizeAndEncodeBatch<<<num_blocks, threads_per_block>>>(
        d_sequences, d_offsets, d_output, num_sequences, MAX_TOKENS);
    cudaDeviceSynchronize();

    // ── Stage 5: Wrap GPU buffer as torch::Tensor (ZERO-COPY) ───────────
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    auto d_seq_ptr = d_sequences;
    auto d_off_ptr = d_offsets;
    auto d_out_ptr = d_output;

    auto raw = torch::from_blob(
        d_output,
        {static_cast<int64_t>(num_sequences), static_cast<int64_t>(MAX_TOKENS)},
        /*deleter=*/[d_seq_ptr, d_off_ptr, d_out_ptr](void*) {
            cudaFree(d_seq_ptr);
            cudaFree(d_off_ptr);
            cudaFree(d_out_ptr);
        },
        options
    );

    // Cast to int64 for nn.Embedding input — stays on GPU, no D2H transfer.
    return std::make_tuple(raw.to(torch::kInt64), io_time_us);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tokenize_naive_cuda", &tokenize_naive_cuda,
          "Tokenize sequences on GPU using naive CUDA kernel (zero-copy).\n"
          "Args:\n"
          "  input_path (str): Path to the input file.\n"
          "Returns:\n"
          "  Tuple (Tensor, double) containing:\n"
          "    1. Tensor of shape (N, MAX_TOKENS), dtype int64, on CUDA.\n"
          "    2. File I/O time in microseconds.",
          py::arg("input_path"));
}
