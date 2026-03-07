/**
 * @file gpu_tokenizer.cu
 * @brief PyTorch CUDA extension that wraps the GPU tokenization kernel.
 *
 * Reads a text file, uploads data to GPU, runs the CUDA tokenization kernel
 * (from main.cu), and returns the result as a torch::Tensor on CUDA device.
 * Data stays on GPU — zero-copy handoff to PyTorch.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <tuple>

// Project headers (copies in csrc/)
#include "constants.cuh"
#include "cudaErr.cuh"

// ─── LUT in constant memory ─────────────────────────────────────────────────
__constant__ int8_t d_lut[256];

// ─── Tokenization kernel (copied from src/main.cu) ──────────────────────────
__global__
void tokenize_kernel(char* d_all_seqs, size_t* offsets, uint32_t* d_output,
                     int8_t* d_output_shifted, size_t num_seq) {
    __shared__ char shared[(MAX_TOKENS + N_GRAMS) * 4];

    size_t seq_id = blockIdx.x;
    size_t i = threadIdx.x;
    size_t start = offsets[seq_id];
    size_t length = offsets[seq_id + 1] - start;

    if (i >= min((size_t)(MAX_TOKENS + N_GRAMS), length)) return;

    // Round 1: load one character per thread into every 4th byte of shared mem
    __syncthreads();
    shared[i * 4] = d_all_seqs[start + i];

    // Round 2: copy the next character to form the 2nd byte of each packed word
    __syncthreads();
    shared[i * 4 + 1] = shared[(i + 1) * 4];

    // Round 3: copy the character after that to form the 3rd byte
    __syncthreads();
    shared[i * 4 + 2] = shared[(i + 2) * 4];

    // Reinterpret shared memory as packed uint32 trigrams
    __syncthreads();
    uint32_t* shared_int = reinterpret_cast<uint32_t*>(shared);

    uint32_t tok = shared_int[i];
    uint8_t tok_shifted = (((tok >> 16) & 0x07) << 5) |
                          ((((tok >> 8) & 0x06) >> 1) << 3) |
                          ((tok & 0x07));

    if (i >= MAX_TOKENS || i >= length - N_GRAMS + 1) return;
    d_output[seq_id * MAX_TOKENS + i] = tok;
    d_output_shifted[seq_id * MAX_TOKENS + i] = d_lut[tok_shifted];
}

// ─── Host-side file reader (from utils.cuh) ─────────────────────────────────
static bool read_file(
    const std::string& filename,
    std::string& allText,
    std::vector<size_t>& offsets)
{
    std::ifstream infile(filename);
    if (!infile.is_open()) return false;

    std::string line;
    size_t currentOffset = 0;
    offsets.clear();
    offsets.push_back(0);

    while (std::getline(infile, line)) {
        allText += 'B';            // start sentinel
        allText += line;
        allText += 'E';            // end sentinel
        currentOffset += static_cast<size_t>(line.size()) + 2;
        offsets.push_back(currentOffset);
    }
    return true;
}

// ─── LUT file reader (from utils.cuh) ───────────────────────────────────────
static std::vector<int8_t> read_lut_file(const std::string& filename) {
    std::vector<int8_t> values;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open LUT file: " + filename);
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        int temp;
        ss >> temp;
        if (ss.fail()) throw std::runtime_error("Invalid integer in LUT file");
        values.push_back(static_cast<int8_t>(temp));
    }
    return values;
}

/**
 * Tokenize sequences from a text file on GPU using the CUDA kernel.
 *
 * @param input_path  Path to the input file (one sequence per line).
 * @param lut_path    Path to the LUT.txt file (256 entries).
 * @return std::tuple<torch::Tensor, double> containing:
 *         1. torch::Tensor of shape (num_seqs, MAX_TOKENS), dtype int64, on CUDA.
 *         2. double representing the file I/O time in microseconds.
 */
std::tuple<torch::Tensor, double> tokenize_cuda(const std::string& input_path, const std::string& lut_path) {
    auto t0 = std::chrono::high_resolution_clock::now();
    // ── Stage 1: Read data on host ──────────────────────────────────────
    std::string text;
    std::vector<size_t> offsets;
    TORCH_CHECK(read_file(input_path, text, offsets),
                "Cannot open input file: ", input_path);
    auto lut = read_lut_file(lut_path);
    TORCH_CHECK(lut.size() == 256, "LUT must have 256 entries, got ", lut.size());

    size_t num_seqs = offsets.size() - 1;
    TORCH_CHECK(num_seqs > 0, "No sequences found in: ", input_path);

    auto t1 = std::chrono::high_resolution_clock::now();
    double io_time_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // ── Stage 2: Upload to GPU ──────────────────────────────────────────
    cudaMemcpyToSymbol(d_lut, lut.data(), 256 * sizeof(int8_t), 0, cudaMemcpyHostToDevice);

    char* d_all_seqs;
    cudaMalloc(&d_all_seqs, text.size());
    cudaMemcpy(d_all_seqs, text.data(), text.size(), cudaMemcpyHostToDevice);

    size_t* d_offsets;
    cudaMalloc(&d_offsets, offsets.size() * sizeof(size_t));
    cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    uint32_t* d_output;
    cudaMalloc(&d_output, MAX_TOKENS * num_seqs * sizeof(uint32_t));

    int8_t* d_output_shifted;
    cudaMalloc(&d_output_shifted, MAX_TOKENS * num_seqs * sizeof(int8_t));
    cudaMemset(d_output_shifted, 0, MAX_TOKENS * num_seqs * sizeof(int8_t));

    // ── Stage 3: Launch kernel ──────────────────────────────────────────
    size_t num_threads = 1024;
    size_t num_blocks = num_seqs;
    tokenize_kernel<<<num_blocks, num_threads>>>(
        d_all_seqs, d_offsets, d_output, d_output_shifted, num_seqs);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // ── Stage 4: Wrap GPU buffer as torch::Tensor (ZERO-COPY) ───────────
    // d_output_shifted is int8_t on GPU. We wrap it and cast to int64.
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);

    // Use from_blob with a custom deleter that frees all GPU allocations
    // when the tensor is no longer needed.
    auto d_seqs_ptr = d_all_seqs;
    auto d_off_ptr = d_offsets;
    auto d_out_ptr = d_output;
    auto d_shifted_ptr = d_output_shifted;

    // Keep `raw` alive as a named local so the custom deleter (cudaFree) does
    // not fire until after the async .to(kInt64) GPU copy has completed.
    auto raw = torch::from_blob(
        d_output_shifted,
        {static_cast<int64_t>(num_seqs), static_cast<int64_t>(MAX_TOKENS)},
        /*deleter=*/[d_seqs_ptr, d_off_ptr, d_out_ptr, d_shifted_ptr](void*) {
            cudaFree(d_seqs_ptr);
            cudaFree(d_off_ptr);
            cudaFree(d_out_ptr);
            cudaFree(d_shifted_ptr);
        },
        options
    );

    // Cast to int64 for nn.Embedding input — stays on GPU, no D2H transfer.
    // `raw` is still alive here, so the deleter fires only after this returns.
    return std::make_tuple(raw.to(torch::kInt64), io_time_us);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tokenize_cuda", &tokenize_cuda,
          "Tokenize sequences on GPU using CUDA kernel (zero-copy).\n"
          "Args:\n"
          "  input_path (str): Path to the input file.\n"
          "  lut_path (str): Path to the LUT.txt file.\n"
          "Returns:\n"
          "  Tuple (Tensor, double) containing:\n"
          "    1. Tensor of shape (N, MAX_TOKENS), dtype int64, on CUDA.\n"
          "    2. File I/O time in microseconds.",
          py::arg("input_path"),
          py::arg("lut_path"));
}
