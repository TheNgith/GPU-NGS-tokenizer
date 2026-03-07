/**
 * @file main.cu
 * @brief GPU-accelerated DNA sequence trigram tokenizer.
 *
 * Pipeline overview:
 *   1. Read DNA sequences from a text file (one sequence per line).
 *   2. Read a 256-entry lookup table (LUT) that maps packed trigram
 *      bit-patterns to integer token IDs.
 *   3. Upload sequences, offsets, and LUT to GPU memory.
 *   4. Launch one CUDA block per sequence.  Each block uses shared memory
 *      to assemble 3-character trigrams as packed uint32 values, then
 *      extracts selected bits and looks up the token ID via the LUT
 *      stored in constant memory.
 *   5. Copy the tokenized output back to the host and write it to CSV.
 */

#include <iostream>
#include "utils.cuh"
#include "cudaErr.cuh"
#include "constants.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include <cstring>

/** LUT in CUDA constant memory – 256 entries, one per possible trigram hash. */
__constant__ int8_t d_lut[256];

/**
 * @brief CUDA kernel: tokenize DNA sequences into trigram IDs.
 *
 * Each block processes one sequence.  Threads cooperatively load characters
 * into shared memory, assemble 3-character trigrams as packed uint32 words,
 * extract a compressed 8-bit hash from selected bits of the three ASCII
 * bytes, and use that hash to index into the constant-memory LUT to obtain
 * the final token ID.
 *
 * Shared-memory layout (4 bytes per position):
 *   - Round 1: each thread i copies char[i] to shared[i*4].
 *   - Round 2: each thread copies the next character into shared[i*4+1].
 *   - Round 3: each thread copies the character after that into shared[i*4+2].
 *   After these rounds, reinterpreting shared as uint32_t gives the packed
 *   trigram for position i.
 *
 * Bit-extraction formula (tok_shifted):
 *   bits[7:5] = char[2] bits [2:0]     (3rd character, low 3 bits)
 *   bits[4:3] = char[1] bits [2:1]     (2nd character, bits 2-1)
 *   bits[2:0] = char[0] bits [2:0]     (1st character, low 3 bits)
 *
 * @param d_all_seqs      Flat buffer of all concatenated sequences.
 * @param offsets          Byte offsets for each sequence (length num_seq+1).
 * @param d_output         [out] Raw packed trigrams     (num_seq × MAX_TOKENS).
 * @param d_output_shifted [out] LUT-mapped token IDs    (num_seq × MAX_TOKENS).
 * @param num_seq          Number of sequences.
 */
__global__
void tokenize(char* d_all_seqs, size_t* offsets, uint32_t* d_output, int8_t* d_output_shifted, size_t num_seq) {
    __shared__ char shared[(MAX_TOKENS+N_GRAMS)*4];

    size_t seq_id = blockIdx.x;
    size_t i = threadIdx.x;
    size_t start = offsets[seq_id];
    size_t length = offsets[seq_id + 1] - start;

    if (i >= min((size_t)(MAX_TOKENS + N_GRAMS), length)) return;

    // Round 1: load one character per thread into every 4th byte of shared mem
    __syncthreads();
    shared[i*4] = d_all_seqs[start + i];

    // Round 2: copy the next character to form the 2nd byte of each packed word
    __syncthreads();
    shared[i*4 + 1] = shared[(i+1)*4];
    
    // Round 3: copy the character after that to form the 3rd byte
    __syncthreads();
    shared[i*4 + 2] = shared[(i+2)*4];

    // Reinterpret shared memory as packed uint32 trigrams
    __syncthreads();
    uint32_t* shared_int = reinterpret_cast<uint32_t*>(shared);
    
    uint32_t tok = shared_int[i];
    uint8_t tok_shifted = (((tok >> 16) & 0x07) << 5) | ((((tok >> 8) & 0x06) >> 1) << 3) | ((tok & 0x07));
    
    if (i >= MAX_TOKENS || i >= length - N_GRAMS + 1) return;
    d_output[seq_id * MAX_TOKENS + i] = tok;
    d_output_shifted[seq_id * MAX_TOKENS + i] = d_lut[tok_shifted];
    
    return;
}


int main(int argc, char* argv[]) {
    bool benchmarking = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--benchmarking") == 0) {
            benchmarking = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::cerr << "Usage: " << argv[0] << " [--benchmarking]\n";
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            std::cerr << "Usage: " << argv[0] << " [--benchmarking]\n";
            return 1;
        }
    }
    auto start_global = std::chrono::steady_clock::now();

    // ── Stage 1: Read input data from disk ──────────────────────────────
    std::string text;
    std::vector<size_t> offsets; // n+1 long
    read_file(DEFAULT_INPUT_FILE, text, offsets, false);
    std::vector<int8_t> lut = read_lut_file("build/LUT.txt");
    size_t num_seqs = offsets.size() - 1;

    auto elapsed_1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_global);
    std::cout << "Reading data from disk finished in: " << elapsed_1.count() << "us" << std::endl;

    // ── Stage 2: Upload data to GPU ─────────────────────────────────────
    auto start_e2e = std::chrono::steady_clock::now();
    cudaMemcpyToSymbol(d_lut, lut.data(), 256*sizeof(int8_t), 0, cudaMemcpyHostToDevice);

    char* d_all_seqs;
    cudaMalloc(&d_all_seqs, text.size());
    cudaMemcpy(d_all_seqs, text.data(), text.size(), cudaMemcpyHostToDevice);

    size_t* d_offsets;
    cudaMalloc(&d_offsets, offsets.size()*sizeof(size_t));
    cudaMemcpy(d_offsets, offsets.data(), offsets.size()*sizeof(size_t), cudaMemcpyHostToDevice);

    uint32_t* d_output;
    uint32_t* h_output = new uint32_t[MAX_TOKENS*num_seqs];
    cudaMalloc(&d_output, MAX_TOKENS*num_seqs*sizeof(int));

    int8_t* d_output_shifted;
    int8_t* h_output_shifted = new int8_t[MAX_TOKENS*num_seqs];
    cudaMalloc(&d_output_shifted, MAX_TOKENS*num_seqs*sizeof(int8_t));
    cudaMemset(d_output_shifted, 0, MAX_TOKENS*num_seqs*sizeof(int8_t));

    auto elapsed_2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_global);
    std::cout << "Setting up data in GPU finished in: " << elapsed_2.count() - elapsed_1.count() << "us" << std::endl;

    // ── Stage 3: Launch tokenization kernel ─────────────────────────────
    size_t num_threads = 1024;
    size_t num_blocks = num_seqs;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    tokenize<<<num_blocks, num_threads>>>(d_all_seqs, d_offsets, d_output, d_output_shifted, num_seqs);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "GPU kernel finished in: " << gpuTime*1000 << "us" << std::endl;
    
    CUDA_CHECK(cudaGetLastError());

    // ── Stage 4: Copy results back to host ──────────────────────────────
    cudaMemcpy(h_output, d_output, MAX_TOKENS*num_seqs*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_shifted, d_output_shifted, MAX_TOKENS*num_seqs*sizeof(int8_t), cudaMemcpyDeviceToHost);
    auto elapsed_3 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_global);
    std::cout << "Transferring output to host finished in: " << elapsed_3.count() - elapsed_2.count() - gpuTime*1000 << "us" << std::endl;

    auto elapsed_e2e = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_e2e);
    std::cout << "End-to-end (excl. I/O) finished in: " << elapsed_e2e.count() << "us" << std::endl;

    // ── Independent experiment: raw D2H copy time for d_output_shifted ──
    {
        int8_t* h_raw_copy = new int8_t[MAX_TOKENS * num_seqs];
        cudaEvent_t d2h_start, d2h_stop;
        cudaEventCreate(&d2h_start);
        cudaEventCreate(&d2h_stop);
        cudaEventRecord(d2h_start, 0);
        cudaMemcpy(h_raw_copy, d_output_shifted,
                   MAX_TOKENS * num_seqs * sizeof(int8_t),
                   cudaMemcpyDeviceToHost);
        cudaEventRecord(d2h_stop, 0);
        cudaEventSynchronize(d2h_stop);
        float d2hTime;
        cudaEventElapsedTime(&d2hTime, d2h_start, d2h_stop);
        std::cout << "[Experiment] D2H copy of int8_t d_output_shifted: "
                  << d2hTime * 1000 << "us" << std::endl;
        cudaEventDestroy(d2h_start);
        cudaEventDestroy(d2h_stop);
        delete[] h_raw_copy;
    }

    // ── Stage 5: Write output to CSV ────────────────────────────────────
    if (!benchmarking) {
        write_int8_flat_to_csv(h_output_shifted, MAX_TOKENS*num_seqs, "output/output.csv");
    } else {
        std::cout << "Benchmarking mode: skipping output write.\n";
    }

    // ── Cleanup ─────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_all_seqs));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_output));

    delete[] h_output;

    return 0;
}