/**
 * @file naive_preprocess.cuh
 * @brief Naive GPU tokenization pipeline (CUDA).
 *
 * Extracts the two CUDA kernels from model_inference_GPU/src/inference.cu
 * as an independent module, preserving their original GPU logic:
 *
 *   __device__  trigramToNumber()      → naive_trigramToNumber()
 *   __global__  tokenizeAndEncodeBatch → naive_tokenizeAndEncodeBatch()
 *
 * Data conventions (matching inference.cu):
 *   - Sequences are read flat with '<' (begin) and '>' (end) sentinels.
 *   - char2idx maps exactly: '<'→0 'A'→1 'C'→2 'G'→3 'T'→4 '>'→5
 *     (as in build_LUT.hpp / inference.cu buildChar2Idx).
 *   - LUT[216] is built from the same trigram_code.hpp dictionary via
 *     hash x*36 + y*6 + z (as in build_LUT.hpp buildLUT).
 *   - MAX_TOKENS and N_GRAMS come from constants.cuh.
 *   - Output values are int (token IDs 0-95, or 0 for padding).
 */

#pragma once

#include <cstdint>
#include <array>
#include "constants.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Constant memory for the naive pipeline (GPU)
// Mirrors d_char2idx / d_LUT from inference.cu
// ─────────────────────────────────────────────────────────────────────────────

__constant__ int d_naive_char2idx[256];
__constant__ int d_naive_LUT[216];

// ─────────────────────────────────────────────────────────────────────────────
// Inline dictionary (mirrors trigram_code.hpp)
// ─────────────────────────────────────────────────────────────────────────────

struct NaiveTrigramCode {
    const char* trigram;
    int         code;
};

static const int NAIVE_NUM_CODES = 96;

// Exact copy of trigram_code.hpp dictionary
static const NaiveTrigramCode naive_dictionary[NAIVE_NUM_CODES] = {
    {"<AA", 0},  {"<AC", 1},  {"<AG", 2},  {"<AT", 3},
    {"<CA", 4},  {"<CC", 5},  {"<CG", 6},  {"<CT", 7},
    {"<GA", 8},  {"<GC", 9},  {"<GG",10},  {"<GT",11},
    {"<TA",12},  {"<TC",13},  {"<TG",14},  {"<TT",15},

    {"AA>",16}, {"AAA",17}, {"AAC",18}, {"AAG",19}, {"AAT",20},
    {"AC>",21}, {"ACA",22}, {"ACC",23}, {"ACG",24}, {"ACT",25},
    {"AG>",26}, {"AGA",27}, {"AGC",28}, {"AGG",29}, {"AGT",30},
    {"AT>",31}, {"ATA",32}, {"ATC",33}, {"ATG",34}, {"ATT",35},

    {"CA>",36}, {"CAA",37}, {"CAC",38}, {"CAG",39}, {"CAT",40},
    {"CC>",41}, {"CCA",42}, {"CCC",43}, {"CCG",44}, {"CCT",45},
    {"CG>",46}, {"CGA",47}, {"CGC",48}, {"CGG",49}, {"CGT",50},
    {"CT>",51}, {"CTA",52}, {"CTC",53}, {"CTG",54}, {"CTT",55},

    {"GA>",56}, {"GAA",57}, {"GAC",58}, {"GAG",59}, {"GAT",60},
    {"GC>",61}, {"GCA",62}, {"GCC",63}, {"GCG",64}, {"GCT",65},
    {"GG>",66}, {"GGA",67}, {"GGC",68}, {"GGG",69}, {"GGT",70},
    {"GT>",71}, {"GTA",72}, {"GTC",73}, {"GTG",74}, {"GTT",75},

    {"TA>",76}, {"TAA",77}, {"TAC",78}, {"TAG",79}, {"TAT",80},
    {"TC>",81}, {"TCA",82}, {"TCC",83}, {"TCG",84}, {"TCT",85},
    {"TG>",86}, {"TGA",87}, {"TGC",88}, {"TGG",89}, {"TGT",90},
    {"TT>",91}, {"TTA",92}, {"TTC",93}, {"TTG",94}, {"TTT",95},
};

// ─────────────────────────────────────────────────────────────────────────────
// Host helpers: build the char2idx and LUT arrays, then copy to __constant__
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build the char2idx[256] mapping on the host.
 * '<'→0 'A'→1 'C'→2 'G'→3 'T'→4 '>'→5, all others → -1.
 */
inline std::array<int, 256> build_naive_char2idx()
{
    std::array<int, 256> c2i;
    c2i.fill(-1);

    c2i[(unsigned char)'<'] = 0;
    c2i[(unsigned char)'A'] = 1;
    c2i[(unsigned char)'C'] = 2;
    c2i[(unsigned char)'G'] = 3;
    c2i[(unsigned char)'T'] = 4;
    c2i[(unsigned char)'>'] = 5;

    return c2i;
}

/**
 * Build the LUT[216] (6³ entries) on the host, using hash x*36 + y*6 + z.
 */
inline std::array<int, 216> build_naive_lut()
{
    std::array<int, 256> c2i;
    c2i.fill(-1);
    c2i[(unsigned char)'<'] = 0;
    c2i[(unsigned char)'A'] = 1;
    c2i[(unsigned char)'C'] = 2;
    c2i[(unsigned char)'G'] = 3;
    c2i[(unsigned char)'T'] = 4;
    c2i[(unsigned char)'>'] = 5;

    std::array<int, 216> lut;
    lut.fill(-1);

    for (int i = 0; i < NAIVE_NUM_CODES; ++i) {
        const char* tri = naive_dictionary[i].trigram;
        int code        = naive_dictionary[i].code;

        int x = c2i[(unsigned char)tri[0]];
        int y = c2i[(unsigned char)tri[1]];
        int z = c2i[(unsigned char)tri[2]];

        if (x < 0 || y < 0 || z < 0) continue;

        int idx  = x * 36 + y * 6 + z;
        lut[idx] = code;
    }

    return lut;
}

/**
 * Copy the char2idx and LUT arrays to GPU constant memory.
 * Must be called once before launching naive_tokenizeAndEncodeBatch().
 */
inline void naive_copy_tables_to_device()
{
    auto c2i = build_naive_char2idx();
    auto lut = build_naive_lut();
    cudaMemcpyToSymbol(d_naive_char2idx, c2i.data(), 256 * sizeof(int));
    cudaMemcpyToSymbol(d_naive_LUT,      lut.data(), 216 * sizeof(int));
}

// ─────────────────────────────────────────────────────────────────────────────
// __device__ naive_trigramToNumber()
//
// Direct port of the __device__ trigramToNumber() from inference.cu.
// Uses the __constant__ d_naive_char2idx and d_naive_LUT arrays.
// Returns a token ID in [0..95], or -1 if any character is invalid.
// ─────────────────────────────────────────────────────────────────────────────
__device__ int naive_trigramToNumber(char c1, char c2, char c3)
{
    int x = d_naive_char2idx[static_cast<unsigned char>(c1)];
    int y = d_naive_char2idx[static_cast<unsigned char>(c2)];
    int z = d_naive_char2idx[static_cast<unsigned char>(c3)];

    // If any character is invalid, return -1
    if (x < 0 || y < 0 || z < 0) return -1;

    int idx = x * 36 + y * 6 + z;  // hash function (from inference.cu)
    return d_naive_LUT[idx];        // -1 if invalid, else 0..95
}

// ─────────────────────────────────────────────────────────────────────────────
// __global__ naive_tokenizeAndEncodeBatch()
//
// Direct port of the __global__ tokenizeAndEncodeBatch() from inference.cu.
// Each thread processes one sequence (identified by seq_id).
//
// @param sequences     Flat buffer of all concatenated sequences
//                      (with '<'/'>' sentinels already prepended/appended).
// @param offsets       Byte offsets: offsets[i]..offsets[i+1] spans sequence i.
// @param output        [out] Flat array of int, num_sequences × max_tokens.
// @param num_sequences Number of sequences.
// @param max_tokens    Maximum tokens per sequence (from constants.cuh).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void naive_tokenizeAndEncodeBatch(
    const char* sequences, const int* offsets, int* output,
    int num_sequences, int max_tokens)
{
    int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_id >= num_sequences) return;

    int start = offsets[seq_id];
    int end = offsets[seq_id + 1];  // offsets is n+1 long
    const char* seq = sequences + start;
    int trigram_count = min(end - start - 2, max_tokens);
    int out_idx = seq_id * max_tokens;
    for (int i = 0; i < trigram_count; ++i) {
        output[out_idx + i] = naive_trigramToNumber(seq[i], seq[i + 1], seq[i + 2]);
    }
    // Pad the rest with 0
    for (int i = trigram_count; i < max_tokens; ++i) {
        output[out_idx + i] = 0;
    }
}
