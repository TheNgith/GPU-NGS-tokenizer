/**
 * @file constants.cuh
 * @brief Compile-time constants for the tokenization pipeline.
 *
 * MAX_TOKENS  – Maximum number of tokens (trigrams) extracted per sequence.
 * N_GRAMS    – Size of the n-gram window (3 for trigrams).
 */

#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#define MAX_TOKENS 1008
#define N_GRAMS 3

#define DEFAULT_INPUT_FILE "/mnt/sdc1/NFS-share/species-datasets/ecoli/ecoli_1000/ecoli_1000_quer.txt"

#endif // CONSTANTS_CUH