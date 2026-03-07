/**
 * @file cudaErr.cuh
 * @brief CUDA error-checking utility macro.
 *
 * Wraps any CUDA API call and aborts with a descriptive message
 * (including file name and line number) if the call returns an error.
 *
 * Usage:
 *   CUDA_CHECK(cudaMalloc(&ptr, size));
 */

#ifndef CUDA_ERR_CUH
#define CUDA_ERR_CUH

#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error " << cudaGetErrorString(err__)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#endif // CUDA_ERR_CUH