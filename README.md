# GPU-Accelerated Tokenization for DNA Sequences with Memory-Latency Optimization

Machine learning (ML) is emerging in the Next-Generation Sequencing (NGS) field, demanding scalable tokenization tool for DNA sequences.

Major tokenizers are CPU-only, while tokenization is a massively parallel task, making it well suited for GPU execution. However, naïve GPU implementations often suffer from memory inefficiencies.

Our work aims to design a memory-centric GPU tokenizer that: eliminates shared-memory bank conflicts, maximizes coalesced memory access and memory bandwidth via minimal bit length representations.

This repository contains the source code of the [optimized kernel](src/gpu_optimized/main.cu) along with benchmarking instructions.

> [!NOTE]
> **Proudly presented at the 2026 Florida Undergraduate Research Conference (FURC)**  
> March 2026 · Jacksonville, FL  
> 📄 [View poster](./FURC_poster.pdf)

## Project Structure

```text
├── src/
│   ├── common/         # Shared utilities, constants, and CUDA macros
│   ├── cpu/            # CPU-based C++ tokenization implementation
│   ├── gpu_naive/      # Naive GPU tokenization implementation
│   └── gpu_optimized/  # Optimized GPU tokenization (LUT/Shared Memory)
├── experiments/
│   ├── pipeline_cpu/       # End-to-end CPU pipeline (Tokenization + Inference)
│   ├── pipeline_gpu/       # End-to-end GPU pipeline (Tokenization + Inference)
│   ├── pipeline_cpu_gpu/   # Mixed pipeline (CPU Tokenization + GPU Inference)
│   └── pipeline_naive_gpu/ # Naive End-to-end GPU pipeline
├── scripts/            # LUT generation scripts
├── Makefile            # Build configuration
└── README.md           # This file
```

## Test Data Generation

The build process automatically generates three 1,000-sequence test datasets of lengths 150, 400, and 1000 base pairs under the `data/` directory.

To generate your own custom dataset, you can use the provided data generation script:

```bash
python3 scripts/generate_data.py --num-seqs 1000 --length 150 --output custom_data.txt
```

> [!WARNING]
> The current pipeline is strictly capable of processing sequences up to **1024 base pairs** in length. Attempting to use sequences longer than this will result in undefined behavior, truncation, or memory errors depending on the constant configurations (`MAX_TOKENS`).

## Setup & Build

**Requirements:** NVIDIA GPU, CUDA toolkit (`nvcc`), Python 3, C++17 or later, optionally PyTorch for end-to-end pipelines.

First, build the core tokenization binaries (this auto-generates the necessary LUT and compiles all three tokenizers):

```bash
# Build all tokenizers (Optimized GPU, CPU, Naive GPU)
make

# Clean build artifacts
make clean
```

The compiled binaries will be placed in the `build/` directory.

## Running Benchmarks (Tokenization Only)

You can run the standalone tokenization binaries in benchmarking mode, which skips writing the output CSV to purely measure execution and I/O time.

### 1. Optimized GPU Tokenizer

Uses shared memory and bit-extraction with a constant-memory LUT for maximum performance.

```bash
./build/main --benchmarking
```

### 2. CPU Tokenizer

Multithreaded CPU-based tokenization.

```bash
./build/main_cpu --threads 128 --benchmarking
```

### 3. Naive GPU Tokenizer

GPU tokenization without shared memory optimizations or bit-extraction.

```bash
./build/main_naive --benchmarking
```

## Running End-to-End Pipeline Experiments

The `experiments/` directory contains end-to-end pipelines that integrate tokenization with PyTorch-based model inference (e.g., a BiGRU model). These pipelines JIT-compile their respective PyTorch C++ extensions on the first run.

To run these experiments, activate your Python/Conda environment ensuring `torch` is installed.

### CPU Tokenization + CPU Inference
```bash
python experiments/pipeline_cpu/pipeline_cpu.py \
    --input data/seqs_150.txt \
    --output pipeline_cpu.csv \
    --threads 16 --benchmarking
```

### CPU Tokenization + GPU Inference
```bash
python experiments/pipeline_cpu_gpu/pipeline_cpu_gpu.py \
    --input data/seqs_150.txt \
    --output pipeline_cpu_gpu.csv \
    --threads 16 --benchmarking
```

### Optimized GPU Tokenization + GPU Inference
```bash
python experiments/pipeline_gpu/pipeline_gpu.py \
    --input data/seqs_150.txt \
    --output pipeline_gpu.csv \
    --benchmarking
```

### Naive GPU Tokenization + GPU Inference
```bash
python experiments/pipeline_naive_gpu/pipeline_naive_gpu.py \
    --input data/seqs_150.txt \
    --output pipeline_naive_gpu.csv \
    --benchmarking
```
> [!TIP]
> Omit `--benchmarking` if you want to write the output tensors to a CSV.

### Hardware specifications (for development and testing)
- CPU: AMD EPYC 7H12 64-Core @ 2.60GHz/3.30GHZ
- GPU: NVIDIA H100 
- CUDA version: 13.0
- PyTorch version: 2.5.1