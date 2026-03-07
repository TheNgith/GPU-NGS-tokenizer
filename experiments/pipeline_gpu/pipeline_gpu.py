#!/usr/bin/env python3
"""
GPU Pipeline: tokenize sequences on GPU (CUDA kernel) → run Embedding + BiGRU model on GPU.

Usage:
    python pipeline_gpu.py --input /path/to/sequences.txt --lut /path/to/LUT.txt --output /path/to/output.csv

The CUDA extension is JIT-compiled on first run.
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.cpp_extension import load

# ── JIT-compile the CUDA extension ──────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC_DIR = os.path.join(_SCRIPT_DIR, "csrc")

gpu_tokenizer = load(
    name="gpu_tokenizer",
    sources=[os.path.join(_CSRC_DIR, "gpu_tokenizer.cu")],
    extra_include_paths=[_CSRC_DIR],
    extra_cuda_cflags=["-std=c++17", "-O2"],
    verbose=True,
)

# ── Import model ─────────────────────────────────────────────────────────────
sys.path.insert(0, _SCRIPT_DIR)
from model import TrigramGRU

BATCH_SIZE = 1024
MAX_TOKENS = 1008  # Must match constants.cuh


def save_tensor_to_csv(tensor: torch.Tensor, path: str) -> None:
    """Save a 2D float tensor to CSV."""
    cpu_tensor = tensor.detach().cpu()
    with open(path, "w") as f:
        for i in range(cpu_tensor.size(0)):
            row = cpu_tensor[i]
            f.write(",".join(f"{v.item():.6f}" for v in row))
            f.write("\n")
    print(f"Output saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="GPU tokenization + PyTorch inference pipeline")
    parser.add_argument("--input", required=True, help="Path to input text file (one sequence per line)")
    parser.add_argument("--lut", required=True, help="Path to LUT.txt file (256 entries)")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Inference batch size")
    parser.add_argument("--benchmarking", action="store_true", help="Skip writing the output CSV file")
    args = parser.parse_args()

    device = torch.device("cuda")

    # ── Stage 1: Tokenize on GPU (CUDA kernel) ──────────────────────────
    print(f"[GPU Pipeline] Tokenizing {args.input} on GPU...")
    t0 = time.perf_counter()
    tokens, io_time_us = gpu_tokenizer.tokenize_cuda(args.input, args.lut)
    t1 = time.perf_counter()
    print(f"[GPU Pipeline] Tokenization (incl I/O): {(t1 - t0)*1e6:.0f} us | io_time={io_time_us:.0f} us | shape={tokens.shape} dtype={tokens.dtype} device={tokens.device}")

    # Verify data is on GPU (no D2H transfer happened)
    assert tokens.is_cuda, f"Expected CUDA tensor, got {tokens.device}"

    # ── Padding check ────────────────────────────────────────────────────
    num_seqs = tokens.size(0)
    seq_len = tokens.size(1)
    assert seq_len == MAX_TOKENS, f"Expected seq_len={MAX_TOKENS}, got {seq_len}"
    print(f"[GPU Pipeline] {num_seqs} sequences, padded to {seq_len} tokens")

    # ── Stage 2: Model inference on GPU ──────────────────────────────────
    model = TrigramGRU().to(device)
    model.eval()

    # Batch processing
    outputs = []
    t2 = time.perf_counter()
    with torch.no_grad():
        for start_idx in range(0, num_seqs, args.batch_size):
            end_idx = min(start_idx + args.batch_size, num_seqs)
            batch = tokens[start_idx:end_idx]  # (B, L) on CUDA

            # Pad last batch if needed
            if batch.size(0) < args.batch_size and start_idx + args.batch_size > num_seqs:
                pad_size = args.batch_size - batch.size(0)
                padding = torch.zeros(pad_size, seq_len, dtype=torch.int64, device=device)
                batch = torch.cat([batch, padding], dim=0)
                out = model(batch)
                out = out[:end_idx - start_idx]  # Remove padding
            else:
                out = model(batch)

            outputs.append(out)

    torch.cuda.synchronize()
    t3 = time.perf_counter()

    output_tensor = torch.cat(outputs, dim=0)
    print(f"[GPU Pipeline] Model inference: {(t3 - t2)*1e6:.0f} us | output shape={output_tensor.shape}")
    print(f"[GPU Pipeline] Total time: {(t3 - t0)*1e6:.0f} us")
    e2e_excl_io_us = ((t3 - t0) * 1e6) - io_time_us
    print(f"End-to-end (excl. I/O) finished in: {e2e_excl_io_us:.0f} us")

    # ── Stage 3: Save output ─────────────────────────────────────────────
    if not args.benchmarking:
        save_tensor_to_csv(output_tensor, args.output)
    else:
        print("Benchmarking mode: skipping output write.")


if __name__ == "__main__":
    main()
