#!/usr/bin/env python3
import argparse
import random
import os

def generate_sequences(num_seqs, length, output_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    nucleotides = ['A', 'C', 'G', 'T']
    
    print(f"Generating {num_seqs} DNA sequences of length {length} into {output_path}...")
    
    with open(output_path, "w") as f:
        for _ in range(num_seqs):
            # Generate a random sequence and write directly (no gap between rows)
            seq = ''.join(random.choices(nucleotides, k=length))
            f.write(seq + "\n")
            
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Generate random DNA sequences for testing.")
    parser.add_argument("--num-seqs", type=int, required=True, help="Number of DNA sequences to generate.")
    parser.add_argument("--length", type=int, required=True, help="Length of each DNA sequence in base pairs.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output text file.")
    
    args = parser.parse_args()
    
    generate_sequences(args.num_seqs, args.length, args.output)

if __name__ == "__main__":
    main()
