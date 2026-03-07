#!/usr/bin/env python3
"""
Generate LUT.txt — a 256-entry lookup table that maps compressed 8-bit
trigram hashes to integer token IDs.

The hash is computed by the CUDA kernel as follows:
  1. Pack three ASCII characters (little-endian) into a uint32.
  2. Extract 8 bits:
       bits[7:5] = char[2] bits [2:0]
       bits[4:3] = char[1] bits [2:1]  (i.e. (char[1] & 0x06) >> 1)
       bits[2:0] = char[0] bits [2:0]

Valid trigrams are drawn from (uppercase):
  - first:  {A, T, C, G, B}   (B = sequence-start sentinel, replaces '<')
  - middle: {A, T, C, G}
  - last:   {A, T, C, G, E}   (E = sequence-end sentinel, replaces '>')

Token IDs follow trigram_code.hpp ordering (see TRIGRAM_TO_INDEX below).
Unused hash slots are filled with -1.

Usage:
    python3 generate_lut.py [--output LUT.txt]
"""

import argparse
import itertools
import sys
from typing import List

# ---------------------------------------------------------------------------
# Canonical trigram → token-ID mapping  (matches trigram_code.hpp)
# ---------------------------------------------------------------------------
# Groups: <XX (0-15), then for each first-letter pair XY:
#   XY> followed by XYA XYC XYG XYT  (groups of 5)
TRIGRAM_TO_INDEX = {
    "<AA":  0, "<AC":  1, "<AG":  2, "<AT":  3,
    "<CA":  4, "<CC":  5, "<CG":  6, "<CT":  7,
    "<GA":  8, "<GC":  9, "<GG": 10, "<GT": 11,
    "<TA": 12, "<TC": 13, "<TG": 14, "<TT": 15,

    "AA>": 16, "AAA": 17, "AAC": 18, "AAG": 19, "AAT": 20,
    "AC>": 21, "ACA": 22, "ACC": 23, "ACG": 24, "ACT": 25,
    "AG>": 26, "AGA": 27, "AGC": 28, "AGG": 29, "AGT": 30,
    "AT>": 31, "ATA": 32, "ATC": 33, "ATG": 34, "ATT": 35,

    "CA>": 36, "CAA": 37, "CAC": 38, "CAG": 39, "CAT": 40,
    "CC>": 41, "CCA": 42, "CCC": 43, "CCG": 44, "CCT": 45,
    "CG>": 46, "CGA": 47, "CGC": 48, "CGG": 49, "CGT": 50,
    "CT>": 51, "CTA": 52, "CTC": 53, "CTG": 54, "CTT": 55,

    "GA>": 56, "GAA": 57, "GAC": 58, "GAG": 59, "GAT": 60,
    "GC>": 61, "GCA": 62, "GCC": 63, "GCG": 64, "GCT": 65,
    "GG>": 66, "GGA": 67, "GGC": 68, "GGG": 69, "GGT": 70,
    "GT>": 71, "GTA": 72, "GTC": 73, "GTG": 74, "GTT": 75,

    "TA>": 76, "TAA": 77, "TAC": 78, "TAG": 79, "TAT": 80,
    "TC>": 81, "TCA": 82, "TCC": 83, "TCG": 84, "TCT": 85,
    "TG>": 86, "TGA": 87, "TGC": 88, "TGG": 89, "TGT": 90,
    "TT>": 91, "TTA": 92, "TTC": 93, "TTG": 94, "TTT": 95,
}


# ---------------------------------------------------------------------------
# Bit-extraction helpers (mirrors the CUDA kernel exactly)
# ---------------------------------------------------------------------------
def pack_chars_to_uint32(c0: str, c1: str, c2: str, pad: str = ".") -> int:
    """Pack 4 chars into a little-endian uint32: c0 at byte 0, c3 at byte 3."""
    return ord(c0) | (ord(c1) << 8) | (ord(c2) << 16) | (ord(pad) << 24)


def compress_token(tok: int) -> int:
    """
    Extract the 8-bit hash the kernel computes:
        bits[7:5] = (tok >> 16) & 0x07          # char[2] low 3 bits
        bits[4:3] = ((tok >> 8) & 0x06) >> 1    # char[1] bits 2-1
        bits[2:0] = tok & 0x07                   # char[0] low 3 bits
    """
    return (
        (((tok >> 16) & 0x07) << 5)
        | ((((tok >> 8) & 0x06) >> 1) << 3)
        | (tok & 0x07)
    )


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------
def generate_lut() -> List[int]:
    """Return a 256-element list of int8 token IDs (-1 for unused slots)."""

    bases = ["A", "T", "C", "G"]
    first_letters = bases + ["B"]      # B = sequence-start sentinel (replaces '<')
    middle_letters = bases
    last_letters = bases + ["E"]       # E = sequence-end sentinel (replaces '>')

    # Enumerate all 100 valid trigrams (5 × 4 × 5)
    trigrams = [
        "".join(t)
        for t in itertools.product(first_letters, middle_letters, last_letters)
    ]

    # Build hash → trigram-name mapping
    hash_to_trigram: dict[int, str] = {}
    for tri in trigrams:
        tok = pack_chars_to_uint32(tri[0], tri[1], tri[2])
        h = compress_token(tok) & 0xFF
        # Convert sentinel letters to their display forms for lookup
        display = tri.replace("B", "<").replace("E", ">")
        hash_to_trigram[h] = display

    # Build the 256-entry LUT
    lut = []
    for i in range(256):
        display = hash_to_trigram.get(i)
        if display is not None:
            token_id = TRIGRAM_TO_INDEX.get(display)
            lut.append(token_id if token_id is not None else -1)
        else:
            lut.append(-1)

    return lut


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the 256-entry trigram hash → token-ID lookup table."
    )
    parser.add_argument(
        "--output", "-o",
        default="LUT.txt",
        help="Output file path (default: LUT.txt)",
    )
    args = parser.parse_args()

    lut = generate_lut()

    with open(args.output, "w") as f:
        for value in lut:
            f.write(f"{value}\n")

    valid = sum(1 for v in lut if v != -1)
    print(
        f"[generate_lut] Wrote {len(lut)} entries ({valid} valid, "
        f"{len(lut) - valid} unused) → {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
