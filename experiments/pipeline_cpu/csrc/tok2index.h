#pragma once

#include <tuple>
#include <string>
#include <stdint.h>

// Token ordering matches trigram_code.hpp exactly (uppercase).
// Entry at position i has hashToken(token) == i and index == i.
std::tuple<const char*, uint16_t> _Tok2Index[] = {
    {"<AA",  0}, {"<AC",  1}, {"<AG",  2}, {"<AT",  3},
    {"<CA",  4}, {"<CC",  5}, {"<CG",  6}, {"<CT",  7},
    {"<GA",  8}, {"<GC",  9}, {"<GG", 10}, {"<GT", 11},
    {"<TA", 12}, {"<TC", 13}, {"<TG", 14}, {"<TT", 15},

    {"AA>", 16}, {"AAA", 17}, {"AAC", 18}, {"AAG", 19}, {"AAT", 20},
    {"AC>", 21}, {"ACA", 22}, {"ACC", 23}, {"ACG", 24}, {"ACT", 25},
    {"AG>", 26}, {"AGA", 27}, {"AGC", 28}, {"AGG", 29}, {"AGT", 30},
    {"AT>", 31}, {"ATA", 32}, {"ATC", 33}, {"ATG", 34}, {"ATT", 35},

    {"CA>", 36}, {"CAA", 37}, {"CAC", 38}, {"CAG", 39}, {"CAT", 40},
    {"CC>", 41}, {"CCA", 42}, {"CCC", 43}, {"CCG", 44}, {"CCT", 45},
    {"CG>", 46}, {"CGA", 47}, {"CGC", 48}, {"CGG", 49}, {"CGT", 50},
    {"CT>", 51}, {"CTA", 52}, {"CTC", 53}, {"CTG", 54}, {"CTT", 55},

    {"GA>", 56}, {"GAA", 57}, {"GAC", 58}, {"GAG", 59}, {"GAT", 60},
    {"GC>", 61}, {"GCA", 62}, {"GCC", 63}, {"GCG", 64}, {"GCT", 65},
    {"GG>", 66}, {"GGA", 67}, {"GGC", 68}, {"GGG", 69}, {"GGT", 70},
    {"GT>", 71}, {"GTA", 72}, {"GTC", 73}, {"GTG", 74}, {"GTT", 75},

    {"TA>", 76}, {"TAA", 77}, {"TAC", 78}, {"TAG", 79}, {"TAT", 80},
    {"TC>", 81}, {"TCA", 82}, {"TCC", 83}, {"TCG", 84}, {"TCT", 85},
    {"TG>", 86}, {"TGA", 87}, {"TGC", 88}, {"TGG", 89}, {"TGT", 90},
    {"TT>", 91}, {"TTA", 92}, {"TTC", 93}, {"TTG", 94}, {"TTT", 95}
};