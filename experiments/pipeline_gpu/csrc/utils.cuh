/**
 * @file utils.cuh
 * @brief Host-side utility functions for file I/O and data formatting.
 *
 * Provides helpers used by the main pipeline:
 *   - read_file()                – Load multi-line sequence file into a flat buffer with offsets.
 *   - read_lut_file()            – Parse a 256-entry lookup table from a text file.
 *   - print_binary_with_ascii()  – Debug helper: print a uint32 as binary + ASCII.
 *   - write_int8_flat_to_csv()   – Write tokenized output to CSV (one row per sequence).
 */

#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include "constants.cuh"

/**
 * @brief Read a multi-line text file into a single contiguous string.
 *
 * Each line is appended to @p allText (no newlines are kept).
 * @p offsets stores the byte offset where each line begins/ends,
 * so offsets[i] .. offsets[i+1] spans line i in allText.
 *
 * @param filename  Path to the input file.
 * @param allText   [out] Concatenated text of all lines.
 * @param offsets   [out] Vector of size (num_lines + 1) with byte boundaries.
 * @param verbose   If true, print each line and its offset to stdout.
 * @return true on success, false if the file could not be opened.
 */
bool read_file(
    const std::string &filename,
    std::string &allText,
    std::vector<size_t> &offsets,
    bool verbose = false
) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    std::string line;
    size_t currentOffset = 0;
    int lineIndex = 0;

    offsets.clear();
    offsets.push_back(0);  // start of line 0

    while (std::getline(infile, line)) {
        if (verbose) {
            std::cout << "Line " << lineIndex
                      << " starts at offset " << currentOffset
                      << " | \"" << line << "\"\n";
        }

        // Inject sentinels: 'B' = start boundary ('<'), 'E' = end boundary ('>')
        allText += 'B';            // start sentinel
        allText += line;           // original sequence (uppercase)
        allText += 'E';            // end sentinel
        currentOffset += static_cast<size_t>(line.size()) + 2;  // +2 for sentinels

        offsets.push_back(currentOffset); // offset for next line

        lineIndex++;
    }

    return true;
}

/**
 * @brief Print a 32-bit integer in binary with aligned ASCII characters.
 *
 * Outputs two lines: the first shows all 32 bits grouped by byte,
 * the second shows the printable ASCII character for each byte
 * (or '.' for non-printable values).  Useful for debugging the
 * packed trigram representation.
 *
 * @param u  The 32-bit value to display.
 */
void print_binary_with_ascii(uint32_t u) {

    // ---- Print binary bits ----
    for (int i = 31; i >= 0; --i) {
        std::cout << ((u >> i) & 1);
        if (i % 8 == 0) std::cout << ' ';
    }
    std::cout << "\n ";

    // ---- Print ASCII chars per byte ----
    for (int byte = 3; byte >= 0; --byte) {
        uint8_t b = (u >> (byte * 8)) & 0xFF;

        // printable ASCII? else show '.'
        char c = (b >= 32 && b <= 126) ? static_cast<char>(b) : '.';

        // align under the 8 bits + 1 space = 9 columns
        std::cout << std::setw(8) << c << " ";
    }
    std::cout << "\n";
}

/**
 * @brief Read a lookup table (LUT) from a text file.
 *
 * Expects one integer per line (in the range -128..127).
 * The resulting vector has exactly as many entries as there are
 * non-empty lines in the file (typically 256).
 *
 * @param filename  Path to the LUT text file.
 * @param verbose   If true, print each parsed value.
 * @return A vector of int8_t values read from the file.
 * @throws std::runtime_error on I/O or parse errors.
 */
std::vector<int8_t> read_lut_file(const std::string& filename, bool verbose=false) {
    std::vector<int8_t> values;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    int line_num = 0;

    while (std::getline(file, line)) {
        line_num++;

        // Skip empty lines
        if (line.empty()) continue;

        // Convert string → integer
        std::stringstream ss(line);
        int temp;
        ss >> temp;

        if (ss.fail()) {
            throw std::runtime_error("Invalid integer at line: " + std::to_string(line_num));
        }

        // Check range for int8_t
        if (temp < -128 || temp > 127) {
            throw std::runtime_error(
                "Value out of int8_t range at line " + std::to_string(line_num) +
                ": " + std::to_string(temp)
            );
        }

        values.push_back(static_cast<int8_t>(temp));

        if (verbose) {
            std::cout << "Line " << line_num << ": " << temp << " -> stored as " 
                      << static_cast<int>(values.back()) << "\n";
        }
    }

    return values;
}

/**
 * @brief Write a flat int8_t array to CSV, with MAX_TOKENS values per row.
 *
 * Each row corresponds to one sequence's tokenized output.
 * Values are written as plain integers separated by commas.
 *
 * @param data         Pointer to the flat int8_t array.
 * @param total_elems  Total number of elements in the array.
 * @param filename     Output CSV file path.
 */
void write_int8_flat_to_csv(const int8_t* data, std::size_t total_elems, const std::string& filename) {
    if (!data || total_elems == 0) {
        std::cerr << "Error: null data or zero length\n";
        exit(1);
    }

    if (total_elems % MAX_TOKENS != 0) {
        std::cerr << "Warning: total elements not divisible by MAX_TOKENS. "
                  << "Last row will be partial.\n";
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: cannot open file: " << filename << "\n";
        exit(1);
    }

    std::size_t index = 0;

    while (index < total_elems) {
        std::size_t remaining = total_elems - index;
        std::size_t row_len = std::min<std::size_t>(remaining, MAX_TOKENS);

        for (std::size_t c = 0; c < row_len; ++c) {
            ofs << static_cast<int>(data[index + c]);
            if (c + 1 < row_len) {
                ofs << ",";
            }
        }
        ofs << "\n";

        index += row_len;
    }
}

#endif // UTILS_CUH