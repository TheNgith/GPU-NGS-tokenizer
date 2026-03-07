#pragma once

#include "tok2index.h"
#include <vector>
#include <stdexcept>
#include <thread>
#include <atomic>
#include "progressbar.h"

// Map uppercase DNA bases to 2-bit values.
// A → 0, C → 1, G → 2, T → 3
static inline uint8_t char2Val(const char c)
{
    switch (c)
    {
    case 'A':
        return 0;
    case 'C':
        return 1;
    case 'G':
        return 2;
    case 'T':
        return 3;
    default:
        return 7;
    }
}

// Hash a trigram to its index in _Tok2Index (matching trigram_code.hpp ordering).
//
// Mapping:
//   <XY  →  char2Val(X)*4 + char2Val(Y)                              (0–15)
//   XY>  →  16 + 5*(char2Val(X)*4 + char2Val(Y))                     (16,21,26,...,91)
//   XYZ  →  16 + 5*(char2Val(X)*4 + char2Val(Y)) + 1 + char2Val(Z)   (17–20,22–25,...,92–95)
static inline uint8_t hashToken(const char token0, const char token1, const char token2)
{
    // Handle strings starting with '<'  (start-boundary trigrams)
    if (token0 == '<')
    {
        return (char2Val(token1) << 2) + char2Val(token2);
    }
    // Handle strings ending with '>'  (end-boundary trigrams)
    else if (token2 == '>')
    {
        return 16 + 5 * ((char2Val(token0) << 2) + char2Val(token1));
    }
    // Handle regular trigrams containing only A, C, G, T
    else
    {
        return 16 + 5 * ((char2Val(token0) << 2) + char2Val(token1)) + 1 + char2Val(token2);
    }
}

class Preprocessor
{
private:
    std::vector<const char *> tokens_;
    std::vector<uint16_t> indices_;

public:
    Preprocessor()
    {
        uint8_t i = 0;
        for (auto &[token, index] : _Tok2Index)
        {
            if (hashToken(token[0], token[1], token[2]) != i)
            {
                throw std::runtime_error("Tokens are sorted incorrecly or are not fully defined");
            }
            tokens_.push_back(token);
            indices_.push_back(index);
            i++;
        }
    }

    // extract all substrings of 3 from seq, and return the corresponding index
    inline std::vector<uint16_t> preprocess(const std::string &seq, unsigned maxLen)
    {
        unsigned len = std::min(maxLen, static_cast<unsigned int>(seq.size()));
        std::vector<uint16_t> result(len);

        // First token: start sentinel '<' + first two chars
        char token0 = '<';
        char token1 = seq[0];
        char token2 = seq[1];
        result[0] = this->indices_[hashToken(token0, token1, token2)];

        // Middle tokens: sliding window of 3 chars
        unsigned i = 0;
        for (; i < len - 2; ++i)
        {
            token0 = seq[i];
            token1 = seq[i + 1];
            token2 = seq[i + 2];
            result[i + 1] = this->indices_[hashToken(token0, token1, token2)];
        }

        // Last token: last two chars + end sentinel '>'
        token0 = seq[i++];
        token1 = seq[i++];
        token2 = (i < seq.size()) ? seq[i] : '>';
        result[len - 1] = this->indices_[hashToken(token0, token1, token2)];
        return result;
    }

    std::vector<std::vector<uint16_t>> preprocessBatch(const std::vector<std::string> &seqs, unsigned maxLen)
    {
        std::vector<std::vector<uint16_t>> result(seqs.size());
        // setup progress bar
        indicators::show_console_cursor(false);
        indicators::ProgressBar progressBar{
            indicators::option::BarWidth{80},
            indicators::option::PrefixText{"tokenizing + indexing"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true}};
        for (size_t i = 0; i < seqs.size(); i++)
        {
            result[i] = this->preprocess(seqs[i], maxLen);
            // update progress bar
            if (i % 100000 == 0 || i == seqs.size() - 1)
            {
                float newProgressCompleted = static_cast<float>(i) / seqs.size() * 100;
                progressBar.set_progress(newProgressCompleted);
            }
        }
        // Close Current progress bar
        progressBar.mark_as_completed();
        indicators::show_console_cursor(true);
        return result;
    }

    /// Multi-threaded version of preprocessBatch.
    /// Partitions sequences across `numThreads` threads; each writes to a
    /// disjoint slice of the result vector (no synchronization needed).
    std::vector<std::vector<uint16_t>> preprocessBatchMT(
        const std::vector<std::string> &seqs,
        unsigned maxLen,
        unsigned numThreads)
    {
        const size_t N = seqs.size();
        std::vector<std::vector<uint16_t>> result(N);

        if (numThreads == 0)
            numThreads = 1;
        if (numThreads > N)
            numThreads = static_cast<unsigned>(N);

        // Shared progress counter
        std::atomic<size_t> processed{0};

        // Progress bar (updated from main thread via polling)
        indicators::show_console_cursor(false);
        indicators::ProgressBar progressBar{
            indicators::option::BarWidth{80},
            indicators::option::PrefixText{"tokenizing + indexing (" + std::to_string(numThreads) + " threads)"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true}};

        // Worker lambda — processes seqs[start .. end)
        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                result[i] = this->preprocess(seqs[i], maxLen);
                processed.fetch_add(1, std::memory_order_relaxed);
            }
        };

        // Partition and launch threads
        std::vector<std::thread> threads;
        threads.reserve(numThreads);
        size_t chunkSize = N / numThreads;
        size_t remainder = N % numThreads;
        size_t offset = 0;

        for (unsigned t = 0; t < numThreads; ++t) {
            size_t start = offset;
            size_t end = start + chunkSize + (t < remainder ? 1 : 0);
            threads.emplace_back(worker, start, end);
            offset = end;
        }

        // Poll progress from the main thread
        while (true) {
            size_t done = processed.load(std::memory_order_relaxed);
            float pct = static_cast<float>(done) / N * 100.0f;
            progressBar.set_progress(pct);
            if (done >= N)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        for (auto &th : threads)
            th.join();

        progressBar.mark_as_completed();
        indicators::show_console_cursor(true);
        return result;
    }
};
