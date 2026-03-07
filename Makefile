# ──────────────────────────────────────────────────────────────────────
# Makefile — Build pipeline for the GPU trigram tokenizer
#
#   make          Build everything (generate LUT, compile CUDA binary)
#   make clean    Remove all generated files
# ──────────────────────────────────────────────────────────────────────

NVCC      ?= nvcc
NVCCFLAGS ?= -Isrc/common -Isrc/cpu -Isrc/gpu_naive -Isrc/gpu_optimized
PYTHON    ?= python3
TARGET        := build/main
TARGET_CPU    := build/main_cpu
TARGET_NAIVE  := build/main_naive
SRC           := src/gpu_optimized/main.cu
SRC_CPU       := src/cpu/main_cpu.cpp
SRC_NAIVE     := src/gpu_naive/main_naive.cu
HEADERS       := src/common/constants.cuh src/common/cudaErr.cuh src/common/utils.cuh src/cpu/tok2index.h src/cpu/preprocess.h src/cpu/progressbar.h src/gpu_naive/naive_preprocess.cuh
LUT       := build/LUT.txt
DATA_DIR  := data
DATASETS  := $(DATA_DIR)/seqs_150.txt $(DATA_DIR)/seqs_400.txt $(DATA_DIR)/seqs_1000.txt

.PHONY: all clean directories data

all: directories $(TARGET) $(TARGET_CPU) $(TARGET_NAIVE) data

data: $(DATASETS)

$(DATA_DIR)/seqs_%.txt: scripts/generate_data.py
	@mkdir -p $(DATA_DIR)
	$(PYTHON) scripts/generate_data.py --num-seqs 1000 --length $* --output $@

directories:
	@mkdir -p build output

# ── Step 1: Generate LUT.txt (architecture-specific hash table) ──────
$(LUT): scripts/generate_lut.py
	$(PYTHON) scripts/generate_lut.py --output $@

# ── Step 2: Compile CUDA binary (depends on LUT + all headers) ───────
$(TARGET): $(SRC) $(HEADERS) $(LUT)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC)

# ── Step 3: Compile CPU binary ───────────────────────────────────────
$(TARGET_CPU): $(SRC_CPU) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Xcompiler -pthread -o $@ $(SRC_CPU)

# ── Step 4: Compile naive GPU binary (CUDA, no libtorch) ─────────────────
$(TARGET_NAIVE): $(SRC_NAIVE) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC_NAIVE)

clean:
	rm -rf build/ output/
