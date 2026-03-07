import subprocess
import re
import statistics
import sys
import os

GPU_BIN = "./build/main"
CPU_BIN = "./build/main_cpu"
NAIVE_BIN = "./build/main_naive"
ITERATIONS = 5
CPU_THREAD_CONFIGS = [1, 32, 64]


def run_benchmark(binary, iterations, extra_args=None):
    """Run `binary` `iterations` times and return mean time per step (μs)."""
    times = {}
    args = [binary, "--benchmarking"] + (extra_args or [])
    for _ in range(iterations):
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                match = re.search(r'(.*?)\s*finished in:\s*([0-9.]+)\s*us', line)
                if match:
                    step_name = match.group(1).strip()
                    time_us = float(match.group(2))
                    times.setdefault(step_name, []).append(time_us)
        except subprocess.CalledProcessError as e:
            print(f"Error running {' '.join(args)}: {e}")
            print(e.stderr)
            sys.exit(1)

    return {step: statistics.mean(vals) for step, vals in times.items()}


def print_table(cpu_results: dict, naive_means: dict, gpu_means: dict):
    """
    cpu_results:  {num_threads: {step: mean_us}}
    naive_means:  {step: mean_us}
    gpu_means:    {step: mean_us}
    """
    col_w_step = 30
    col_w_time = 14

    # Header
    thread_labels = [f"CPU {t}T (us)" for t in CPU_THREAD_CONFIGS]
    headers = ["Step"] + thread_labels + ["Naive (us)", "GPU (us)"]
    col_widths = [col_w_step] + [col_w_time] * (len(CPU_THREAD_CONFIGS) + 2)

    total_width = sum(col_widths) + 3 * (len(headers) - 1)
    sep = "=" * total_width
    thin = "-" * total_width

    title = f"Benchmark Results (Mean of {ITERATIONS} runs)"
    print(sep)
    print(f"{title:^{total_width}}")
    print(sep)

    # Column header row
    row_parts = [f"{h:<{w}}" for h, w in zip(headers, col_widths)]
    print(" | ".join(row_parts))
    print(thin)

    # Gather all step names
    all_steps = list(next(iter(cpu_results.values())).keys()) + list(gpu_means.keys())
    all_steps = list(dict.fromkeys(all_steps))

    for step in all_steps:
        cells = [f"{step:<{col_w_step}}"]
        for t in CPU_THREAD_CONFIGS:
            val = cpu_results[t].get(step)
            cells.append(f"{val:<{col_w_time}.1f}" if val is not None else f"{'N/A':<{col_w_time}}")
        naive_val = naive_means.get(step)
        cells.append(f"{naive_val:<{col_w_time}.1f}" if naive_val is not None else f"{'N/A':<{col_w_time}}")
        gpu_val = gpu_means.get(step)
        cells.append(f"{gpu_val:<{col_w_time}.1f}" if gpu_val is not None else f"{'N/A':<{col_w_time}}")
        print(" | ".join(cells))

    print(thin)

    # Totals row
    cells = [f"{'TOTAL TIME':<{col_w_step}}"]
    cpu_totals = []
    for t in CPU_THREAD_CONFIGS:
        total = sum(cpu_results[t].values())
        cpu_totals.append(total)
        cells.append(f"{total:<{col_w_time}.1f}")
    naive_total = sum(naive_means.values())
    cells.append(f"{naive_total:<{col_w_time}.1f}")
    gpu_total = sum(gpu_means.values())
    cells.append(f"{gpu_total:<{col_w_time}.1f}")
    print(" | ".join(cells))

    print(sep)

    # Speedup summary
    print(f"\nSpeedup vs GPU (total time):")
    for t, cpu_tot in zip(CPU_THREAD_CONFIGS, cpu_totals):
        print(f"  CPU {t:>2}T  / GPU   : {cpu_tot / gpu_total:.2f}x slower")
    print(f"  Naive   / GPU   : {naive_total / gpu_total:.2f}x slower")

    # Thread scaling within CPU
    if len(cpu_totals) > 1:
        base = cpu_totals[0]
        print(f"\nCPU thread scaling (vs 1 thread):")
        for t, tot in zip(CPU_THREAD_CONFIGS[1:], cpu_totals[1:]):
            print(f"  {t:>2} threads : {base / tot:.2f}x speedup")


if __name__ == "__main__":
    if not os.path.exists(GPU_BIN) or not os.path.exists(CPU_BIN) or not os.path.exists(NAIVE_BIN):
        print("Error: Binaries not found. Please run 'make' first.")
        sys.exit(1)

    print(f"Running benchmarks ({ITERATIONS} iterations each)...\n")

    print("Running GPU benchmark...")
    gpu_means = run_benchmark(GPU_BIN, ITERATIONS)

    print("Running Naive benchmark...")
    naive_means = run_benchmark(NAIVE_BIN, ITERATIONS)

    cpu_results = {}
    for t in CPU_THREAD_CONFIGS:
        print(f"Running CPU benchmark ({t} thread(s))...")
        cpu_results[t] = run_benchmark(CPU_BIN, ITERATIONS, extra_args=["--threads", str(t)])

    print("\n")
    print_table(cpu_results, naive_means, gpu_means)
