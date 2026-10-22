import os
import time
import platform

import numpy as np

from numba import njit, prange, set_num_threads
from numba import get_num_threads
from numba import config


STUDENT_ID = 230103082

ARRAY_SIZE = 30_000_000

REPEATS = 5



@njit(parallel=True)
def memory_kernel(A, B):

    for i in prange(len(A)):
        B[i] = A[i]



@njit(parallel=True)
def compute_kernel(A, B):

    for i in prange(len(A)):
        B[i] = (
            np.sin(A[i]) * np.cos(A[i])
            + np.tan(A[i])
        )


def benchmark(kernel, A, B, repeats=REPEATS):

    times = []

    for _ in range(repeats):

        start = time.perf_counter()

        kernel(A, B)

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000

        times.append(elapsed_ms)

    return min(times)



def run_kernel_tests(kernel, A, B, core_counts):

    results = []

    baseline_time = None

    for cores in core_counts:

        set_num_threads(cores)

        wall_time = benchmark(kernel, A, B)

        if baseline_time is None:
            baseline_time = wall_time

        speedup = baseline_time / wall_time

        efficiency = speedup / cores

        results.append({
            "cores": cores,
            "time": wall_time,
            "speedup": speedup,
            "efficiency": efficiency
        })

    return results



def print_results(title, results):

    print(f"\n===== {title} =====")

    print(
        f"{'Cores':<10}"
        f"{'Wall Time (ms)':<20}"
        f"{'Speedup':<15}"
        f"{'Efficiency':<15}"
    )

    print("-" * 60)

    for result in results:

        print(
            f"{result['cores']:<10}"
            f"{result['time']:<20.3f}"
            f"{result['speedup']:<15.3f}"
            f"{result['efficiency']:<15.3f}"
        )



def main():

    logical_cores = os.cpu_count() or 1

    max_threads = min(
        logical_cores,
        config.NUMBA_NUM_THREADS
    )

    core_counts = list(dict.fromkeys([
        1,
        2,
        max_threads
    ]))

    core_counts = [
        count for count in core_counts
        if count <= max_threads
    ]

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")

    print(f"Logical CPU Cores: {logical_cores}")
    print(f"Available Numba Threads: {max_threads}")

    print(f"\nArray Size: {ARRAY_SIZE:,}")
    print("Data Type: float64")
    print(f"Benchmark Repeats: {REPEATS}")

    print(f"Thread Counts: {core_counts}")


    rng = np.random.default_rng(
        seed=STUDENT_ID
    )

    A = rng.random(
        ARRAY_SIZE,
        dtype=np.float64
    )

    B = np.empty_like(A)

    print(f"Input Array Size: {A.nbytes / 1e6:.2f} MB")
    print(f"Output Array Size: {B.nbytes / 1e6:.2f} MB")


    warmup_A = np.ones(1000, dtype=np.float64)
    warmup_B = np.empty_like(warmup_A)

    memory_kernel(warmup_A, warmup_B)
    compute_kernel(warmup_A, warmup_B)


    memory_results = run_kernel_tests(
        memory_kernel,
        A,
        B,
        core_counts
    )

    memory_valid = np.array_equal(A, B)


    compute_results = run_kernel_tests(
        compute_kernel,
        A,
        B,
        core_counts
    )

    expected = (
        np.sin(A) * np.cos(A)
        + np.tan(A)
    )

    compute_valid = np.allclose(
        B,
        expected,
        rtol=1e-12,
        atol=1e-12
    )

    del expected


    print_results(
        "KERNEL A: MEMORY-BOUND",
        memory_results
    )

    print_results(
        "KERNEL B: COMPUTE-BOUND",
        compute_results
    )


    memory_max = memory_results[-1]
    compute_max = compute_results[-1]

    print("\n===== SCALING ANALYSIS =====")

    print(
        f"Memory Kernel Maximum Speedup: "
        f"{memory_max['speedup']:.3f}x"
    )

    print(
        f"Memory Kernel Efficiency: "
        f"{memory_max['efficiency']:.3f}"
    )

    print(
        f"\nCompute Kernel Maximum Speedup: "
        f"{compute_max['speedup']:.3f}x"
    )

    print(
        f"Compute Kernel Efficiency: "
        f"{compute_max['efficiency']:.3f}"
    )


    print("\n===== MEMORY BANDWIDTH ANALYSIS =====")

    if memory_max["efficiency"] < compute_max["efficiency"]:

        print(
            "Kernel A (Memory-Bound) shows "
            "lower scaling efficiency."
        )

        print(
            "Conclusion: Kernel A is the likely "
            "candidate for memory bandwidth saturation."
        )

        print(
            "Kernel B scales better because "
            "it performs more CPU computations "
            "per memory access."
        )

    else:

        print(
            "Kernel A does not show lower scaling "
            "efficiency than Kernel B."
        )

        print(
            "Conclusion: Memory bandwidth saturation "
            "was not clearly demonstrated."
        )


    print("\n===== VALIDATION =====")

    print(
        f"Memory Kernel: "
        f"{'PASSED' if memory_valid else 'FAILED'}"
    )

    print(
        f"Compute Kernel: "
        f"{'PASSED' if compute_valid else 'FAILED'}"
    )



if __name__ == "__main__":
    main()