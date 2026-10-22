import os
import time
import platform

import numpy as np

from numba import njit, prange, set_num_threads, get_num_threads


STUDENT_ID = 230103082

ARRAY_SIZES = [
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000
]

A = 2.5
B = 1.5

REPEATS = 5


@njit
def serial_operation(x, y, a, b):

    for i in range(len(x)):
        y[i] = a * x[i] + b



@njit(parallel=True)
def parallel_operation(x, y, a, b):

    for i in prange(len(x)):
        y[i] = a * x[i] + b



def benchmark(function, x, y, a, b):

    times = []

    for _ in range(REPEATS):

        start_time = time.perf_counter()

        function(x, y, a, b)

        end_time = time.perf_counter()

        elapsed_ms = (
            end_time - start_time
        ) * 1000

        times.append(elapsed_ms)

    return min(times)



def main():

    logical_cores = os.cpu_count() or 1

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"Logical CPU Cores: {logical_cores}")

    available_threads = get_num_threads()

    num_threads = min(
        logical_cores,
        available_threads
    )

    set_num_threads(num_threads)

    print(f"Numba Threads: {get_num_threads()}")

    print(f"\nOperation: y[i] = {A} * x[i] + {B}")

    print(f"Array Sizes: {ARRAY_SIZES}")
    print(f"Benchmark Repeats: {REPEATS}")

    warmup_x = np.ones(1000, dtype=np.float64)
    warmup_y = np.empty_like(warmup_x)

    serial_operation(warmup_x, warmup_y, A, B)
    parallel_operation(warmup_x, warmup_y, A, B)

    print("\n===== BENCHMARK RESULTS =====")

    print(
        f"{'N':<15}"
        f"{'Serial (ms)':<18}"
        f"{'Parallel (ms)':<18}"
        f"{'Speedup':<12}"
        f"{'Winner':<12}"
    )

    print("-" * 75)

    results = []

    for n in ARRAY_SIZES:

        rng = np.random.default_rng(
            seed=STUDENT_ID
        )

        x = rng.random(
            n,
            dtype=np.float64
        )

        serial_y = np.empty_like(x)
        parallel_y = np.empty_like(x)

        serial_time = benchmark(
            serial_operation,
            x,
            serial_y,
            A,
            B
        )

        parallel_time = benchmark(
            parallel_operation,
            x,
            parallel_y,
            A,
            B
        )

        if not np.allclose(serial_y, parallel_y):
            raise ValueError(
                f"Validation failed for N={n}"
            )

        speedup = serial_time / parallel_time

        if parallel_time < serial_time:
            winner = "Parallel"
        else:
            winner = "Serial"

        results.append({
            "n": n,
            "serial": serial_time,
            "parallel": parallel_time,
            "speedup": speedup,
            "winner": winner
        })

        print(
            f"{n:<15}"
            f"{serial_time:<18.6f}"
            f"{parallel_time:<18.6f}"
            f"{speedup:<12.3f}"
            f"{winner:<12}"
        )


    print("\n===== CROSSOVER ANALYSIS =====")

    crossover_found = False

    for i in range(1, len(results)):

        previous = results[i - 1]
        current = results[i]

        if (
            previous["winner"] == "Serial"
            and current["winner"] == "Parallel"
        ):

            print("Crossover Detected!")

            print(
                f"Parallel execution becomes faster "
                f"between N = {previous['n']:,} "
                f"and N = {current['n']:,}"
            )

            print(
                f"Crossover Range: "
                f"({previous['n']:,}, {current['n']:,}]"
            )

            crossover_found = True

            break

    if not crossover_found:

        print(
            "No Serial-to-Parallel crossover "
            "detected in tested array sizes."
        )

    print("\n===== SUMMARY =====")

    best_result = max(
        results,
        key=lambda r: r["speedup"]
    )

    print(
        f"Maximum Speedup: "
        f"{best_result['speedup']:.3f}x"
    )

    print(
        f"At Array Size: "
        f"{best_result['n']:,}"
    )

    print("\nValidation: PASSED")


if __name__ == "__main__":
    main()