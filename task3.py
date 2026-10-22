import os
import time
import platform
import threading

import numpy as np
import psutil

from numba import njit


STUDENT_ID = 230103082

ITERATIONS = 50_000_000

CACHE_LINE_SIZE = 64


@njit(nogil=True)
def unpadded_increment(shared_counts, index, iterations):

    for _ in range(iterations):
        shared_counts[index] += 1


@njit(nogil=True)
def local_increment(shared_counts, index, iterations):

    local_count = 0

    for _ in range(iterations):
        local_count += 1

    shared_counts[index] = local_count


def run_threads(worker_function, shared_counts, num_threads):

    threads = []

    start_time = time.perf_counter()

    for i in range(num_threads):

        thread = threading.Thread(
            target=worker_function,
            args=(shared_counts, i, ITERATIONS)
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000


def main():

    physical_cores = psutil.cpu_count(logical=False)

    if physical_cores is None:
        physical_cores = os.cpu_count() or 1

    logical_cores = os.cpu_count() or 1

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")

    print(f"Physical CPU Cores: {physical_cores}")
    print(f"Logical CPU Cores: {logical_cores}")

    print(f"\nThreads: {physical_cores}")
    print(f"Iterations per Thread: {ITERATIONS:,}")
    print(f"Cache Line Size: {CACHE_LINE_SIZE} bytes")


    shared_counts = np.zeros(
        physical_cores,
        dtype=np.int64
    )


    warmup = np.zeros(physical_cores, dtype=np.int64)

    unpadded_increment(warmup, 0, 1)
    local_increment(warmup, 0, 1)


    unpadded_time = run_threads(
        unpadded_increment,
        shared_counts,
        physical_cores
    )

    unpadded_result = np.sum(shared_counts)


    shared_counts.fill(0)

    local_time = run_threads(
        local_increment,
        shared_counts,
        physical_cores
    )

    local_result = np.sum(shared_counts)

    ratio = unpadded_time / local_time

    expected_result = physical_cores * ITERATIONS

    print("\n===== BENCHMARK RESULTS =====")

    print(
        f"{'Method':<25}"
        f"{'Wall Time (ms)':<20}"
    )

    print("-" * 45)

    print(
        f"{'Unpadded Shared':<25}"
        f"{unpadded_time:<20.3f}"
    )

    print(
        f"{'Padded/Local':<25}"
        f"{local_time:<20.3f}"
    )

    print("\n===== EXECUTION RATIO =====")

    print(f"Unpadded / Local: {ratio:.3f}x")

    print("\n===== VALIDATION =====")

    print(f"Expected Result: {expected_result:,}")
    print(f"Unpadded Result: {unpadded_result:,}")
    print(f"Local Result: {local_result:,}")

    if (
        unpadded_result == expected_result
        and local_result == expected_result
    ):
        print("Validation: PASSED")
    else:
        print("Validation: FAILED")


if __name__ == "__main__":
    main()