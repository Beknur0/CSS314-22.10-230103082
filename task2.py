import os
import platform
import time
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory


STUDENT_ID = 230103082

ARRAY_SIZE = 20_000_000

PARALLEL_FRACTION = 0.95


def partial_sum(args):
    shm_name, shape, dtype, start, end = args

    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        data = np.ndarray(
            shape,
            dtype=np.dtype(dtype),
            buffer=shm.buf
        )

        total = 0.0

        for i in range(start, end):
            total += data[i]

        return total

    finally:
        shm.close()


def parallel_sum(shm_name, shape, dtype, workers):

    chunk_size = ARRAY_SIZE // workers

    tasks = []

    for i in range(workers):

        start = i * chunk_size

        if i == workers - 1:
            end = ARRAY_SIZE
        else:
            end = (i + 1) * chunk_size

        tasks.append(
            (shm_name, shape, dtype, start, end)
        )

    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(partial_sum, tasks))

    total_sum = sum(results)

    end_time = time.perf_counter()

    wall_time_ms = (end_time - start_time) * 1000

    return total_sum, wall_time_ms


def amdahl_speedup(workers):

    return 1 / (
        (1 - PARALLEL_FRACTION)
        + PARALLEL_FRACTION / workers
    )


def main():

    max_cores = os.cpu_count() or 1

    worker_counts = [
        1,
        2,
        4,
        max_cores,
        max_cores * 2
    ]

    worker_counts = list(dict.fromkeys(worker_counts))

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"Logical CPU Cores: {max_cores}")

    print(f"\nArray Size: {ARRAY_SIZE:,}")
    print("Data Type: float64")
    print(f"Parallel Fraction: {PARALLEL_FRACTION}")

    rng = np.random.default_rng(seed=STUDENT_ID)

    data = rng.random(
        ARRAY_SIZE,
        dtype=np.float64
    )

    shm = shared_memory.SharedMemory(
        create=True,
        size=data.nbytes
    )

    shared_array = np.ndarray(
        data.shape,
        dtype=data.dtype,
        buffer=shm.buf
    )

    shared_array[:] = data

    del data

    print("\n===== BENCHMARK RESULTS =====")

    print(
        f"{'Cores':<10}"
        f"{'Wall Time (ms)':<20}"
        f"{'Speedup':<15}"
        f"{'Amdahl Upper Bound':<20}"
    )

    print("-" * 65)

    baseline_time = None

    try:
        for workers in worker_counts:

            total_sum, wall_time = parallel_sum(
                shm.name,
                shared_array.shape,
                shared_array.dtype.str,
                workers
            )

            if baseline_time is None:
                baseline_time = wall_time

            speedup = baseline_time / wall_time

            theoretical_speedup = amdahl_speedup(workers)

            print(
                f"{workers:<10}"
                f"{wall_time:<20.3f}"
                f"{speedup:<15.3f}"
                f"{theoretical_speedup:<20.3f}"
            )

        print("\n===== FINAL RESULT =====")
        print(f"Array Sum: {total_sum:.6f}")

    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()