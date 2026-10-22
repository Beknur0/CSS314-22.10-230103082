import time
import threading
import platform
import os

import numpy as np

from numba import njit


STUDENT_ID = 230103082

NUM_THREADS = 4
INCREMENTS = 2_000_000

EXPECTED = NUM_THREADS * INCREMENTS



@njit(nogil=True)
def unsafe_increment(counter, iterations):

    for _ in range(iterations):
        counter[0] += 1



def safe_increment(counter, lock, iterations):

    for _ in range(iterations):

        with lock:
            counter[0] += 1



def run_unsafe():

    counter = np.zeros(1, dtype=np.int64)

    threads = []

    start_time = time.perf_counter()

    for _ in range(NUM_THREADS):

        thread = threading.Thread(
            target=unsafe_increment,
            args=(counter, INCREMENTS)
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.perf_counter()

    elapsed_ms = (
        end_time - start_time
    ) * 1000

    return int(counter[0]), elapsed_ms



def run_safe():

    counter = np.zeros(1, dtype=np.int64)

    lock = threading.Lock()

    threads = []

    start_time = time.perf_counter()

    for _ in range(NUM_THREADS):

        thread = threading.Thread(
            target=safe_increment,
            args=(counter, lock, INCREMENTS)
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.perf_counter()

    elapsed_ms = (
        end_time - start_time
    ) * 1000

    return int(counter[0]), elapsed_ms


def main():

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"Logical CPU Cores: {os.cpu_count()}")

    print(f"\nThreads: {NUM_THREADS}")
    print(f"Increments per Thread: {INCREMENTS:,}")

    print(f"Expected Final Value: {EXPECTED:,}")


    warmup_counter = np.zeros(1, dtype=np.int64)

    unsafe_increment(warmup_counter, 1)


    unsafe_result, unsafe_time = run_unsafe()

    lost_updates = EXPECTED - unsafe_result

    lost_percentage = (
        lost_updates / EXPECTED
    ) * 100


    safe_result, safe_time = run_safe()


    penalty_ms = safe_time - unsafe_time

    slowdown = safe_time / unsafe_time


    print("\n===== RACE CONDITION RESULTS =====")

    print(f"Expected Value: {EXPECTED:,}")
    print(f"Actual Unsafe Value: {unsafe_result:,}")

    print(f"Lost Updates: {lost_updates:,}")
    print(f"Lost Updates Percentage: {lost_percentage:.3f}%")

    print("\n===== SYNCHRONIZATION RESULTS =====")

    print(f"Safe Final Value: {safe_result:,}")

    print(f"Unsafe Wall Time: {unsafe_time:.3f} ms")
    print(f"Safe Wall Time: {safe_time:.3f} ms")

    print(f"Lock Penalty: {penalty_ms:.3f} ms")
    print(f"Synchronization Slowdown: {slowdown:.3f}x")


    print("\n===== VALIDATION =====")

    if safe_result == EXPECTED:
        print("Safe Counter Validation: PASSED")
    else:
        print("Safe Counter Validation: FAILED")

    if unsafe_result != EXPECTED:
        print("Race Condition Detected: YES")
    else:
        print("Race Condition Detected: NO")


if __name__ == "__main__":
    main()