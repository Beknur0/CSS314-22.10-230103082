import os
import platform
import time
import numpy as np

STUDENT_ID = 230103082

ARRAY_SIZE = 20_000_000


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU Model: {platform.processor()}")
    print(f"Logical CPU Cores: {os.cpu_count()}")

    print(f"\nArray Size: {ARRAY_SIZE:,}")
    print("Data Type: float64")
    print("Execution Mode: Single-threaded (Serial)")

    rng = np.random.default_rng(seed=STUDENT_ID)
    data = rng.random(ARRAY_SIZE, dtype=np.float64)

    start_time = time.perf_counter()

    total_sum = 0.0

    for value in data:
        total_sum += value

    mean = total_sum / ARRAY_SIZE

    squared_diff_sum = 0.0

    for value in data:
        squared_diff_sum += (value - mean) ** 2

    std_dev = (squared_diff_sum / ARRAY_SIZE) ** 0.5

    end_time = time.perf_counter()

    execution_time_ms = (end_time - start_time) * 1000

    print("\n===== RESULTS =====")
    print(f"Sum: {total_sum:.6f}")
    print(f"Mean: {mean:.6f}")
    print(f"Standard Deviation: {std_dev:.6f}")
    print(f"Execution Time: {execution_time_ms:.3f} ms")



if __name__ == "__main__":
    main()