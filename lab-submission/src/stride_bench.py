import numpy as np
import time
import statistics
import csv
from pathlib import Path

data = np.ones(16_777_216, dtype=np.float64)
strides = [1, 2, 4, 8, 16, 32, 64, 128]

print("Array size:", data.nbytes / (1024**2), "MiB")

results = []
for stride in strides:
    measurements = []
    elements_touched = len(range(0, len(data), stride))

    for repeat in range(10):
        start = time.perf_counter_ns()

        total = 0.0

        # Здесь добавь цикл чтения массива.
        # Используй range(0, len(data), stride).
        for i in range(0, len(data), stride):
            total += data[i]

        elapsed = time.perf_counter_ns() - start

        # Первые 5 запусков — прогрев.
        # В measurements сохраняй только последние 5.
        if repeat >= 5:
            measurements.append(elapsed)

    # Здесь вычисли медиану measurements.
    # Затем раздели её на elements_touched.
    median_ns = statistics.median(measurements)
    latency_ns = median_ns / elements_touched
    results.append({
        "stride": stride,
        "elements_touched": elements_touched,
        "median_time_ns": median_ns,
        "latency_ns_per_element": latency_ns,
        "checksum": total
    })

    print("Stride:", stride)
    print("Median time (ns):", median_ns)
    print("Latency per element (ns):", latency_ns)
    print("Checksum:", total)
    print("----------------------------")


Path("results").mkdir(exist_ok=True)

with open("results/stride_results.tsv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "stride",
            "elements_touched",
            "median_time_ns",
            "latency_ns_per_element",
            "checksum"
        ],
        delimiter="\t"
    )

    writer.writeheader()
    writer.writerows(results)

print("Saved: results/stride_results.tsv")