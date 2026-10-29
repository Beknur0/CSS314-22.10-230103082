import numpy as np
import time
import json
import statistics
from pathlib import Path

with open("lab-submission/results/salted_params.json", "r") as f:
    params = json.load(f)

N = params["N"]
M = N // 2

print("N:", N)
print("M:", M)

rng = np.random.default_rng(params["seed"])

A = rng.random((M, M))
B = rng.random((M, M))


def naive_multiply(A, B):
    M = A.shape[0]
    C = np.zeros((M, M), dtype=np.float64)

    for i in range(M):
        for j in range(M):
            total = 0.0

            # Здесь нужен цикл по k.
            # На каждой итерации прибавляй A[i, k] * B[k, j].
            for k in range(M):
                total += A[i, k] * B[k, j]

            C[i, j] = total

    return C


def interchanged_multiply(A, B):
    M = A.shape[0]
    C = np.zeros((M, M), dtype=np.float64)

    for i in range(M):
        for k in range(M):
            a = A[i, k]

            # Здесь нужен цикл по j.
            # Обновляй C[i, j], прибавляя a * B[k, j].
            for j in range(M):
                C[i, j] += a * B[k, j]

    return C


naive_times = []
interchanged_times = []

for run in range(3):
    start = time.perf_counter_ns()

    C1 = naive_multiply(A, B)

    naive_times.append((time.perf_counter_ns() - start) / 1e9)

    start = time.perf_counter_ns()

    C2 = interchanged_multiply(A, B)

    interchanged_times.append((time.perf_counter_ns() - start) / 1e9)

    print(f"Run {run + 1} completed")


naive_median = statistics.median(naive_times)
interchanged_median = statistics.median(interchanged_times)

speedup = naive_median / interchanged_median

print("Naive median:", naive_median, "seconds")
print("Interchanged median:", interchanged_median, "seconds")
print("Speedup:", speedup)
print("Results match:", np.allclose(C1, C2, atol=1e-12))

reference = A @ B

print("Naive correct:", np.allclose(C1, reference, atol=1e-12))
print("Interchanged correct:", np.allclose(C2, reference, atol=1e-12))
