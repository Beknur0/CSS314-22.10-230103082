import os
import time
import queue
import platform
import threading



STUDENT_ID = 230103082

TOTAL_ITEMS = 1000
NUM_WORKERS = 4

CHUNK_SIZE = 5


def process_item(i):
    time.sleep(0.0001 * (i % 20))


def static_worker(start, end, worker_id, completion_times):

    for i in range(start, end):
        process_item(i)

    completion_times[worker_id] = time.perf_counter()


def run_static():

    threads = []
    completion_times = [0.0] * NUM_WORKERS

    chunk_size = TOTAL_ITEMS // NUM_WORKERS

    start_time = time.perf_counter()

    for worker_id in range(NUM_WORKERS):

        start = worker_id * chunk_size

        if worker_id == NUM_WORKERS - 1:
            end = TOTAL_ITEMS
        else:
            end = (worker_id + 1) * chunk_size

        thread = threading.Thread(
            target=static_worker,
            args=(
                start,
                end,
                worker_id,
                completion_times
            )
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.perf_counter()

    wall_time_ms = (end_time - start_time) * 1000

    completion_delta_ms = (
        max(completion_times) - min(completion_times)
    ) * 1000

    return wall_time_ms, completion_delta_ms


def dynamic_worker(task_queue, worker_id, completion_times):

    while True:

        try:
            chunk = task_queue.get_nowait()

        except queue.Empty:
            break

        for i in chunk:
            process_item(i)

        task_queue.task_done()

    completion_times[worker_id] = time.perf_counter()


def run_dynamic():

    task_queue = queue.Queue()

    completion_times = [0.0] * NUM_WORKERS

    for start in range(0, TOTAL_ITEMS, CHUNK_SIZE):

        end = min(start + CHUNK_SIZE, TOTAL_ITEMS)

        chunk = range(start, end)

        task_queue.put(chunk)

    threads = []

    start_time = time.perf_counter()

    for worker_id in range(NUM_WORKERS):

        thread = threading.Thread(
            target=dynamic_worker,
            args=(
                task_queue,
                worker_id,
                completion_times
            )
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.perf_counter()

    wall_time_ms = (end_time - start_time) * 1000

    completion_delta_ms = (
        max(completion_times) - min(completion_times)
    ) * 1000

    return wall_time_ms, completion_delta_ms



def main():

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"Logical CPU Cores: {os.cpu_count()}")

    print(f"\nTotal Items: {TOTAL_ITEMS}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Dynamic Chunk Size: {CHUNK_SIZE}")

    print("\nWorkload:")
    print("sleep(0.0001 * (i % 20))")

    static_time, static_delta = run_static()


    dynamic_time, dynamic_delta = run_dynamic()


    speedup = static_time / dynamic_time

    print("\n===== BENCHMARK RESULTS =====")

    print(
        f"{'Strategy':<25}"
        f"{'Wall Time (ms)':<20}"
        f"{'Completion Delta (ms)':<25}"
    )

    print("-" * 70)

    print(
        f"{'Static Scheduling':<25}"
        f"{static_time:<20.3f}"
        f"{static_delta:<25.3f}"
    )

    print(
        f"{'Dynamic Scheduling':<25}"
        f"{dynamic_time:<20.3f}"
        f"{dynamic_delta:<25.3f}"
    )

    print("\n===== PERFORMANCE COMPARISON =====")

    print(f"Dynamic Speedup: {speedup:.3f}x")

    print(
        f"Static Completion Delta: "
        f"{static_delta:.3f} ms"
    )

    print(
        f"Dynamic Completion Delta: "
        f"{dynamic_delta:.3f} ms"
    )

    if dynamic_time < static_time:
        print("Faster Strategy: Dynamic Scheduling")
    else:
        print("Faster Strategy: Static Scheduling")


if __name__ == "__main__":
    main()