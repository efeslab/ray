#!/usr/bin/env python3

import os
import ray
import numpy as np
import time
import argparse
from ray.data import DataContext


def run_ray_data(output_dir):
    NUM_NODES = 1
    NUM_ITEMS = 2 * 20_000 * NUM_NODES
    ITEM_SHAPE = 1024 * 1024  # elements
    DTYPE_SIZE = 8  # bytes

    data_context = DataContext.get_current()
    data_context.op_resource_reservation_ratio = 0
    # data_context.execution_options.verbose_progress = True
    data_context.override_object_store_memory_limit_fraction = 1
    # data_context.target_max_block_size = 1024 ** 3  # 1 GB
    ray.init("auto")

    ds = ray.data.range_tensor(NUM_ITEMS, shape=(ITEM_SHAPE,))
    ds = ds.flat_map(lambda x: [], num_cpus=0.99)

    start_time = time.perf_counter()
    for batch in ds.iter_batches():
        continue
    end_time = time.perf_counter()

    total_data_size = NUM_ITEMS * ITEM_SHAPE * DTYPE_SIZE / (1024 ** 3)  # GB
    print("[Ray Data]")
    print("Total data size in GB:", total_data_size)
    print("Total time taken in seconds:", end_time - start_time)
    print("Throughput in GB/s:", total_data_size / (end_time - start_time))
    print(ds.stats())

    ray.timeline(os.path.join(output_dir, "timeline_ray_data_scalability.json"))
    ray.shutdown()


@ray.remote
def ray_original_task():
    ray.put(np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(np.arange(0, 128), tuple(range(1, 1 + 2))))


def run_ray_original(output_dir):
    NUM_ITEMS = 16
    ITEM_SHAPE = 1024 * 1024  # elements
    DTYPE_SIZE = 8  # bytes

    ray.init("auto")

    # Warm up workers
    warmup_tasks = [ray_original_task.remote() for _ in range(300)]
    ray.get(warmup_tasks)

    start_time = time.perf_counter()
    tasks = [ray_original_task.remote() for _ in range(455)]
    ray.get(tasks)
    end_time = time.perf_counter()

    # Each task handles approx 1 GB tensor
    total_data_size = 455  # GB
    print("[Ray Original]")
    print("Total data size in GB:", total_data_size)
    print("Total time taken in seconds:", end_time - start_time)
    print("Throughput in GB/s:", total_data_size / (end_time - start_time))

    ray.timeline(os.path.join(output_dir, "timeline_ray_original_scalability.json"))
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Ray Data vs Ray Original")
    parser.add_argument("--mode", choices=["ray_data", "ray_original"], required=True, help="Which benchmark to run")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    if args.mode == "ray_data":
        run_ray_data(args.output_dir)
    elif args.mode == "ray_original":
        run_ray_original(args.output_dir)
