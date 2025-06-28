#!/usr/bin/env python3

import os
import ray
import numpy as np
import time
import argparse
import logging
from datetime import datetime
from ray.data import DataContext


class MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')


def run_ray_data(output_dir):
    NUM_NODES = 8
    NUM_ITEMS = 2 * 20_000 * NUM_NODES / 25
    ITEM_SHAPE = 1024 * 1024  # elements
    DTYPE_SIZE = 8  # bytes

    data_context = DataContext.get_current()
    data_context.op_resource_reservation_ratio = 0
    # data_context.execution_options.verbose_progress = True
    data_context.override_object_store_memory_limit_fraction = 1
    # data_context.target_max_block_size = 1024 ** 3  # 1 GB
    ray.init()
    
    # warmup
    for i in range(5):
        ds = ray.data.range_tensor(NUM_ITEMS, shape=(ITEM_SHAPE,))
        ds = ds.flat_map(lambda x: [], num_cpus=0.99)
        for batch in ds.iter_batches():
            continue

    for i in range(5):
        logging.info(f"Start {i}-th benchmark") 
        ds = ray.data.range_tensor(NUM_ITEMS, shape=(ITEM_SHAPE,))
        ds = ds.flat_map(lambda x: [], num_cpus=0.99)

        # logging.info("Start actual benchmark")
        
        start_time = time.perf_counter()
        for batch in ds.iter_batches():
            continue
        end_time = time.perf_counter()

        total_data_size = NUM_ITEMS * ITEM_SHAPE * DTYPE_SIZE / (1024 ** 3)  # GB
        print("[Ray Data]")
        print("Total data size in GB:", total_data_size)
        print("Total time taken in seconds:", end_time - start_time)
        print("Throughput in GB/s:", total_data_size / (end_time - start_time))
    # print(ds.stats())

    ray.timeline(os.path.join(output_dir, "timeline_ray_data_scalability.json"))
    ray.shutdown()


@ray.remote
def ray_original_task():
    # ray.put(np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(np.arange(0, 128), tuple(range(1, 1 + 2))))
    yield np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(np.arange(0, 16), tuple(range(1, 1 + 2)))
    # return np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(np.arange(0, 16), tuple(range(1, 1 + 2)))


def run_ray_original(output_dir):
    NUM_NODES = 8
    NUM_WARMUP_ITEMS = 100 * NUM_NODES
    NUM_ITEMS = 100 * NUM_NODES
    ITEM_SHAPE = 1024 * 1024  # elements
    DTYPE_SIZE = 8  # bytes

    ray.init()

    # Warm up workers
    for i in range(2):
        warmup_tasks = [ray_original_task.remote() for _ in range(NUM_WARMUP_ITEMS)]
        # ray.get(warmup_tasks)
        while warmup_tasks:
            ready_tasks, warmup_tasks = ray.wait(warmup_tasks)
            # for ready_task in ready_tasks:
            #     ray.get(next(ready_task))

    for i in range(2):
        start_time = time.perf_counter()
        tasks = [ray_original_task.remote() for _ in range(NUM_ITEMS)]
        # ray.get(tasks)
        while tasks:
            ready_tasks, tasks = ray.wait(tasks)
            # for ready_task in ready_tasks:
            #     ray.get(next(ready_task))
        end_time = time.perf_counter()

        # Each task handles approx 1 GB tensor
        total_data_size = NUM_ITEMS / 8  # GB
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
    
    # set logging level to INFO and create a log file with the current timestamp, each line in log should have a timestamp
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S.%f'
    # )
    # log_file = os.path.join(args.output_dir, f"benchmark_scalability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    # logging.getLogger().addHandler(logging.FileHandler(log_file))
    
    handler = logging.StreamHandler()
    formatter = MicrosecondFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S.%f'
    )
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    if args.mode == "ray_data":
        run_ray_data(args.output_dir)
    elif args.mode == "ray_original":
        run_ray_original(args.output_dir)
