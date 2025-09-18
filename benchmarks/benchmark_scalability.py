#!/usr/bin/env python3

import os
import ray
import numpy as np
import time
import argparse
import logging
from datetime import datetime
from ray.data import DataContext

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

def _create_pg(m: int, cpus_per_bundle: int):
    # Reserve bundles across m distinct nodes.
    base = {"CPU": float(cpus_per_bundle)}
    bundles = [base.copy() for _ in range(m)]
    pg = placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    return pg



class MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S.%f')


def run_ray_data(output_dir, num_nodes, cpus_per_node, size):
    # 128 * size GB per node
    NUM_ITEMS = 128 * size * num_nodes
    ITEM_SHAPE = 1024 * 1024  # elements
    DTYPE_SIZE = 8  # bytes

    data_context = DataContext.get_current()
    data_context.op_resource_reservation_ratio = 0
    # data_context.execution_options.verbose_progress = True
    # data_context.override_object_store_memory_limit_fraction = 1
    data_context.target_max_block_size = 128 * 1024 ** 2  # 128 MB
    # data_context.target_max_block_size = 512 * 1024 ** 2  # 512 MB
    # data_context.target_max_block_size = 1024 ** 3  # 1 GB
    ray.init("auto")
    
    # # Reserve m nodes via PG
    # pg = _create_pg(num_nodes, cpus_per_node)
    # sched = PlacementGroupSchedulingStrategy(pg)

    # # Helper to run a no-op map inside the PG so execution stays within the PG
    # def _in_pg(ds):
    #     return ds.map_batches(
    #         lambda b: b,
    #         num_cpus=cpus_per_node,
    #         scheduling_strategy=sched,
    #     )
        
    # warmup
    for i in range(5):
        ds = ray.data.range_tensor(NUM_ITEMS, shape=(ITEM_SHAPE,))
        ds = ds.flat_map(lambda x: [], num_cpus=0.99)
        # ds = _in_pg(ds)
        ds.materialize()
        # for batch in ds.iter_batches():
        #     continue


    total_time = 0
    profile_time = 10
    for i in range(profile_time):
        logging.info(f"Start {i}-th benchmark") 
        ds = ray.data.range_tensor(NUM_ITEMS, shape=(ITEM_SHAPE,))
        # ds = _in_pg(ds)
        ds = ds.flat_map(lambda x: [], num_cpus=0.99)
        start_time = time.perf_counter()
        ds.materialize()
        end_time = time.perf_counter()
        total_time += end_time - start_time
        # print("Total time taken in seconds:", end_time - start_time)

        # logging.info("Start actual benchmark")
        
        # start_time = time.perf_counter()
        # for batch in ds.iter_batches():
        #     continue
        # end_time = time.perf_counter()

    total_data_size = NUM_ITEMS * ITEM_SHAPE * DTYPE_SIZE / (1024 ** 3)  # GB
    
    avg_time = total_time / profile_time
    print("[Ray Data]")
    print("Total data size in GB:", total_data_size)
    print("Total time taken in seconds:", avg_time)
    print("Throughput in GB/s:", total_data_size / avg_time)
    # print(ds.stats())

    ray.timeline(os.path.join(output_dir, "timeline_ray_data_scalability.json"))
    ray.shutdown()


@ray.remote
def ray_original_task():
    # 128MB per task
    # ray.put(np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(np.arange(0, 128), tuple(range(1, 1 + 2))))
    # yield np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(np.arange(0, 16), tuple(range(1, 1 + 2)))
    # return ray.put(np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(np.arange(0, 16), tuple(range(1, 1 + 2))))
    return np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(np.arange(0, 16), tuple(range(1, 1 + 2)))
    # return np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(np.arange(0, 64), tuple(range(1, 1 + 2)))
    # return np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(np.arange(0, 128), tuple(range(1, 1 + 2)))

                                                                                                
@ray.remote
def g_empty(_x):
    # Empty consumer
    return None

@ray.remote
def ray_original_task_streaming(num_partitions: int = 1):
    # Streaming generator that yields one partiton size
    # yield np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(
    #     np.arange(0, 128, dtype=np.int64), (1, 2)
    # )
    for _ in range(num_partitions):
        yield np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(np.arange(0, 16), tuple(range(1, 1 + 2)))

# @ray.remote
# def ray_original_task_streaming():
#     # Streaming generator that yields one partiton size
#     # yield np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(
#     #     np.arange(0, 128, dtype=np.int64), (1, 2)
#     # )
#     yield np.ones((1024, 1024), dtype=np.int64) * np.expand_dims(np.arange(0, 16), tuple(range(1, 1 + 2)))

# @ray.remote
# def streaming_orchestrator(num_items: int, max_inflight: int = 256):
#     # Launch up to max_inflight tasks, then keep a sliding window full.
#     pending = [ray_original_task.remote()
#                for _ in range(min(num_items, max_inflight))]
#     launched = len(pending)

#     while pending:
#         ready, pending = ray.wait(pending, num_returns=1, fetch_local=False)
#         # Yield the finished partition (ObjectRef to avoid copying)
#         yield ready[0]
#         if launched < num_items:
#             pending.append(ray_original_task.remote())
#             launched += 1


def run_ray_original(output_dir, num_nodes, cpus_per_node, size, mode):
    # 8 x size GB per node
    NUM_WARMUP_ITEMS = 8 * size * num_nodes
    NUM_ITEMS = 8 * size * num_nodes
    # NUM_WARMUP_ITEMS = 2 * size * num_nodes
    # NUM_ITEMS = 2 * size * num_nodes
    # NUM_WARMUP_ITEMS = 1 * size * num_nodes
    # NUM_ITEMS = 1 * size * num_nodes
    # ITEM_SHAPE = 1024 * 1024  # elements
    DTYPE_SIZE = 8  # bytes

    ray.init("auto")

    # # Reserve m nodes via PG
    # pg = _create_pg(num_nodes, cpus_per_node)
    # sched = PlacementGroupSchedulingStrategy(pg)

    # # Helper to run a no-op map inside the PG so execution stays within the PG
    # def _in_pg(ds):
    #     return ds.map_batches(
    #         lambda b: b,
    #         # Make sure the op actually consumes CPU so the bundle is used:
    #         ray_remote_args={
    #             "num_cpus": cpus_per_node,
    #             "scheduling_strategy": sched,
    #         },
    #     )

    # Warm up workers
    for i in range(5):
        if mode == "ray_original":
            f_refs = [ray_original_task.remote() for _ in range(NUM_WARMUP_ITEMS)]
            warmup_tasks = [g_empty.remote(f_ref) for f_ref in f_refs]
            # ray.get(warmup_tasks)
            while warmup_tasks:
                ready_tasks, warmup_tasks = ray.wait(warmup_tasks, num_returns=1, fetch_local=False)
        elif mode == "ray_original_streaming":
            MAX_INFLIGHT = num_nodes * cpus_per_node * 2  # good default
            f_refs = []
            for j in range(NUM_WARMUP_ITEMS):
                gen = ray_original_task_streaming.remote(num_partitions=1)
                f_refs.append(gen)
            inflight = []
            for g in f_refs:                 # iterate EACH generator you created
                for ref in g:                # each yields exactly 1 partition in your code
                    inflight.append(g_empty.remote(ref))
                    if len(inflight) >= MAX_INFLIGHT:
                        _, inflight = ray.wait(inflight, num_returns=1, fetch_local=False)

            # drain
            while inflight:
                _, inflight = ray.wait(inflight, num_returns=1, fetch_local=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        # warmup_tasks = [g_empty.remote(f_ref) for f_ref in f_refs]
        # # ray.get(warmup_tasks)
        # while warmup_tasks:
        #     ready_tasks, warmup_tasks = ray.wait(warmup_tasks, num_returns=1, fetch_local=False)
            # for ready_task in ready_tasks:
            #     ray.get(ready_task)
            # del ready_tasks

    total_time = 0
    profile_time = 10
    for i in range(profile_time):
        logging.info(f"Start {i}-th benchmark")
        start_time = time.perf_counter()
        if mode == "ray_original":
            f_refs = [ray_original_task.remote() for _ in range(NUM_ITEMS)]
            tasks = [g_empty.remote(f_ref) for f_ref in f_refs]
            del f_refs
            # ray.get(tasks)
            while tasks:
                ready_tasks, tasks = ray.wait(tasks, num_returns=1, fetch_local=False)
        elif mode == "ray_original_streaming":
            MAX_INFLIGHT = num_nodes * cpus_per_node * 2  # good default
            f_refs = []
            for j in range(NUM_ITEMS):
                gen = ray_original_task_streaming.remote(num_partitions=1)
                f_refs.append(gen)
            inflight = []
            for g in f_refs:                 # iterate EACH generator you created
                for ref in g:                # each yields exactly 1 partition in your code
                    inflight.append(g_empty.remote(ref))
                    if len(inflight) >= MAX_INFLIGHT:
                        _, inflight = ray.wait(inflight, num_returns=1, fetch_local=False)

            # drain
            while inflight:
                _, inflight = ray.wait(inflight, num_returns=1, fetch_local=False)
            # f_refs.append(gen)
            # TODO: change to ray.wait on the f_refs
        else:
            raise ValueError(f"Unknown mode: {mode}")
        # tasks = [g_empty.remote(f_ref) for f_ref in f_refs]
        # del f_refs
        # # ray.get(tasks)
        # while tasks:
        #     ready_tasks, tasks = ray.wait(tasks, num_returns=1, fetch_local=False)
            # for ready_task in ready_tasks:
            #     ray.get(ready_task)
            # del ready_tasks
        end_time = time.perf_counter()
        total_time += end_time - start_time
        # Each task handles approx 1 GB tensor
    total_data_size = NUM_ITEMS / 8  # GB
    # total_data_size = NUM_ITEMS / 2
    # total_data_size = NUM_ITEMS
        
    avg_time = total_time / profile_time
    print("[Ray Original]" if mode == "ray_original" else "[Ray Original Streaming]")
    print("Total data size in GB:", total_data_size)
    print("Total time taken in seconds:", avg_time)
    print("Throughput in GB/s:", total_data_size / avg_time)

    ray.timeline(os.path.join(output_dir, "timeline_ray_original_scalability.json"))
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Ray Data vs Ray Original")
    parser.add_argument("--mode", choices=["ray_data", "ray_original", "ray_original_streaming"], required=True, help="Which benchmark to run")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use")
    parser.add_argument("--cpus_per_node", type=int, default=8, help="CPU per node")
    parser.add_argument("--size", type=int, default=100, help="Size of the dataset in GB")
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
        run_ray_data(args.output_dir, args.num_nodes, args.cpus_per_node, args.size)
    elif args.mode == "ray_original" or args.mode == "ray_original_streaming":
        run_ray_original(args.output_dir, args.num_nodes, args.cpus_per_node, args.size, args.mode)
