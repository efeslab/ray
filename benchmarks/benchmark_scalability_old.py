import ray
import time

NUM_NODES = 1
NUM_ITEMS = 2 * 20_000 * NUM_NODES
ITEM_SHAPE = 1024 * 1024  # bytes
DTYPE_SIZE = 8  # bytes

data_context = ray.data.DataContext.get_current()
data_context.op_resource_reservation_ratio = 0
# data_context.execution_options.verbose_progress = True
data_context.override_object_store_memory_limit_fraction=1
# data_context.target_max_block_size = 1024 ** 3  # 1 GB
# data_context.target_min_block_size = 1024 ** 3  # 1 GB

ray.init("auto")
ds = ray.data.range_tensor(NUM_ITEMS, shape=(ITEM_SHAPE,))
ds = ds.flat_map(lambda x: [], num_cpus=0.99)

start_time = time.perf_counter()
for batch in ds.iter_batches():
    continue
end_time = time.perf_counter()

# total data size in GB
total_data_size = NUM_ITEMS * ITEM_SHAPE * DTYPE_SIZE / (1024**3)
print("Total data size in GB:", total_data_size)
print("Total time taken in seconds:", end_time - start_time)
print("Throughput in GB/s:", total_data_size / (end_time - start_time))

print(ds.stats())
ray.timeline("/home/yilegu/ray/ray/benchmarks/timeline_ray_data_scalability.json")

ray.shutdown()  # Kill all workers and clear object store