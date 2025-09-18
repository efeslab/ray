#!/bin/bash

NUM_NODES=$1


source ~/miniconda3/etc/profile.d/conda.sh
conda activate raydata

for i in {1..5}
do
    echo "===== Run $i $(date) =====" >> ray_data.log
    python3 /home/ubuntu/ray-data-yilegu/benchmarks/benchmark_scalability.py \
        --mode ray_data --num_nodes "${NUM_NODES}" --size 5 >> ray_data.log 2>&1

    # echo "===== Run $i $(date) =====" >> ray_original.log
    # python3 /home/ubuntu/ray-data-yilegu/benchmarks/benchmark_scalability.py \
    #     --mode ray_original --num_nodes "${NUM_NODES}" --size 5 >> ray_original.log 2>&1

    # echo "===== Run $i $(date) =====" >> ray_original_streaming.log
    # python3 /home/ubuntu/ray-data-yilegu/benchmarks/benchmark_scalability.py \
    #     --mode ray_original_streaming --num_nodes "${NUM_NODES}" --size 5 >> ray_original_streaming.log 2>&1
done
# python3 /home/ubuntu/ray-data-yilegu/benchmarks/benchmark_scalability.py --mode ray_original_streaming --num_nodes ${NUM_NODES} --size 10 > ray_original_streaming.log 2>&1