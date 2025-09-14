#!/bin/bash
set -euo pipefail

NUM_NODES=${1:-1}
HEAD_NODE="ray-data-head"
HEAD_IP="10.42.1.167"
RAY_PORT=6379
OBJ_STORE_MEM=21474836480

# Start head node
echo "[HEAD] Starting head node..."
ssh ubuntu@$HEAD_NODE <<EOF
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate raydata
  ray stop --force || true
  ray start --head --port=$RAY_PORT --dashboard-host=0.0.0.0 --num-cpus=0
EOF

# Start worker nodes
for i in $(seq 1 $NUM_NODES); do
  WORKER_NODE="ray-data-worker-$i"
  echo "[WORKER-$i] Starting..."
  ssh ubuntu@$WORKER_NODE <<EOF
    sudo rm -rf /tmp/ray
    sudo rm -rf /dev/shm/*
    sleep 5
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate raydata
    ray start --address=$HEAD_IP:$RAY_PORT --object-store-memory=$OBJ_STORE_MEM
EOF
done
