#!/bin/bash

# Distributed Training Configuration
MASTER_ADDR="210.125.67.55"  # 실제 마스터 노드 IP로 변경
MASTER_PORT=1234
WORLD_SIZE=18  # 총 GPU 수
NUM_GPUS=4     # 노드당 GPU 수
NODE_RANK=$1   # 실행 시 인자로 받음
NODES=3        # 총 노드 수

# Environment Setup
VENV_PATH="/home/kh/llama/train-llama-test/llama-venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH. Exiting."
    exit 1
fi

source "$VENV_PATH/bin/activate"

# Verify Master Node Connectivity
if ! ping -c 1 $MASTER_ADDR &>/dev/null; then
    echo "Cannot reach master node at $MASTER_ADDR. Exiting."
    exit 1
fi

echo "Launching training on node_rank=$NODE_RANK with $NUM_GPUS GPUs"
echo "Master Address: $MASTER_ADDR, Master Port: $MASTER_PORT, World Size: $WORLD_SIZE"

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=10

# Run Torch Distributed Training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /home/kh/llama/train-llama-test/train2.py 2>&1 | tee train_node_${NODE_RANK}.log