#!/bin/bash

# Distributed Training Configuration
MASTER_ADDR="210.125.67.55"
MASTER_PORT=1234
WORLD_SIZE=18
NUM_GPUS=2
NODE_RANK=0 # Default to 0 for master node
NODES=4
# Environment Setup
VENV_PATH="/home/kh/llama/train-llama-test/llama-venv"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

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

# OMP_NUM_THREADS =  num_cores / nproc_per_node
export OMP_NUM_THREADS=24
# Run Torch Distributed Training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /home/kh/llama/train-llama-test/train.py
    #/home/kh/llama/train-llama-test/train.py > train_node_${NODE_RANK}.log 2>&1 &
