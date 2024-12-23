#!/bin/bash

# Distributed Training Configuration
MASTER_ADDR="147.47.122.200"
MASTER_PORT=1234
WORLD_SIZE=17
NUM_GPUS=2
NODE_RANK=2 # Default to 1 for slave node

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

# OMP_NUM_THREADS =  num_cores / nproc_per_node
export OMP_NUM_THREADS=24
# Run Torch Distributed Training
nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=5 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /home/kh/llama/train-llama-test/train.py > train_node_${NODE_RANK}.log 2>&1 &

echo "Training process launched. Check train_node_${NODE_RANK}.log for logs."