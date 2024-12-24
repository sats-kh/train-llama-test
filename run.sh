#!/bin/bash

# Distributed Training Configuration
MASTER_ADDR="210.125.67.55"  # 실제 마스터 노드 IP로 변경
MASTER_PORT=1234
WORLD_SIZE=10  # 총 GPU 수
NODE_RANK=$1   # 실행 시 인자로 받음
NUM_GPUS=$2     # 노드당 GPU 수
NODES=3        # 총 노드 수

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1
#export NCCL_SOCKET_IFNAME=eno1  # 실제 네트워크 인터페이스 이름으로 변경
export NCCL_P2P_DISABLE=1       # P2P 연결 비활성화
export NCCL_SHM_DISABLE=1
export NCCL_IP_VERSION=4

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