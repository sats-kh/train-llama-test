#!/bin/bash

# 마스터 노드 설정
MASTER_ADDR="147.47.122.200"
MASTER_PORT=1234
WORLD_SIZE=17

# GPU 수
NUM_GPUS=1

# 실행 루프
for ((GPU_ID=0; GPU_ID<$NUM_GPUS; GPU_ID++)); do
    echo "Launching on MASTER NODE GPU $GPU_ID"

    source /home/kh/train-llama-test/llama-venv/bin/activate
    nohup torchrun \
        --nproc_per_node=1 \
        --nnodes=5 \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        /home/kh/gpuc/train-llama-test/train.py --local_rank=$GPU_ID &
done