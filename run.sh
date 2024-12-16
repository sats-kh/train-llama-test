#!/bin/bash

# 마스터 노드 설정
MASTER_ADDR="147.47.122.200"
MASTER_PORT=1234
WORLD_SIZE=17

# 머신 별 GPU 수 설정
declare -A MACHINES=(
    ["147.47.122.200:3298"]=1
    ["147.47.122.200:9944"]=2
    ["147.47.122.200:9933"]=2
    ["gpu1.bigdata.re.kr:9922"]=4
    ["gpu2.bigdata.re.kr:9922"]=8
)

# 현재 글로벌 랭크 트래킹 변수
CURRENT_RANK=0

# 모든 머신에서 학습 실행
for MACHINE in "${!MACHINES[@]}"; do
    NUM_GPUS=${MACHINES[$MACHINE]}
    IFS=":" read -r MACHINE_IP MACHINE_PORT <<< "$MACHINE"

    for ((GPU_ID=0; GPU_ID<$NUM_GPUS; GPU_ID++)); do
        echo "Launching on $MACHINE_IP:$MACHINE_PORT GPU $GPU_ID with rank $CURRENT_RANK"

        ssh -p $MACHINE_PORT kh@$MACHINE_IP "nohup python -m torch.distributed.launch \
            --nproc_per_node=1 \
            --nnodes=5 \
            --node_rank=$((CURRENT_RANK / NUM_GPUS)) \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            /home/kh/gpuc/train-llama-test/train.py --local_rank=$GPU_ID &" &

        # 글로벌 랭크 증가
        ((CURRENT_RANK++))
    done
done

wait
