import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datasets import load_dataset

# 모델 및 데이터 설정
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DATASET_NAME = "wikitext"
DATASET_SPLIT = "wikitext-2-raw-v1"

def main():
    # DDP 초기화
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://147.47.122.200:1234",  # 마스터 노드 IP 및 포트
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"])
    )

    # 로컬 프로세스 ID
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda(local_rank)

    # DDP 래핑
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 데이터셋 로드 및 전처리
    dataset = load_dataset(DATASET_NAME, DATASET_SPLIT, split="train")

    def encode(batch):
        return tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

    encoded_dataset = dataset.map(encode, batched=True)
    train_loader = DataLoader(
        encoded_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x
    )

    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 학습 루프
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            inputs = {k: torch.stack([b[k].squeeze() for b in batch]).cuda(local_rank) for k in batch[0].keys()}
            outputs = model(**inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if dist.get_rank() == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    # 종료
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
