import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, default_data_collator
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME = "wikitext"
DATASET_SPLIT = "wikitext-2-raw-v1"


def main():
    # These are set by torchrun automatically
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    # Print debug information
    print(f"Starting process with rank {rank}, local_rank {local_rank}, world_size {world_size}")

    try:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://210.125.67.55:1234",
            world_size=world_size,
            rank=rank
        )
    except Exception as e:
        print(f"Failed to initialize process group: {e}")
        exit(1)

    torch.cuda.set_device(local_rank)

    # Modified model loading - remove device_map="auto" since we're using DDP
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="./offload",  # 오프로드 데이터를 저장할 디렉터리
        offload_state_dict=True,
    ).to(f"cuda:{local_rank}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Enable gradient checkpointing before DDP wrapping
    model.gradient_checkpointing_enable()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Rest of your code remains the same...
    dataset = load_dataset(DATASET_NAME, DATASET_SPLIT, split="train")

    def encode(batch):
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        return {key: val for key, val in inputs.items()}

    encoded_dataset = dataset.map(encode, batched=True, remove_columns=["text"])
    train_loader = torch.utils.data.DataLoader(
        encoded_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=default_data_collator,
        pin_memory=True
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()

    for epoch in range(3):
        model.train()
        for step, batch in enumerate(train_loader):
            inputs = {k: torch.stack([b[k] for b in batch]).to(local_rank) for k in batch[0].keys()}
            with autocast():
                outputs = model(**inputs)
                loss = outputs.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % 10 == 0 and dist.get_rank() == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()