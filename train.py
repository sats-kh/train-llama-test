import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, BitsAndBytesConfig
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME = "wikitext"
DATASET_SPLIT = "wikitext-2-raw-v1"


def main():
    # Distributed training setup
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://210.125.67.55:1234",
        world_size=world_size,
        rank=rank
    )

    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load dataset
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

    dataset = dataset.map(encode, batched=True, remove_columns=["text"])
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: {k: torch.cat([item[k] for item in x]) for k in x[0].keys()},
        pin_memory=True
    )

    # Optimizer and scaler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()

    for epoch in range(3):
        model.train()
        for step, batch in enumerate(train_loader):
            # Move inputs to the correct device
            inputs = {k: v.to(local_rank, non_blocking=True) for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast():
                outputs = model(**inputs)
                loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % 10 == 0 and rank == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()