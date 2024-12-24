import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, default_data_collator
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME = "wikitext"
DATASET_SPLIT = "wikitext-2-raw-v1"

def main():
    # Get environment variables for distributed training
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    # Initialize the process group
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

    # Set the local GPU device
    torch.cuda.set_device(local_rank)

    # Load tokenizer and model with mixed precision and device mapping
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.gradient_checkpointing_enable()

    # Load dataset
    dataset = load_dataset(DATASET_NAME, DATASET_SPLIT, split="train")

    # Preprocess dataset
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
        encoded_dataset, batch_size=1, shuffle=True, collate_fn=default_data_collator, pin_memory=True
    )

    # Prepare optimizer and scaler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()

    # Training loop
    for epoch in range(3):
        model.train()
        for step, batch in enumerate(train_loader):
            # Move batch to the appropriate device
            inputs = {k: v.to(local_rank) for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast():
                outputs = model(**inputs)
                loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log progress
            if step % 10 == 0 and dist.get_rank() == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()