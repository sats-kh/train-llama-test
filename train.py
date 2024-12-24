import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler
import deepspeed

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME = "wikitext"
DATASET_SPLIT = "wikitext-2-raw-v1"

# DeepSpeed configuration
ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,  # ZeRO Stage 2 for optimizer state partitioning
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "contiguous_gradients": True
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10
}

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

    # Load model with DeepSpeed
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )

    # Dataset loading
    dataset = load_dataset(DATASET_NAME, DATASET_SPLIT, split="train", streaming=True)

    def encode(batch):
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_token_type_ids=False
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

    scaler = GradScaler()

    for epoch in range(3):
        model_engine.train()
        for step, batch in enumerate(train_loader):
            inputs = {k: v.cuda(local_rank) for k, v in batch.items()}
            with autocast():
                outputs = model_engine(**inputs)
                loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            if step % 10 == 0 and rank == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()