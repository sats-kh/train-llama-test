import os
import torch
import functools
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
from tqdm import tqdm

MODEL_NAME = "meta-llama/Llama-3.2-3B"
OUTPUT_DIR = "./llama-3.2-3b-finetuned"
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5

def setup_distributed():
    """Initialize distributed training environment."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def prepare_model_and_tokenizer(local_rank):
    """Load tokenizer and model, and wrap model with FSDP."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # Define mixed precision and wrapping policy
    mixed_precision_config = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    )
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={AutoModelForCausalLM}
    )

    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision_config,
        device_id=local_rank
    )
    return model, tokenizer

def prepare_dataset(tokenizer, max_length=512):
    print("Loading and preparing dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"  # 텐서 형식 반환
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4
    )
    print(f"Dataset features: {tokenized_dataset.features}")
    return tokenized_dataset

def train_model(model, dataloader, optimizer, epochs, rank):
    """Train the model using FSDP."""
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in tqdm(dataloader):
            # 디버깅: batch 구조 확인
            print(f"Batch keys: {batch.keys() if isinstance(batch, dict) else type(batch)}")

            # 데이터 GPU로 이동
            for k, v in batch.items():
                if isinstance(v, list) or isinstance(v, torch.Tensor):
                    batch[k] = torch.tensor(v, dtype=torch.long).to(rank)
                elif isinstance(v, torch.Tensor):
                    batch[k] = v.to(rank)
                else:
                    raise TypeError(f"Unsupported data type for batch key {k}: {type(v)}")

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} completed.")

def main():
    local_rank, rank, world_size = setup_distributed()

    model, tokenizer = prepare_model_and_tokenizer(local_rank)
    dataset = prepare_dataset(tokenizer)

    # Prepare distributed sampler and dataloader
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    train_model(model, dataloader, optimizer, EPOCHS, local_rank)

    if rank == 0:
        print("Saving model...")
        model_state = model.state_dict()
        torch.save(model_state, os.path.join(OUTPUT_DIR, "llama-3.2-3b-finetuned.pt"))
        tokenizer.save_pretrained(OUTPUT_DIR)

    destroy_process_group()


if __name__ == "__main__":
    main()