import os
import torch
import torch.distributed as dist
import functools
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import ActivationCheckpointing
from datetime import timedelta

MODEL_NAME = "meta-llama/Llama-3.2-3B"
OUTPUT_DIR = "./llama-3.2-3b-fsdp-finetuned"

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Starting process with rank {rank}, local_rank {local_rank}, world_size {world_size}")

    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=10)
        )
    except Exception as e:
        print(f"Failed to initialize process group: {e}")
        exit(1)

    torch.cuda.set_device(local_rank)
    return local_rank

def setup_model_and_tokenizer(local_rank):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(f"cuda:{local_rank}")

    model.gradient_checkpointing_enable()

    # FSDP Wrapping
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={AutoModelForCausalLM})
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,  # Parameters stored in fp16
        reduce_dtype=torch.float16,  # Gradients reduced in fp16
        buffer_dtype=torch.float16  # Buffers stored in fp16
    )

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision_policy,
        cpu_offload=CPUOffload(offload_params=True),
        activation_checkpointing=ActivationCheckpointing(policy="default")  # Enable activation checkpointing
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
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4
    )
    return tokenized_dataset

def get_training_arguments(local_rank):
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=100,
        save_steps=1000,
        fp16=True,
        half_precision_backend="auto",
        push_to_hub=False,
        save_total_limit=2,
        report_to="tensorboard",
        local_rank=local_rank,
        remove_unused_columns=False,  # Fix ValueError for dataset columns
        fsdp="full_shard auto_wrap",  # Enable FSDP
        fsdp_config={
            "offload_to_cpu": True,
            "mixed_precision": True,
            "activation_checkpointing": True,  # Enable activation checkpointing
        },
        ddp_find_unused_parameters=False
    )

def main():
    local_rank = setup_distributed()

    model, tokenizer = setup_model_and_tokenizer(local_rank)
    dataset = prepare_dataset(tokenizer)
    training_args = get_training_arguments(local_rank)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    if dist.get_rank() == 0:
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()