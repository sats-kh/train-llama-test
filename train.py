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
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
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


def wrap_modules_in_policy():
    """Define which modules to wrap with FSDP"""
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    return {LlamaDecoderLayer}


def setup_model_and_tokenizer(local_rank):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    # Mixed precision policy
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float16,
    )

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_cache=False,
    )

    def lambda_auto_wrap_policy(module, recurse, unwrapped_params):
        return True if isinstance(module, tuple(wrap_modules_in_policy())) else False

    # FSDP 설정
    model = FSDP(
        model,
        auto_wrap_policy=lambda_auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        forward_prefetch=True,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False),
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

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
        remove_unused_columns=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        torch_compile=False,  # Disable torch.compile
    )


def main():
    local_rank = setup_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True

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
        # Save the model using FSDP state dict utils
        full_state_dict = FSDP.state_dict(model, StateDictType.FULL_STATE_DICT)
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        torch.save(full_state_dict, os.path.join(OUTPUT_DIR, "pytorch_model.bin"))
        tokenizer.save_pretrained(OUTPUT_DIR)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()