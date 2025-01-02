import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_wrap_policy,
    size_based_auto_wrap_policy,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets import load_dataset
from datetime import timedelta

MODEL_NAME = "meta-llama/Llama-3.2-3B"
OUTPUT_DIR = "./llama-3.2-3b-finetuned"


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Starting process with rank {rank}, local_rank {local_rank}, world_size {world_size}")

    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
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
    )

    # FSDP mixed precision policy
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    )

    # FSDP CPU offload policy (optional)
    cpu_offload_policy = CPUOffload(offload_params=True)

    # Define wrapping policy
    wrap_policy = transformer_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # Initialize FSDP wrapped model
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        cpu_offload=cpu_offload_policy,
        sync_module_states=True,  # Ensure params are synced across ranks initially
        forward_prefetch=True,  # Enable prefetching for better performance
        limit_all_gathers=True  # Prevent memory spikes
    )

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
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False
    )


def save_model(model, tokenizer, output_dir):
    """Save the FSDP model using state dict"""
    # Switch to FULL_STATE_DICT for saving
    full_state_dict = model.state_dict()

    if dist.get_rank() == 0:
        print("Saving model...")
        # Create a config to save
        config = model.module.config

        # Save the model state dict
        torch.save(full_state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


def main():
    local_rank = setup_distributed()

    # Set device
    torch.cuda.set_device(local_rank)

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

    # Save model with proper FSDP state dict handling
    save_model(trainer.model, tokenizer, OUTPUT_DIR)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()