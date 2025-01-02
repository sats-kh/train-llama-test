import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
    MixedPrecision,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
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
    )

    # Define mixed precision settings
    mixed_precision_config = MixedPrecision(
        param_dtype=torch.float16,  # Parameters in FP16
        reduce_dtype=torch.float16,  # Communication in FP16
        buffer_dtype=torch.float16,  # Buffers in FP16
    )

    # Wrap entire model manually with FSDP
    model = FSDP(
        model,
        device_id=local_rank,
        mixed_precision=mixed_precision_config  # Pass mixed precision settings
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
        remove_columns=dataset.column_names,  # Remove original text column
        num_proc=4
    )
    # Ensure the dataset has input_ids and attention_mask
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
        push_to_hub=False,
        save_total_limit=2,
        report_to="tensorboard",
        ddp_find_unused_parameters=False,  # FSDP requires this to be False
        remove_unused_columns=False  # Do not remove unused columns
    )


def save_model(model, tokenizer):
    print("Saving model...")
    state_dict_config = FullStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_dict_config):
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(OUTPUT_DIR, "fsdp_model.pt"))
    tokenizer.save_pretrained(OUTPUT_DIR)


def main():
    local_rank = setup_distributed()

    model, tokenizer = setup_model_and_tokenizer(local_rank)
    dataset = prepare_dataset(tokenizer)
    training_args = get_training_arguments(local_rank)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()

    if dist.get_rank() == 0:
        save_model(model, tokenizer)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()