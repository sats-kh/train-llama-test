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
    StateDictType
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from datetime import timedelta
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Llama-3.2-3B"
OUTPUT_DIR = "./llama-3.2-3b-fsdp-finetuned"


def setup_distributed():
    """Initialize the distributed training environment."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger.info(f"Starting process with rank {rank}, local_rank {local_rank}, world_size {world_size}")

    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=10)
        )
    except Exception as e:
        logger.error(f"Failed to initialize process group: {e}")
        raise

    torch.cuda.set_device(local_rank)
    return local_rank


def setup_model_and_tokenizer(local_rank):
    """Set up the model and tokenizer with proper configurations."""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(f"cuda:{local_rank}")

    # Print model's embedding shape for validation
    logger.info(f"Embedding shape: {model.get_input_embeddings().weight.shape}")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # FSDP Wrapping
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer}
    )

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    )

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        sharding_strategy=None if torch.cuda.device_count() == 1 else "FULL_SHARD"
    )

    # Enable activation checkpointing
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
    )
    from functools import partial

    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
    apply_fsdp_wrapper = lambda submodule: wrap(submodule, wrapper_cls=non_reentrant_wrapper)

    model.apply(
        lambda m: apply_fsdp_wrapper(m) if check_fn(m) else None
    )

    return model, tokenizer


def prepare_dataset(tokenizer, max_length=512):
    """Prepare and validate the dataset."""
    logger.info("Loading and preparing dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,  # Important: don't return PyTorch tensors here
            return_attention_mask=True,
            return_token_type_ids=False
        )
        # Add labels for language modeling
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4
    )

    # Validate dataset
    logger.info(f"Dataset size: {len(tokenized_dataset)}")
    logger.info(f"Sample features: {list(tokenized_dataset[0].keys())}")

    return tokenized_dataset


def get_training_arguments(local_rank):
    """Configure training arguments."""
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
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "offload_to_cpu": False,
            "mixed_precision": True,
        },
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True
    )


def main():
    """Main training function."""
    try:
        local_rank = setup_distributed()

        model, tokenizer = setup_model_and_tokenizer(local_rank)
        dataset = prepare_dataset(tokenizer)
        training_args = get_training_arguments(local_rank)

        # Disable caching for training
        model.config.use_cache = False

        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        logger.info("Starting training...")
        trainer.train()

        # Save model and tokenizer (only on rank 0)
        if dist.get_rank() == 0:
            logger.info("Saving model...")
            trainer.save_model()
            tokenizer.save_pretrained(OUTPUT_DIR)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()