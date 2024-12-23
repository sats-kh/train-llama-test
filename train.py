import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, default_data_collator
from datasets import load_dataset

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DATASET_NAME = "wikitext"
DATASET_SPLIT = "wikitext-2-raw-v1"

def main():
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    try:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://147.47.122.200:1234",
            world_size=world_size,
            rank=rank
        )
    except Exception as e:
        print(f"Failed to initialize process group: {e}")
        exit(1)

    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto").cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

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
        encoded_dataset, batch_size=1, shuffle=True, collate_fn=default_data_collator
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        model.train()
        for step, batch in enumerate(train_loader):
            inputs = {k: torch.stack([b[k] for b in batch]).to(local_rank) for k in batch[0].keys()}
            outputs = model(**inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0 and dist.get_rank() == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()