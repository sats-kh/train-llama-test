import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datasets import load_dataset

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DATASET_NAME = "wikitext"
DATASET_SPLIT = "wikitext-2-raw-v1"

def main():
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://147.47.122.200:1234",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"])
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    dataset = load_dataset(DATASET_NAME, DATASET_SPLIT, split="train")

    def encode(batch):
        return tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

    encoded_dataset = dataset.map(encode, batched=True)
    train_loader = torch.utils.data.DataLoader(
        encoded_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        model.train()
        for batch in train_loader:
            inputs = {k: torch.stack([b[k].squeez@e() for b in batch]).cuda(local_rank) for k in batch[0].keys()}
            outputs = model(**inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if dist.get_rank() == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()