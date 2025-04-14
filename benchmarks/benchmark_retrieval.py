import argparse
import json
import time
import random
import numpy as np
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch


def load_triviaqa_prompts(path, num_prompts):
    with open(path) as f:
        data = json.load(f)
    all_data = data["Data"]
    random.shuffle(all_data)
    return [item["Question"] for item in all_data[:num_prompts]]


def encode_contriever(queries, tokenizer, model, device, batch_size=64):
    embs = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model(**tokens).pooler_output  # (batch, 768)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0).numpy().astype("float32")


def retrieve_batch(queries, index, tokenizer, model, device, k):
    q_emb = encode_contriever(queries, tokenizer, model, device)
    _, I = index.search(q_emb, k)
    return I


def benchmark_retrieval(requests, index, tokenizer, model, device, batch_size, k):
    total_time = 0.0
    num_batches = 0

    for i in tqdm(range(0, len(requests), batch_size)):
        batch = requests[i:i+batch_size]
        start = time.perf_counter()
        _ = retrieve_batch(batch, index, tokenizer, model, device, k)
        end = time.perf_counter()

        total_time += (end - start)
        num_batches += 1

    avg_time = total_time / num_batches
    return avg_time, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--kb-prefix", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=5000)
    parser.add_argument("--retrieve-batch-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--nprobe", type=int, default=64)
    args = parser.parse_args()

    print("Loading KB...")
    with open(f"{args.kb_prefix}_kb.json") as f:
        kb = json.load(f)
    index = faiss.read_index(f"{args.kb_prefix}_kb.index")
    index.nprobe = args.nprobe

    print("Loading Contriever model (CPU only)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever").to("cpu")
    model.eval()
    device = "cpu"

    print("Loading Queries...")
    requests = load_triviaqa_prompts(args.dataset, args.num_prompts)

    print("Benchmarking Retrieval with Contriever (CPU)...")
    avg_time, total_time = benchmark_retrieval(
        requests, index, tokenizer, model, device, args.retrieve_batch_size, args.topk
    )

    print(f"Avg Retrieval Time per Batch: {avg_time:.4f} s")
    print(f"Total Retrieval Time: {total_time:.2f} s")
    print(f"Throughput: {len(requests) / total_time:.2f} queries/s")
