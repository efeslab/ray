# SPDX-License-Identifier: Apache-2.0
"""Build FAISS IVF KB from TriviaQA SearchResults using Contriever and save to disk"""

import json
import faiss
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch


def load_triviaqa_build_kb(path, num_prompts):
    with open(path) as f:
        data = json.load(f)
    all_data = data["Data"]

    kb = []
    if num_prompts == -1:
        num_prompts = len(all_data)
    else:
        num_prompts = min(num_prompts, len(all_data))
    print(f"Loading {num_prompts} prompts from {path}...")
    for item in all_data[:num_prompts]:
        search_results = item["SearchResults"]
        for doc in search_results:
            kb.append({
                "title": doc.get("Title", ""),
                "content": doc.get("Description", ""),
            })
    return kb


def encode_contriever(texts, tokenizer, model, device, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model(**tokens).pooler_output  # shape: (batch, 768)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0).numpy().astype("float32")

def encode_e5(texts, tokenizer, model, device, batch_size=64):
    embs = []
    # Add "passage: " prefix for better performance
    texts = ["passage: " + t for t in texts]
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**tokens).last_hidden_state  # shape: (B, L, H)
            emb = output.mean(dim=1)  # mean pooling
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0).numpy().astype("float32")


def build_and_save_kb_ivf(dataset_path, output_prefix, num_prompts, nlist=100):
    kb = load_triviaqa_build_kb(dataset_path, num_prompts)
    print(f"Building FAISS IVF Index over {len(kb)} docs...")

    docs = [item["content"] for item in kb]
    
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
    model = AutoModel.from_pretrained("intfloat/e5-large-v2").to("cuda")
    model.eval()

    embs = encode_e5(docs, tokenizer, model, device="cuda")
    dim = embs.shape[1]
    print(f"Embedding dimension: {dim}")

    # tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    # model = AutoModel.from_pretrained("facebook/contriever").to("cuda")
    # model.eval()

    # embs = encode_contriever(docs, tokenizer, model, device="cuda")
    # dim = embs.shape[1]
    # print(f"Embedding dimension: {dim}")

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    # res = faiss.StandardGpuResources()
    # index = faiss.index_cpu_to_gpu(res, 0, index)
    index = faiss.index_cpu_to_all_gpus(index)

    print("Training IVF quantizer...")
    index.train(embs)

    print("Adding vectors to IVF index...")
    batch_size = 10000
    for i in tqdm(range(0, len(embs), batch_size)):
        index.add(embs[i:i+batch_size])

    index = faiss.index_gpu_to_cpu(index)

    print(f"Total vectors: {index.ntotal}")

    with open(f"{output_prefix}_kb.json", "w") as f:
        json.dump(kb, f)
    faiss.write_index(index, f"{output_prefix}_kb.index")

    print(f"Saved {output_prefix}_kb.json and {output_prefix}_kb.index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=-1)
    parser.add_argument("--nlist", type=int, default=100)
    args = parser.parse_args()

    build_and_save_kb_ivf(args.dataset, args.output_prefix, args.num_prompts, args.nlist)
