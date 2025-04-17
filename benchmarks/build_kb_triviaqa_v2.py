# SPDX-License-Identifier: Apache-2.0
"""Build FAISS IVF KB from TriviaQA SearchResults using Contriever and save to disk (Multi-GPU Version)"""

import json
import faiss
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset


def load_triviaqa_build_kb(split, num_prompts):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split=split)

    kb = []
    if num_prompts == -1:
        num_prompts = len(ds)
    else:
        num_prompts = min(num_prompts, len(ds))
    print(f"Loading {num_prompts} questions from TriviaQA {split} split...")

    for item in ds.select(range(num_prompts)):
        search_results = item["search_results"]
        kb.extend(search_results["description"])
    return kb


def encode_contriever(texts, tokenizer, model, device, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model(**tokens).pooler_output
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0).numpy().astype("float32")


def build_and_save_kb_ivf(split, output_prefix, num_prompts, nlist=100):
    docs = load_triviaqa_build_kb(split, num_prompts)
    print(f"Building FAISS IVF Index over {len(docs)} docs...")


    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever").to("cuda")
    model.eval()

    embs = encode_contriever(docs, tokenizer, model, device="cuda")
    dim = embs.shape[1]
    print(f"Embedding dimension: {dim}")

    quantizer = faiss.IndexFlatL2(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    # res = faiss.StandardGpuResources()
    # gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    # Multi-GPU resources
    ngpu = faiss.get_num_gpus()
    print(f"Detected {ngpu} GPUs for FAISS")

    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

    print("Training IVF quantizer...")
    gpu_index.train(embs)

    print("Adding vectors to IVF index...")
    batch_size = 10000
    for i in tqdm(range(0, len(embs), batch_size)):
        gpu_index.add(embs[i:i + batch_size])

    print(f"Total vectors: {gpu_index.ntotal}")

    final_index = faiss.index_gpu_to_cpu(gpu_index)

    with open(f"{output_prefix}_kb.json", "w") as f:
        json.dump(docs, f)
    faiss.write_index(final_index, f"{output_prefix}_kb.index")

    print(f"Saved {output_prefix}_kb.json and {output_prefix}_kb.index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=-1)
    parser.add_argument("--nlist", type=int, default=8192)
    args = parser.parse_args()

    build_and_save_kb_ivf(args.split, args.output_prefix, args.num_prompts, args.nlist)
