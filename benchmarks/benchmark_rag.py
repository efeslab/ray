# SPDX-License-Identifier: Apache-2.0
"""Benchmark Ray Data RAG using FAISS IVF Index with Contriever (CPU-only retrieval)"""

import ray
import ray.data
import json
import faiss
import argparse
import time
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor


def load_triviaqa_prompts(path, num_prompts):
    with open(path) as f:
        data = json.load(f)
    all_data = data["Data"]
    random.shuffle(all_data)
    return [item["Question"] for item in all_data[:num_prompts]]


def encode_contriever(queries, tokenizer, model, batch_size=64):
    embs = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb = model(**tokens).pooler_output  # (batch, 768)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0).numpy().astype("float32")


def retrieve_batch(rows, index, docs, tokenizer, model, k=5):
    # print(f"Debug: {rows}")
    start_time = time.perf_counter()
    queries = rows["query"]  # np.array of strings
    q_emb = encode_contriever(queries.tolist(), tokenizer, model)
    end_time = time.perf_counter()
    # print(f"Encoding time: {end_time - start_time:.2f} s for {len(queries)} queries")
    _, I = index.search(q_emb, k)
    end_time = time.perf_counter()
    # print(f"Search time: {end_time - start_time:.2f} s for {len(queries)} queries")

    retrieved_docs = []
    for neighbors in I:
        retrieved_docs.append([docs[i] for i in neighbors])

    end_time = time.perf_counter()
    # print(f"Total Retrieval time: {end_time - start_time:.2f} s for {len(queries)} queries")
    return {
        "query": queries,
        "retrieved_docs": np.array(retrieved_docs)
    }
    

def build_prompt(row):
    context = "\n".join(row["retrieved_docs"])
    row["prompt"] = f"Context:\n{context}\n\nQuestion: {row['query']}\nAnswer:"
    return row


def run_ray_data_rag(requests, model, index, docs, tokenizer, contriever_model, retrieve_batch_size, topk):
    # ds = ray.data.range(len(requests))
    # ds = ds.map(lambda x: {"query": requests[x["id"]]}, concurrency=4)
    ds = ray.data.from_items([{"query": q} for q in requests])

    ds = ds.map_batches(
        lambda rows: retrieve_batch(rows, index, docs, tokenizer, contriever_model, k=topk),
        batch_size=retrieve_batch_size,
        # concurrency=4,
        # num_cpus=32
    )

    # print("Sample after retrieval:")
    # print(ds.take(1))

    ds = ds.map(build_prompt, concurrency=4)

    processor = build_llm_processor(
        vLLMEngineProcessorConfig(
            model_source=model, 
            concurrency=1, 
            batch_size=256,
            max_pending_requests=10000,
            max_concurrent_batches=8
        ),
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": row["prompt"]},
            ],
            sampling_params=dict(
                n=1,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=256,
            ),
        ),
        postprocess=lambda row: dict(answer=row["generated_text"], **row),
    )

    ds = processor(ds)

    start = time.perf_counter()
    _ = ds.take_all()
    end = time.perf_counter()

    return end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--kb-prefix", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--nprobe", type=int, default=512)
    parser.add_argument("--retrieve-batch-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    ray.init()

    print("Loading KB...")
    with open(f"{args.kb_prefix}_kb.json") as f:
        kb = json.load(f)
    docs = [item["content"] for item in kb]

    index = faiss.read_index(f"{args.kb_prefix}_kb.index")
    index.nprobe = args.nprobe  # IVF Search

    print("Loading Contriever Model (CPU)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever").to("cpu")
    model.eval()

    print("Loading Queries...")
    requests = load_triviaqa_prompts(args.dataset, args.num_prompts)

    print("Running RAG Benchmark: Retrieval on CPU, Generation on GPU...")
    elapsed_time = run_ray_data_rag(
        requests,
        args.model,
        index,
        docs,
        tokenizer,
        model,
        args.retrieve_batch_size,
        args.topk
    )

    print(f"Elapsed Time: {elapsed_time:.2f} s")
    print(f"Throughput: {len(requests) / elapsed_time:.2f} queries/s")
