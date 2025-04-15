# SPDX-License-Identifier: Apache-2.0
"""Benchmark Ray Data RAG using FAISS IVF Index with Contriever (CPU-only retrieval)"""

import ray
import ray.data
import json
import faiss
import argparse
import os
import logging
import threading
import time
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor


class ContrieverEncoder:
    def __init__(self, batch_size=64):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.model = AutoModel.from_pretrained("facebook/contriever").to("cpu")
        self.model.eval()
        # logging.info(f"debug: {self.batch_size}")
        self.batch_size = batch_size

    def __call__(self, rows: dict):
        queries = rows["query"].tolist()
        embs = []
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                emb = self.model(**tokens).pooler_output
            embs.append(emb.cpu())
        rows["q_emb"] = torch.cat(embs, dim=0).numpy().astype("float32")
        return rows


class Retriever:
    def __init__(self, docs_path, index_path, topk=5, nprobe=512):
        with open(docs_path) as f:
            self.docs = json.load(f)
        self.docs = [item["content"] for item in self.docs]
        self.index = faiss.read_index(index_path)
        self.index.nprobe = nprobe
        self.k = topk

    def __call__(self, rows: dict):
        q_emb = rows["q_emb"]
        _, I = self.index.search(q_emb, self.k)
        retrieved_docs = [[self.docs[i] for i in neighbors] for neighbors in I]

        return {
            "query": rows["query"].tolist(),
            "retrieved_docs": np.array(retrieved_docs)
        }


def build_prompt(row):
    context = "\n".join(row["retrieved_docs"])
    row["prompt"] = f"Context:\n{context}\n\nQuestion: {row['query']}\nAnswer:"
    return row


def load_triviaqa_prompts(path, num_prompts):
    with open(path) as f:
        data = json.load(f)
    all_data = data["Data"]
    logging.info(f"Loaded {len(all_data)} prompts from {path}")
    random.shuffle(all_data)
    return [item["Question"] for item in all_data[:num_prompts]]


def run_ray_data_rag(requests, model, retrieve_batch_size, docs_path, index_path, topk, nprobe, output_dir):
    ds = ray.data.from_items([{"query": q} for q in requests])

    ds = ds.map_batches(
        ContrieverEncoder,
        fn_constructor_args=[
            64,
        ],
        batch_size=retrieve_batch_size,
        concurrency=2,
        num_cpus=32,
    )

    ds = ds.map_batches(
        Retriever,
        fn_constructor_args=[
            docs_path,
            index_path,
            topk,
            nprobe,
        ],
        batch_size=retrieve_batch_size,
        concurrency=2,
        num_cpus=32,
    )

    ds = ds.map(build_prompt, concurrency=4)

    processor = build_llm_processor(
        vLLMEngineProcessorConfig(
            model_source=model,
            concurrency=1,
            batch_size=64,
            max_pending_requests=10000,
            max_concurrent_batches=8,
            engine_kwargs={
                "enable_chunked_prefill": True,
                # "max_num_seqs": 1024
                # "tensor_parallel_size": 2
                # "max_num_batched_tokens": 4096,
                # "max_model_len": 16384,
            },
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
                max_tokens=64,
            ),
        ),
        postprocess=lambda row: dict(answer=row["generated_text"], **row),
    )

    ds = processor(ds)

    
    throughput_dict = defaultdict(int)

    start = time.perf_counter()
    for row in tqdm(ds.iter_rows(), total=ds.count()):
        now = time.perf_counter()
        elapsed_time = int(now - start)
        throughput_dict[elapsed_time] += 1
    end = time.perf_counter()
    with open(f"{output_dir}/throughput.json", "w") as f:
        json.dump(throughput_dict, f)
        
    ray.timeline(
        f"{output_dir}/timeline.json"
    )
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
    parser.add_argument("--output-dir", type=str, default="/home/yilegu/ray/logs")
    args = parser.parse_args()
    
    # prepare output directory
    # add a time prefix with good format
    time_prefix = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_dir = f"{args.output_dir}/{time_prefix}"
    os.makedirs(output_dir, exist_ok=True)

    # set up logging
    logging.basicConfig(
        filename=f"{output_dir}/log.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    
    ray.init()

    logging.info("Loading KB...")
    docs_path = f"{args.kb_prefix}_kb.json"
    index_path = f"{args.kb_prefix}_kb.index"
    # with open(f"{args.kb_prefix}_kb.json") as f:
    #     kb = json.load(f)
    # docs = [item["content"] for item in kb]

    # index = faiss.read_index(f"{args.kb_prefix}_kb.index")
    # index.nprobe = args.nprobe

    logging.info("Loading Queries...")
    requests = load_triviaqa_prompts(args.dataset, args.num_prompts)

    # logging.info("Creating Contriever Encoder and Retriever...")
    # encoder = ContrieverEncoder(batch_size=64)
    # retriever = Retriever(docs, index, topk=args.topk)

    logging.info("Running RAG Benchmark: Retrieval on CPU, Generation on GPU...")
    elapsed_time = run_ray_data_rag(
        requests,
        args.model,
        args.retrieve_batch_size,
        docs_path,
        index_path,
        args.topk,
        args.nprobe,
        output_dir,
    )

    logging.info(f"Elapsed Time: {elapsed_time:.2f} s")
    logging.info(f"Throughput: {len(requests) / elapsed_time:.2f} queries/s")
