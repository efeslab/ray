# SPDX-License-Identifier: Apache-2.0
"""Benchmark RAG using FAISS + Contriever with Ray Data or staged batch baseline"""

import os
import json
import time
import uuid
import faiss
import torch
import argparse
import logging
import asyncio
import random
import numpy as np
import ray
import ray.data

from tqdm import tqdm
from collections import defaultdict
from typing import List
from transformers import AutoTokenizer, AutoModel
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from vllm import AsyncLLMEngine, SamplingParams, inputs
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs

class E5Encoder:
    def __init__(self, batch_size=64):
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
        self.model = AutoModel.from_pretrained("intfloat/e5-large-v2").to("cpu")
        self.model.eval()
        self.batch_size = batch_size

    def __call__(self, rows: dict):
        queries = rows["query"].tolist()
        embs = []
        texts = ["passage: " + t for t in queries]
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch = texts[i:i+self.batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**tokens).last_hidden_state  # shape: (B, L, H)
                emb = output.mean(dim=1)  # mean pooling
            embs.append(emb.cpu())
        rows["q_emb"] = torch.cat(embs, dim=0).numpy().astype("float32")
        return rows

class ContrieverEncoder:
    def __init__(self, batch_size=64):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.model = AutoModel.from_pretrained("facebook/contriever").to("cpu")
        self.model.eval()
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
    def __init__(self, docs_path, index_path, topk=5, nprobe=512, batch_size=64):
        with open(docs_path) as f:
            self.docs = json.load(f)
        self.docs = [item["content"] for item in self.docs]
        self.index = faiss.read_index(index_path)
        self.index.nprobe = nprobe
        self.k = topk
        self.batch_size = batch_size

    def __call__(self, rows: dict):
        q_emb = rows["q_emb"]
        all_I = []

        for i in range(0, len(q_emb), self.batch_size):
            batch_emb = q_emb[i: i + self.batch_size]
            _, I = self.index.search(batch_emb, self.k)
            all_I.append(I)

        I = np.vstack(all_I)  # concat all retrieved indices

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
    # if all data is smaller than num_prompts, copy until we reach num_prompts
    if len(all_data) < num_prompts:
        all_data = all_data * (num_prompts // len(all_data)) + all_data[:num_prompts % len(all_data)]
        
    random.shuffle(all_data)
    return [item["Question"] for item in all_data[:num_prompts]]


def run_ray_data_rag(requests, model, data_parallel_size, retrieve_batch_size, docs_path, index_path, topk, nprobe, output_dir, mode):
    ray.init()
    
    if mode == "ray_data_static":
        if data_parallel_size > 6:
            configuration = {
                "ContrieverEncoder": {
                    "batch_size": retrieve_batch_size,
                    # "concurrency": 2 * data_parallel_size,
                    "concurrency": 2 * 6,
                    "num_cpus": 16 
                },
                "Retriever": {
                    "batch_size": retrieve_batch_size,
                    # "concurrency": 1 * data_parallel_size,
                    "concurrency": 1 * 6,
                    "num_cpus": 8
                },
            }
        else:
            configuration = {
                "ContrieverEncoder": {
                    "batch_size": retrieve_batch_size,
                    "concurrency": 2 * data_parallel_size,
                    "num_cpus": 16 
                },
                "Retriever": {
                    "batch_size": retrieve_batch_size,
                    "concurrency": 1 * data_parallel_size,
                    "num_cpus": 8
                },
            }     
    elif mode == "ray_data_dynamic":
        configuration = {
            "ContrieverEncoder": {
                "batch_size": retrieve_batch_size,
                "concurrency": (2, 8),
                "num_cpus": 16,
            },
            "Retriever": {
                "batch_size": retrieve_batch_size,
                "concurrency": (1, 4),
                "num_cpus": 16,
            },
        }
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    ds = ray.data.from_items([{"query": q} for q in requests])

    ds = ds.map_batches(
        ContrieverEncoder,
        # E5Encoder,
        fn_constructor_args=[64],
        batch_size=configuration["ContrieverEncoder"]["batch_size"],
        concurrency=configuration["ContrieverEncoder"]["concurrency"],
        num_cpus=configuration["ContrieverEncoder"]["num_cpus"],
    )

    ds = ds.map_batches(
        Retriever,
        fn_constructor_args=[docs_path, index_path, topk, nprobe, 64],
        batch_size=configuration["Retriever"]["batch_size"],
        concurrency=configuration["Retriever"]["concurrency"],
        num_cpus=configuration["Retriever"]["num_cpus"],
    )

    ds = ds.map(build_prompt, concurrency=4)

    processor = build_llm_processor(
        vLLMEngineProcessorConfig(
            model_source=model,
            concurrency=data_parallel_size,
            batch_size=256,
            max_pending_requests=10000,
            # max_concurrent_batches=8,
            max_concurrent_batches=1,
            engine_kwargs={
                "enable_chunked_prefill": True,
                "max_num_seqs": 1024,
                "enforce_eager": True,
            }
        ),
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": row["prompt"]},
            ],
            sampling_params=dict(
                n=1, temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=256,
            ),
        ),
        postprocess=lambda row: dict(answer=row["generated_text"], **row),
    )

    ds = processor(ds)

    throughput_dict = defaultdict(int)
    start = time.perf_counter()
    # ds = ds.materialize()
    total = len(requests)
    for row in tqdm(ds.iter_rows(), total=total):
        elapsed_time = int(time.perf_counter() - start)
        throughput_dict[elapsed_time] += 1
    end = time.perf_counter()

    with open(f"{output_dir}/ray_data_throughput.json", "w") as f:
        json.dump(throughput_dict, f)
    ray.timeline(f"{output_dir}/timeline.json")
    return end - start


async def run_staged_batch_baseline_async(requests, model, data_parallel_size, docs_path, index_path, topk, nprobe, output_dir, engine_args):
    # TODO: need to aligh with Ray Data!!!
    
    # restrict_cpu_threads(40 * data_parallel_size)
    
    start = time.perf_counter()
    
    
    logging.info("Encoding queries...")
    start_encoding = time.perf_counter()
    encoder = ContrieverEncoder(batch_size=64)
    # encoder = E5Encoder(batch_size=64)
    all_encoded = encoder({"query": np.array(requests)})
    q_emb = all_encoded["q_emb"]
    end_encoding = time.perf_counter()
    logging.info(f"Encoding time: {end_encoding - start_encoding:.2f} s")

    logging.info("Retrieving documents...")
    start_retrieving = time.perf_counter()
    retriever = Retriever(docs_path, index_path, topk, nprobe, 64)
    retrieved = retriever({"query": np.array(requests), "q_emb": q_emb})

    prompts = []
    for query, docs in zip(retrieved["query"], retrieved["retrieved_docs"]):
        context = "\n".join(docs)
        prompts.append(f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")

    end_retrieving = time.perf_counter()
    logging.info(f"Retrieving time: {end_retrieving - start_retrieving:.2f} s")
    
    
    logging.info("Generating answers with AsyncLLMEngine...")
    start_generation = time.perf_counter()
    #                 "enable_chunked_prefill": True,
                #"max_num_seqs": 1024,
                #"enforce_eager": True,
    engine_args.max_num_seqs = 1024
    engine_args.enable_chunked_prefill = True
    engine_args.enforce_eager = True
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    throughput_dict = defaultdict(int)

    async def generate_single(prompt):
        sampling_param = SamplingParams(
            n=1, temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=256,
        )
        stream = await engine.add_request(
            request_id=str(uuid.uuid4()),
            prompt=inputs.TextPrompt(prompt=prompt),
            params=sampling_param,
        )
        async for output in stream:
            if output.finished:
                elapsed = int(time.perf_counter() - start)
                throughput_dict[elapsed] += 1
                return output
        raise RuntimeError("Request did not finish.")

    # send prompts in batch of 25,000
    for start_idx in range(0, len(prompts), 25000):
        end_idx = min(start_idx + 25000, len(prompts))
        logging.info(f"Sending prompts {start_idx} to {end_idx}...")
        tasks = [asyncio.create_task(generate_single(p)) for p in prompts[start_idx:end_idx]]
        for fut in asyncio.as_completed(tasks):
            await fut
        logging.info(f"Finished sending prompts {start_idx} to {end_idx}...")

    end = time.perf_counter()
    
    logging.info(f"Generation time: {end - start_generation:.2f} s")

    with open(f"{output_dir}/staged_batch_throughput.json", "w") as f:
        json.dump(throughput_dict, f)
    return end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    # parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--kb-prefix", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--nprobe", type=int, default=512)
    parser.add_argument("--retrieve-batch-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="/home/yilegu/ray/logs")
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--mode", type=str, choices=["ray_data_static", "ray_data_dynamic", "staged_batch"], default="ray_data_dynamic")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    time_prefix = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_dir = f"{args.output_dir}/{time_prefix}-{args.mode}-dp{args.data_parallel_size}-nprobe{args.nprobe}-{args.num_prompts}"
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{output_dir}/log.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logging.info("Loading KB...")
    docs_path = f"{args.kb_prefix}_kb.json"
    index_path = f"{args.kb_prefix}_kb.index"

    logging.info("Loading Queries...")
    requests = load_triviaqa_prompts(args.dataset, args.num_prompts)

    if args.mode == "ray_data_static" or args.mode == "ray_data_dynamic":
        logging.info(f"Running RAG Benchmark: {args.mode} mode...")
        elapsed_time = run_ray_data_rag(
            requests,
            args.model,
            args.data_parallel_size,
            args.retrieve_batch_size,
            docs_path,
            index_path,
            args.topk,
            args.nprobe,
            output_dir,
            args.mode
        )
    elif args.mode == "staged_batch":
        logging.info("Running RAG Benchmark: staged_batch async mode...")
        engine_args = AsyncEngineArgs.from_cli_args(args)
        elapsed_time = asyncio.run(run_staged_batch_baseline_async(
            requests,
            args.model,
            args.data_parallel_size,
            docs_path,
            index_path,
            args.topk,
            args.nprobe,
            output_dir,
            engine_args
        ))
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    logging.info(f"Elapsed Time: {elapsed_time:.2f} s")
    logging.info(f"Throughput: {len(requests) / elapsed_time:.2f} queries/s")
