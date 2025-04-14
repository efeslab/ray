# SPDX-License-Identifier: Apache-2.0
"""Benchmark offline inference throughput with multiple backends (vLLM, HF, MII, ray_data)."""

import argparse
import datetime
import dataclasses
import json
import random
import os
import time
from functools import cache
from typing import Dict, List, Optional, Tuple

import torch
import uvloop
from PIL import Image
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.inputs import TextPrompt
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.sampling_params import BeamSearchParams
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_lora_tokenizer
from vllm.utils import FlexibleArgumentParser, merge_async_iterators

# -----------------------------------------------------------------------------
# Dataclasses & caching
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class SampleRequest:
    """A class representing a single inference request for benchmarking.

    Attributes:
        prompt: The input text prompt for the model.
        prompt_len: The length of the prompt in tokens.
        expected_output_len: The expected length of the output in tokens.
        multi_modal_data: Optional dictionary containing multi-modal data (e.g. images).
        lora_request: Optional LoRARequest specifying the LoRA to use.
    """
    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[MultiModalDataDict] = None
    lora_request: Optional[LoRARequest] = None


@cache
def lora_path_on_disk(lora_path: str) -> str:
    return get_adapter_absolute_path(lora_path)


lora_tokenizer_cache: Dict[int, AnyTokenizer] = {}

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def get_random_lora_request(
        args: argparse.Namespace
) -> Tuple[LoRARequest, Optional[AnyTokenizer]]:
    global lora_tokenizer_cache
    lora_id = random.randint(1, args.max_loras)
    lora_request = LoRARequest(
        lora_name=str(lora_id),
        lora_int_id=lora_id,
        lora_path=lora_path_on_disk(args.lora_path),
    )
    if lora_id not in lora_tokenizer_cache:
        lora_tokenizer_cache[lora_id] = get_lora_tokenizer(lora_request)
    return lora_request, lora_tokenizer_cache[lora_id]


def _get_prompt_for_image_model(question: str, *, model: str) -> str:
    """Prepend and append special tokens around the question to form a prompt for image models."""
    model = model.lower()
    if "pixtral" in model:
        return f"<s>[INST]{question}\n[IMG][/INST]"
    raise ValueError(f"Unsupported model {model}")


def sample_requests(tokenizer: PreTrainedTokenizerBase,
                    args: argparse.Namespace) -> List[SampleRequest]:
    dataset_path: str = args.dataset
    num_requests: int = args.num_prompts
    fixed_output_len: Optional[int] = args.output_len
    model: str = args.model
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]

    # Shuffle the dataset.
    random.shuffle(dataset)

    filtered_dataset: List[SampleRequest] = []
    for data in tqdm(dataset, desc="sampling requests"):
        if len(filtered_dataset) == num_requests:
            break

        # Only keep the first two turns of each conversation.
        prompt = data["conversations"][0]["value"]
        completion = data["conversations"][1]["value"]

        multi_modal_data: Optional[MultiModalDataDict] = None
        if "image" in data:
            multi_modal_data = multi_modal_data or {}
            image_path = data["image"]
            # For demonstration, we assume single-image input
            try:
                multi_modal_data["image"] = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                # Ignore datapoint if the asset is missing
                continue
            prompt = _get_prompt_for_image_model(question=prompt, model=model)

        request_tokenizer = tokenizer
        lora_request: Optional[LoRARequest] = None
        if args.enable_lora:
            lora_request, lora_tokenizer = get_random_lora_request(args)
            if lora_tokenizer:
                request_tokenizer = lora_tokenizer

        # Tokenize the prompts and completions.
        prompt_token_ids = request_tokenizer(prompt).input_ids
        completion_token_ids = request_tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len

        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue

        filtered_dataset.append(
            SampleRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                expected_output_len=output_len,
                multi_modal_data=multi_modal_data,
                lora_request=lora_request,
            )
        )

    return filtered_dataset

# -----------------------------------------------------------------------------
# Benchmark backends
# -----------------------------------------------------------------------------

def run_vllm(
    requests: List[SampleRequest],
    n: int,
    engine_args: EngineArgs,
) -> float:
    from vllm import LLM, SamplingParams
    start = time.perf_counter()
    print(f"run_vllm timer started: {start}")
    
    llm = LLM(**dataclasses.asdict(engine_args))

    # Add the requests to the engine.
    prompts: List[TextPrompt] = []
    sampling_params: List[SamplingParams] = []
    for request in requests:
        prompts.append(
            TextPrompt(
                prompt=request.prompt,
                multi_modal_data=request.multi_modal_data,
            )
        )
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
            )
        )
    lora_requests: Optional[List[LoRARequest]] = None
    if engine_args.enable_lora:
        lora_requests = [request.lora_request for request in requests]

    # For demonstration, we're ignoring the beam_search path here
    start_generate = time.perf_counter()
    llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_requests,
        use_tqdm=False,
    )
    end_generate = time.perf_counter()
    print(f"[VLLM] llm.generate Time taken: {end_generate - start_generate:.2f} seconds")
    return end_generate - start


async def run_vllm_async(
    requests: List[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> float:
    from vllm import SamplingParams
    
    start = time.perf_counter()
    print(f"run_vllm_async timer started: {start}")

    async with build_async_engine_client_from_engine_args(
        engine_args,
        disable_frontend_multiprocessing,
    ) as llm:

        prompts: List[TextPrompt] = []
        sampling_params: List[SamplingParams] = []
        lora_requests: List[Optional[LoRARequest]] = []

        for request in requests:
            prompts.append(
                TextPrompt(
                    prompt=request.prompt,
                    multi_modal_data=request.multi_modal_data,
                )
            )
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=request.expected_output_len,
                )
            )
            lora_requests.append(request.lora_request)

        generators = []
        start_generate = time.perf_counter()
        for i, (prompt, sp, lr) in enumerate(zip(prompts, sampling_params, lora_requests)):
            generator = llm.generate(prompt, sp, lora_request=lr, request_id=f"test{i}")
            generators.append(generator)

        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass
        end = time.perf_counter()

        print(f"[VLLM] llm.generate async Time taken: {end - start_generate:.2f} seconds")
        return end - start

async def run_vllm_async_ray_style(
    requests: List[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> float:
    
    from vllm import SamplingParams, AsyncLLMEngine, inputs
    import asyncio
    import uuid
    
    start = time.perf_counter()
    print(f"run_vllm_async_ray_style timer started: {start}")

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate_single_request(request: SampleRequest):
        if request.multi_modal_data:
            prompt = inputs.TextPrompt(
                prompt=request.prompt,
                multi_modal_data=request.multi_modal_data,
            )
        else:
            prompt = inputs.TextPrompt(prompt=request.prompt)

        sampling_param = SamplingParams(
            n=n,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=request.expected_output_len,
        )

        stream = await engine.add_request(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            params=sampling_param,
            lora_request=request.lora_request,
        )

        async for output in stream:
            if output.finished:
                output.prompt = request.prompt  # Restore original prompt
                print(f"[VLLM] {output.request_id} finished")
                print(f"[VLLM] {output.request_id} output: {output.outputs[0].text}")
                return output
        raise RuntimeError("Request did not finish.")

    start_generate = time.perf_counter()

    tasks = [asyncio.create_task(generate_single_request(request)) for request in requests]

    # Wait for completion as results arrive (like Ray Data's UDF)
    for fut in asyncio.as_completed(tasks):
        output = await fut  # In real benchmark, maybe store/analyze output

    end = time.perf_counter()
    print(f"[VLLM] llm.generate async ray style Time taken: {end - start_generate:.2f} seconds")
    return end - start

def run_hf(
    requests: List[SampleRequest],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    max_batch_size: int,
    trust_remote_code: bool,
) -> float:
    """Run Hugging Face model inference for a set of requests."""
    llm = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code
    )
    if llm.config.model_type == "llama":
        # LLaMA models often need a pad token = eos token for batch padding
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()

    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0

    for i in range(len(requests)):
        prompt = requests[i].prompt
        prompt_len = requests[i].prompt_len
        output_len = requests[i].expected_output_len

        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)

        # Decide if we should flush the batch to generation
        if (len(batch) < max_batch_size) and (i != len(requests) - 1):
            next_prompt_len = requests[i + 1].prompt_len
            next_output_len = requests[i + 1].expected_output_len
            # Heuristic: if the next request still fits the 2048 limit, keep going
            if max(max_prompt_len, next_prompt_len) + max(max_output_len, next_output_len) <= 2048:
                continue

        # Generate
        input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids.cuda()
        llm_outputs = llm.generate(
            input_ids=input_ids,
            do_sample=True,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Force decoding so that we measure the overhead
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)

        pbar.update(len(batch))
        # Clear
        batch = []
        max_prompt_len = 0
        max_output_len = 0

    end = time.perf_counter()
    return end - start


def run_mii(
    requests: List[SampleRequest],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
) -> float:
    from mii import client, serve
    llm = serve(model, tensor_parallel=tensor_parallel_size)
    prompts = [request.prompt for request in requests]

    start = time.perf_counter()
    llm.generate(prompts, max_new_tokens=output_len)
    end = time.perf_counter()

    c = client(model)
    c.terminate_server()
    return end - start

# -----------------------------------------------------------------------------
# New: Ray Data backend
# -----------------------------------------------------------------------------

def run_ray_data(
    requests: List[SampleRequest],
    model: str,
    concurrency: int,
    batch_size: int,
) -> float:
    """Use Ray Dataset + LLMProcessor (vLLMEngineProcessorConfig) to run inference."""

    import ray
    from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

    # Convert the list of prompts into Ray Dataset rows
    # Each row must be a dict to feed into the processor's preprocess function
    # For example, row["item"] is the user prompt
    # prompts = [r.prompt for r in requests]
    # ds = ray.data.from_items(prompts).map(lambda prompt: {"item": prompt})
    

    # Suppose `requests` is a list of SampleRequest objects
    # We map each request to a dict for Ray Dataset
    print(os.environ.get("RAY_PROFILING"))
    print(os.environ.get("RAY_task_events_report_interval_ms"))

    # ray.init(address="auto")
    
    requests_data = []
    for r in requests:
        requests_data.append({
            "prompt": r.prompt,
            "expected_output_len": r.expected_output_len,
            "multi_modal_data": r.multi_modal_data,  # optional
        })

    ds = ray.data.from_items(requests_data)

    # Build the config for the vLLM engine
    config = vLLMEngineProcessorConfig(
        model_source=model,
        engine_kwargs={
            "enable_chunked_prefill": True,
            # "tensor_parallel_size": 2
            # "max_num_batched_tokens": 4096,
            # "max_model_len": 16384,
        },
        concurrency=concurrency,  
        batch_size=batch_size,    
        max_pending_requests=10000,
        max_concurrent_batches=8
    )

    # Build the processor: specify how to convert each dataset row into messages
    # and how to postprocess the results
    processor = build_llm_processor(
        config,
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
                max_tokens=row["expected_output_len"],
            ),
        ),
        postprocess=lambda row: dict(
            # The "generated_text" field is automatically added by the processor
            answer=row["generated_text"],
            **row,  # pass along the original columns
        ),
    )

    # Benchmark
    ds = processor(ds)      # transform the dataset
    start = time.perf_counter()
    _ = ds.take_all()        # force materialization of all results
    end = time.perf_counter()

    ray.timeline("ray_timeline.json")  # Optional: save the timeline for analysis
    return end - start

def run_ray_data2(
    requests: List[SampleRequest],
    model: str,
    concurrency: int,
    batch_size: int,
) -> float:
    """Use Ray Dataset + LLMProcessor (vLLMEngineProcessorConfig) to run inference."""

    import ray
    from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

    # Convert the list of prompts into Ray Dataset rows
    # Each row must be a dict to feed into the processor's preprocess function
    # For example, row["item"] is the user prompt
    # prompts = [r.prompt for r in requests]
    # ds = ray.data.from_items(prompts).map(lambda prompt: {"item": prompt})
    

    # Suppose `requests` is a list of SampleRequest objects
    # We map each request to a dict for Ray Dataset
    requests_data = []
    for r in requests:
        requests_data.append({
            "prompt": r.prompt,
            "expected_output_len": r.expected_output_len,
            "multi_modal_data": r.multi_modal_data,  # optional
        })

    ds = ray.data.from_items(requests_data)

    @ray.remote(num_gpus=1)
    class VLLMActor:
        def __init__(self, model: str, engine_args: dict):
            from vllm import LLM
            self.llm = LLM(**dataclasses.asdict(engine_args))

        def generate(self, batch: List[dict]):
            from vllm import SamplingParams
            # for row in batch:
            #     print(row)
            # batch = batch.to_dict(orient="records")
            # print(batch)
            # print(len(batch["prompt"]))
            prompts = batch["prompt"]
            sampling_params = [SamplingParams(n=1, temperature=1.0, top_p=1.0,
                                            ignore_eos=True, max_tokens=expected_output_len) for expected_output_len in batch["expected_output_len"]]
            start_time = time.perf_counter()
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
            end_time = time.perf_counter()
            print(f"[RayData] llm.generate Time taken: {end_time - start_time:.2f} seconds")
            # Convert to Ray-friendly dicts
            result = []
            for output in outputs:
                # output.outputs[0].text is the generated string
                result.append({"generated_text": output.outputs[0].text})

            return {"generated_text": result}


    # Usage:
    actor = VLLMActor.options(num_gpus=1).remote(model, EngineArgs.from_cli_args(args))

    ds = ray.data.from_items(requests_data)
    ds = ds.map_batches(lambda batch: ray.get(actor.generate.remote(batch)),
                        batch_size=10000, zero_copy_batch=True)

    start = time.perf_counter()
    print("Start Running inference...")
    _ = ds.take_all()
    end = time.perf_counter()
    return end - start

def run_ray_data_optimized(
    requests: List[SampleRequest],
    model: str,
    concurrency: int,
    batch_size: int,
) -> float:
    import ray
    import pandas as pd
    from ray.data import from_items

    @ray.remote(num_gpus=1)
    class VLLMActor:
        def __init__(self, model: str, engine_args: dict):
            from vllm import LLM, SamplingParams
            self.llm = LLM(**dataclasses.asdict(engine_args))
            self.SamplingParams = SamplingParams

        def generate(self, batch: pd.DataFrame):
            prompts = batch["prompt"].tolist()
            output_lens = batch["expected_output_len"].tolist()
            sampling_params = [self.SamplingParams(
                n=1, temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=l
            ) for l in output_lens]
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
            texts = [o.outputs[0].text for o in outputs]
            return pd.DataFrame({"generated_text": texts})

    actor = VLLMActor.options(num_gpus=1).remote(model, EngineArgs.from_cli_args(args))

    # Prepare data
    requests_data = [{"prompt": r.prompt, "expected_output_len": r.expected_output_len} for r in requests]
    ds = from_items(requests_data)

    # Optimized pipeline
    ds = ds.map_batches(
        lambda batch: ray.get(actor.generate.remote(batch)),
        batch_size=batch_size,
        zero_copy_batch=True,
        batch_format="pandas",
    )

    start = time.perf_counter()
    _ = ds.materialize()  # Triggers execution but avoid full `take_all()` blocking fetch
    ray.data.context.DataContext.get_current().execution_options.preserve_order = False  # Optional: Skip preserving input order
    end = time.perf_counter()

    return end - start

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # -------------------------------------------------------------------------
    # 1) Gather all requests into a list of SampleRequest
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.dataset is None:
        # Synthesize random requests
        vocab_size = tokenizer.vocab_size
        requests = []
        for _ in range(args.num_prompts):
            request_tokenizer = tokenizer
            lora_request: Optional[LoRARequest] = None
            if args.enable_lora:
                lora_request, lora_tokenizer = get_random_lora_request(args)
                if lora_tokenizer:
                    request_tokenizer = lora_tokenizer

            # Synthesize a prompt with the given input length.
            candidate_ids = [
                random.randint(0, vocab_size - 1)
                for _ in range(args.input_len)
            ]
            for _ in range(5):
                candidate_prompt = request_tokenizer.decode(candidate_ids)
                tokenized_len = len(request_tokenizer.encode(candidate_prompt))
                if tokenized_len == args.input_len:
                    break
                diff = args.input_len - tokenized_len
                if diff > 0:
                    candidate_ids.extend([
                        random.randint(100, vocab_size - 100)
                        for _ in range(diff)
                    ])
                else:
                    candidate_ids = candidate_ids[:diff]
            requests.append(
                SampleRequest(
                    prompt=candidate_prompt,
                    prompt_len=args.input_len,
                    expected_output_len=args.output_len,
                    lora_request=lora_request,
                )
            )
    else:
        # Load from JSON dataset
        requests = sample_requests(tokenizer, args)

    is_multi_modal = any(request.multi_modal_data is not None for request in requests)

    # -------------------------------------------------------------------------
    # 2) Run the chosen backend
    # -------------------------------------------------------------------------
    if args.backend == "vllm":
        if args.async_engine:
            print(
                "Running vLLM async engine (OpenAI-like) with multiprocessing frontend."
            )
            elapsed_time = uvloop.run(
                run_vllm_async_ray_style(
                    requests,
                    args.n,
                    AsyncEngineArgs.from_cli_args(args),
                    args.disable_frontend_multiprocessing,
                )
            )
        else:
            print(
                "Running vLLM default engine (LLM class)"
            )
            elapsed_time = run_vllm(requests, args.n, EngineArgs.from_cli_args(args))

    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(
            requests,
            args.model,
            tokenizer,
            args.n,
            args.hf_max_batch_size,
            args.trust_remote_code,
        )

    elif args.backend == "mii":
        elapsed_time = run_mii(
            requests,
            args.model,
            args.tensor_parallel_size,
            args.output_len,
        )

    # -------------------------------------------------------------------------
    # 3) NEW: Ray Data backend
    # -------------------------------------------------------------------------
    elif args.backend == "ray_data":
        concurrency = 1
        batch_size = 256

        # If you want to override concurrency / batch_size from separate CLI flags,
        # add them in your parser and reference them here.
        elapsed_time = run_ray_data(
            requests,
            model=args.model,
            concurrency=concurrency,
            batch_size=batch_size,
        )
        

    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # -------------------------------------------------------------------------
    # 4) Compute throughput stats
    # -------------------------------------------------------------------------
    total_num_tokens = sum(r.prompt_len + r.expected_output_len for r in requests)
    total_output_tokens = sum(r.expected_output_len for r in requests)

    if is_multi_modal:
        print(
            "\033[91mWARNING\033[0m: Multi-modal request detected. The following metrics "
            "are not entirely accurate because image tokens are not counted. "
            "See vllm-project/vllm/issues/9778 for details."
        )

    req_throughput = len(requests) / elapsed_time if elapsed_time > 0 else 0.0
    total_tok_throughput = total_num_tokens / elapsed_time if elapsed_time > 0 else 0.0
    output_tok_throughput = total_output_tokens / elapsed_time if elapsed_time > 0 else 0.0

    print(
        f"Throughput: {req_throughput:.2f} requests/s, "
        f"{total_tok_throughput:.2f} total tokens/s, "
        f"{output_tok_throughput:.2f} output tokens/s"
    )

    # -------------------------------------------------------------------------
    # 5) Optionally save JSON results
    # -------------------------------------------------------------------------
    results = {
        "elapsed_time": elapsed_time,
        "num_requests": len(requests),
        "total_num_tokens": total_num_tokens,
        "requests_per_second": req_throughput,
        "tokens_per_second (total)": total_tok_throughput,
        "tokens_per_second (output)": output_tok_throughput,
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
    else:
        # Example auto-filename
        base_dir = "/home/yilegu/ray"
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if "/" in args.model:
            model_name = args.model.split("/")[-1]
        else:
            model_name = args.model
        filename = os.path.join(
            base_dir,
            f"throughput_{model_name}_{args.backend}_{timestamp}.json"
        )
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii", "ray_data"],
                        default="vllm")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to the dataset (JSON).")
    parser.add_argument("--input-len", type=int, default=None,
                        help="Input prompt length for each request.")
    parser.add_argument("--output-len", type=int, default=None,
                        help="Output length for each request.")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--hf-max-batch-size", type=int, default=None,
                        help="Maximum batch size for HF or ray_data.")
    parser.add_argument('--output-json', type=str, default=None,
                        help='Path to save throughput results in JSON.')
    parser.add_argument("--async-engine", action='store_true', default=False,
                        help="Use vLLM async engine (OpenAI-like) vs LLM class.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    # LoRA
    parser.add_argument("--lora-path", type=str, default=None)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # Additional checks:
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None, "Must specify --input-len if no dataset"
        assert args.output_len is not None, "Must specify --output-len if no dataset"
    else:
        assert args.input_len is None, "Cannot specify --input-len if using a dataset"
    if args.enable_lora:
        assert args.lora_path is not None, "Must provide --lora-path with --enable-lora"

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("hf_max_batch_size is only relevant for HF or ray_data backends.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("hf_max_batch_size required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.enable_lora:
            raise ValueError("LoRA is not currently supported for HF backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("hf_max_batch_size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII.")
        if args.enable_lora:
            raise ValueError("LoRA not supported for MII in this script.")
    elif args.backend == "ray_data":
        # We allow concurrency/batch_size from existing flags or set defaults
        pass

    # Run benchmark
    main(args)
