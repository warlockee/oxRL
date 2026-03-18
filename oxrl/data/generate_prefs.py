"""
Preference data generation pipeline for NeurIPS 2026 experiments.

Takes a prompt-only dataset, generates N responses per prompt using vLLM,
scores each response with a reward function, and outputs a
(prompt, chosen, rejected) Parquet file.

The output is compatible with oxrl.datasets.prompt_preference.PromptPreferenceDataset.

Usage:
    python -m oxrl.data.generate_prefs \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --dataset ./data/gsm8k_qwen2.5-1.5b-instruct_wsp_train.parquet \
        --reward gsm8k_reward_func \
        --n-responses 16 \
        --output /home/ec2-user/fsx/oxrl_data/neurips2026/gsm8k_1.5b_prefs.parquet

    # Best-of-N evaluation (no training, just pick highest-reward response):
    python -m oxrl.data.generate_prefs \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --dataset ./data/gsm8k_qwen2.5-1.5b-instruct_wsp_test.parquet \
        --reward gsm8k_reward_func \
        --n-responses 64 \
        --best-of-n \
        --output /home/ec2-user/fsx/oxrl_data/neurips2026/gsm8k_1.5b_bon64.json
"""
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Lazy-import vLLM to avoid CUDA initialization errors when importing
# the module for unit testing or on non-GPU nodes.
LLM = None
SamplingParams = None

def _ensure_vllm():
    global LLM, SamplingParams
    if LLM is None:
        from vllm import LLM as _LLM, SamplingParams as _SP
        LLM = _LLM
        SamplingParams = _SP


# ──────────────────────────────────────────────────────────────────────
# Reward resolution
# ──────────────────────────────────────────────────────────────────────

def resolve_reward_func(name: str) -> Callable:
    """Resolve a reward function name to a callable.

    Supports both short names (e.g. 'gsm8k_reward_func') that are looked up
    in oxrl.rewards, and fully-qualified dotted paths.
    """
    if "." in name:
        module_path, func_name = name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    module = importlib.import_module("oxrl.rewards")
    return getattr(module, name)


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_prompts(
    dataset_path: str,
    prompt_key: str = "prompt",
    answer_key: str = "answer",
    max_prompts: int = 0,
) -> List[Dict]:
    """Load prompts from a Parquet or JSONL file.

    Returns a list of dicts, each with at minimum 'prompt' (list of message
    dicts) and optionally 'answer' (ground truth for reward scoring).
    """
    fmt = "json" if dataset_path.endswith((".jsonl", ".json")) else "parquet"
    ds = load_dataset(fmt, data_files=dataset_path, split="train")

    if max_prompts > 0:
        ds = ds.select(range(min(max_prompts, len(ds))))

    records = []
    for row in ds:
        prompt = row[prompt_key]
        record = {"prompt": prompt}
        if answer_key and answer_key in row:
            record["answer"] = row[answer_key]
        records.append(record)

    return records


# ──────────────────────────────────────────────────────────────────────
# Generation + scoring
# ──────────────────────────────────────────────────────────────────────

def generate_and_score(
    llm: LLM,
    tokenizer: AutoTokenizer,
    prompts: List[Dict],
    reward_fn: Callable,
    n_responses: int = 16,
    max_tokens: int = 512,
    temperature: float = 1.0,
    batch_size: int = 256,
) -> List[Dict]:
    """Generate n_responses per prompt, score each, return scored results.

    Returns a list of dicts, one per prompt:
        {
            "prompt": [...],           # original message list
            "answer": "...",           # ground truth (if available)
            "responses": [
                {"text": "...", "reward": 0.0, "finish_reason": "..."},
                ...
            ]
        }
    """
    _ensure_vllm()
    sampling_params = SamplingParams(
        n=n_responses,
        temperature=temperature,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_tokens,
        logprobs=0,
    )

    # Tokenize prompts for vLLM
    vllm_inputs = []
    for p in prompts:
        prompt_ids = tokenizer.apply_chat_template(
            conversation=p["prompt"],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )
        vllm_inputs.append({"prompt_token_ids": prompt_ids})

    # Generate in batches to avoid OOM on large datasets
    all_outputs = []
    for start in range(0, len(vllm_inputs), batch_size):
        batch = vllm_inputs[start : start + batch_size]
        outputs = llm.generate(batch, sampling_params=sampling_params, use_tqdm=True)
        all_outputs.extend(outputs)

    # Score each response
    results = []
    for prompt_idx, output in enumerate(all_outputs):
        prompt_data = prompts[prompt_idx]
        prompt_ids = list(output.prompt_token_ids or [])
        metadata_base = {}
        if "answer" in prompt_data:
            metadata_base["answer"] = prompt_data["answer"]

        scored_responses = []
        for resp in output.outputs:
            response_ids = list(resp.token_ids)
            finish_reason = getattr(resp, "finish_reason", None)

            # Build metadata for reward function
            metadata = dict(metadata_base)
            metadata["response_text"] = getattr(resp, "text", "")

            # Score
            reward_tensor, _ = reward_fn(
                prompt_ids, response_ids, finish_reason, metadata=metadata
            )
            total_reward = float(reward_tensor.sum().item())

            scored_responses.append({
                "text": metadata["response_text"],
                "reward": total_reward,
                "finish_reason": str(finish_reason),
            })

        results.append({
            "prompt": prompt_data["prompt"],
            "answer": prompt_data.get("answer", ""),
            "responses": scored_responses,
        })

    return results


# ──────────────────────────────────────────────────────────────────────
# Preference pair extraction
# ──────────────────────────────────────────────────────────────────────

def extract_preference_pairs(
    scored_results: List[Dict],
) -> List[Dict]:
    """From scored results, extract (prompt, chosen, rejected) triplets.

    For each prompt, chosen = highest reward response, rejected = lowest.
    Prompts where all responses have the same reward are discarded (no signal).

    Returns list of dicts compatible with PromptPreferenceDataset:
        {"prompt": [...], "chosen": "...", "rejected": "...", "answer": "..."}
    """
    pairs = []
    discarded = 0

    for result in scored_results:
        responses = result["responses"]
        if not responses:
            discarded += 1
            continue

        rewards = [r["reward"] for r in responses]
        max_reward = max(rewards)
        min_reward = min(rewards)

        # Skip prompts with no reward signal
        if abs(max_reward - min_reward) < 1e-8:
            discarded += 1
            continue

        # Select best and worst
        best_idx = rewards.index(max_reward)
        worst_idx = rewards.index(min_reward)

        pairs.append({
            "prompt": result["prompt"],
            "chosen": responses[best_idx]["text"],
            "rejected": responses[worst_idx]["text"],
            "answer": result.get("answer", ""),
            "chosen_reward": max_reward,
            "rejected_reward": min_reward,
        })

    print(f"Extracted {len(pairs)} preference pairs, discarded {discarded} prompts (no signal).")
    return pairs


# ──────────────────────────────────────────────────────────────────────
# Best-of-N evaluation
# ──────────────────────────────────────────────────────────────────────

def best_of_n_eval(scored_results: List[Dict]) -> Dict:
    """Evaluate Best-of-N: for each prompt, pick the highest-reward response.

    Returns summary dict with accuracy and per-prompt details.
    """
    correct = 0
    total = len(scored_results)
    per_prompt = []

    for result in scored_results:
        responses = result["responses"]
        if not responses:
            per_prompt.append({"reward": 0.0, "correct": False})
            continue

        # Pick the response with highest reward
        best = max(responses, key=lambda r: r["reward"])
        is_correct = best["reward"] > 0.5  # binary reward threshold

        if is_correct:
            correct += 1

        per_prompt.append({
            "reward": best["reward"],
            "correct": is_correct,
            "text": best["text"][:200],  # truncate for storage
        })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "mode": "best_of_n",
        "n_responses": len(scored_results[0]["responses"]) if scored_results else 0,
        "num_prompts": total,
        "num_correct": correct,
        "accuracy": accuracy,
        "per_prompt": per_prompt,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate preference data or run Best-of-N evaluation."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to prompt-only dataset (Parquet or JSONL).",
    )
    parser.add_argument(
        "--reward", type=str, default="gsm8k_reward_func",
        help="Reward function name (from oxrl.rewards) or dotted path.",
    )
    parser.add_argument(
        "--n-responses", type=int, default=16,
        help="Number of responses to generate per prompt.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens per response.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for Parquet (preference) or JSON (best-of-n).",
    )
    parser.add_argument(
        "--best-of-n", action="store_true",
        help="Run Best-of-N evaluation instead of generating preference data.",
    )
    parser.add_argument(
        "--prompt-key", type=str, default="prompt",
        help="Column name for prompts in the dataset.",
    )
    parser.add_argument(
        "--answer-key", type=str, default="answer",
        help="Column name for ground-truth answers.",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=0,
        help="Max number of prompts to process (0 = all).",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85,
        help="vLLM GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Number of prompts per vLLM generation batch.",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Trust remote code when loading model/tokenizer.",
    )
    args = parser.parse_args()

    # Validate output path is on fsx (not root disk)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[generate_prefs] Model:        {args.model}")
    print(f"[generate_prefs] Dataset:      {args.dataset}")
    print(f"[generate_prefs] Reward:       {args.reward}")
    print(f"[generate_prefs] N responses:  {args.n_responses}")
    print(f"[generate_prefs] Mode:         {'best-of-n' if args.best_of_n else 'preference-pairs'}")
    print(f"[generate_prefs] Output:       {args.output}")

    # 1. Load reward function
    reward_fn = resolve_reward_func(args.reward)
    print(f"[generate_prefs] Reward function loaded: {reward_fn.__name__}")

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load prompts
    prompts = load_prompts(
        args.dataset,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
        max_prompts=args.max_prompts,
    )
    print(f"[generate_prefs] Loaded {len(prompts)} prompts")

    # 4. Load vLLM engine
    _ensure_vllm()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    # 5. Generate and score
    t0 = time.time()
    scored_results = generate_and_score(
        llm=llm,
        tokenizer=tokenizer,
        prompts=prompts,
        reward_fn=reward_fn,
        n_responses=args.n_responses,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0
    print(f"[generate_prefs] Generation + scoring completed in {elapsed:.1f}s")

    # 6. Output
    if args.best_of_n:
        # Best-of-N evaluation mode
        bon_result = best_of_n_eval(scored_results)
        bon_result["model"] = args.model
        bon_result["dataset"] = args.dataset
        bon_result["reward_func"] = args.reward
        bon_result["elapsed_seconds"] = elapsed

        with open(args.output, "w") as f:
            json.dump(bon_result, f, indent=2)
        print(f"[generate_prefs] Best-of-{args.n_responses} accuracy: "
              f"{bon_result['accuracy']:.4f} ({bon_result['num_correct']}/{bon_result['num_prompts']})")
        print(f"[generate_prefs] Results saved to {args.output}")

    else:
        # Preference pair extraction mode
        pairs = extract_preference_pairs(scored_results)

        if not pairs:
            print("[generate_prefs] WARNING: No preference pairs extracted. "
                  "All prompts had uniform reward. Check your reward function and data.")
            sys.exit(1)

        df = pd.DataFrame(pairs)
        df.to_parquet(args.output, index=False)
        print(f"[generate_prefs] Saved {len(df)} preference pairs to {args.output}")

        # Print summary statistics
        print(f"[generate_prefs] Chosen reward  -- mean: {df['chosen_reward'].mean():.4f}, "
              f"std: {df['chosen_reward'].std():.4f}")
        print(f"[generate_prefs] Rejected reward -- mean: {df['rejected_reward'].mean():.4f}, "
              f"std: {df['rejected_reward'].std():.4f}")

    # Cleanup
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
