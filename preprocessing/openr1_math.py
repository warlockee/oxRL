import argparse
import os
import pandas as pd
from datasets import load_dataset
from typing import List, Dict

def create_prompt(problem: str):
    # Standard DeepSeek-R1 style prompt for reasoning models
    return [
        {
            "role": "system",
            "content": "A conversation between a user and an LLM. The LLM first thinks about the problem and then provides an answer. The thought process is wrapped in <thought> tags, and the final answer is wrapped in <answer> tags."
        },
        {
            "role": "user",
            "content": f"{problem}\\nSolve the problem step by step and wrap your final answer in <answer> tags."
        }
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default="data")
    parser.add_argument("--run_id", type=str, default="openr1_math")
    parser.add_argument("--use_system_prompt", type=str, default="True")
    args = parser.parse_args()

    print(f"Loading OpenR1-Math-220k...")
    # Load a reasonable subset for post-training verification
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train", num_proc=4)
    
    # Filter for high quality/verifiable ones if possible, but for now just use it
    
    def process_fn(example):
        return {
            "prompt": create_prompt(example["problem"]),
            "answer": example["answer"],
            "problem_type": example.get("problem_type", "math"),
        }

    print("Mapping dataset...")
    ds = ds.map(process_fn, remove_columns=ds.column_names)
    
    # Split into train/test
    ds = ds.train_test_split(test_size=0.05, seed=42)
    
    os.makedirs(args.local_dir, exist_ok=True)
    
    train_path = os.path.join(args.local_dir, f"openr1_math_{args.run_id}_wsp_train.parquet")
    test_path = os.path.join(args.local_dir, f"openr1_math_{args.run_id}_wsp_test.parquet")
    
    print(f"Saving to {train_path}...")
    ds["train"].to_parquet(train_path)
    ds["test"].to_parquet(test_path)
    print("Done!")

if __name__ == "__main__":
    main()
