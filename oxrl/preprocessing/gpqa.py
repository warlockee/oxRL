import argparse
import os
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default="data")
    parser.add_argument("--run_id", type=str, default="gpqa")
    args = parser.parse_args()

    print(f"Loading GPQA...")
    # Load GPQA Diamond (the most difficult split)
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    
    def process_fn(example):
        # GPQA is multiple choice. We format it for reasoning.
        question = example["Question"]
        options = [example["Correct Answer"], example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
        # Shuffling is usually handled by the dataset or here
        import random
        indices = [0, 1, 2, 3]
        random.shuffle(indices)
        shuffled_options = [options[i] for i in indices]
        correct_letter = chr(65 + indices.index(0)) # A, B, C, or D
        
        prompt_text = f"{question}\n\nOptions:\n"
        for i, opt in enumerate(shuffled_options):
            prompt_text += f"{chr(65+i)}. {opt}\n"
        
        return {
            "prompt": [
                {"role": "system", "content": "You are a highly capable scientist. Solve the following graduate-level problem step by step, thinking carefully. Wrap your final answer letter (A, B, C, or D) in <answer> tags."},
                {"role": "user", "content": prompt_text}
            ],
            "answer": correct_letter,
        }

    print("Mapping dataset...")
    ds = ds.map(process_fn, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    
    os.makedirs(args.local_dir, exist_ok=True)
    ds["train"].to_parquet(os.path.join(args.local_dir, f"gpqa_{args.run_id}_wsp_train.parquet"))
    ds["test"].to_parquet(os.path.join(args.local_dir, f"gpqa_{args.run_id}_wsp_test.parquet"))
    print("Done!")

if __name__ == "__main__":
    main()
