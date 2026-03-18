"""
Extract SFT data from preference data.

Takes a preference Parquet file (prompt, chosen, rejected) and outputs
a prompt-response Parquet file (prompt, answer) using only the chosen responses.

Usage:
    python -m oxrl.data.extract_sft \
        --input /home/ec2-user/fsx/oxrl_data/neurips2026/gsm8k_qwen2.5-0.5b-instruct_prefs_train.parquet \
        --output /home/ec2-user/fsx/oxrl_data/neurips2026/gsm8k_qwen2.5-0.5b-instruct_sft_train.parquet
"""
import argparse
import pandas as pd


def extract_sft_data(input_path: str, output_path: str) -> None:
    df = pd.read_parquet(input_path)

    sft_records = []
    for _, row in df.iterrows():
        sft_records.append({
            "prompt": row["prompt"],
            "answer": row["chosen"],
        })

    sft_df = pd.DataFrame(sft_records)
    sft_df.to_parquet(output_path, index=False)
    print(f"Extracted {len(sft_df)} SFT examples from {input_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract SFT data from preference data.")
    parser.add_argument("--input", type=str, required=True, help="Input preference Parquet file.")
    parser.add_argument("--output", type=str, required=True, help="Output SFT Parquet file.")
    args = parser.parse_args()
    extract_sft_data(args.input, args.output)


if __name__ == "__main__":
    main()
