import argparse
import os
import datasets
import io
import base64
import numpy as np
import soundfile as sf

def create_prompt(question):
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": question}
            ]
        }
    ]

def make_map_fn(split, params):
    def process_fn(example, idx):
        # Create a dummy 1-second sine wave
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate)
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        buffered = io.BytesIO()
        sf.write(buffered, audio_data, sample_rate, format='WAV')
        audio_str = base64.b64encode(buffered.getvalue()).decode()
        
        question = "What is in this audio?"
        answer = "A sine wave."
        data = {
            "prompt": create_prompt(question),
            "answer": answer,
            "audio_base64": audio_str,
            "split": split,
            "index": idx,
        }
        return data
    return process_fn

def create_file_name(params, split):
    return f"audio_dummy_{params.run_id}_wsp_{split}.parquet"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="dummy")
    parser.add_argument("--use_system_prompt", default=False)
    args = parser.parse_args()
    
    ds = datasets.Dataset.from_dict({"dummy": [0, 1]})
    
    train_dataset = ds.map(function=make_map_fn("train", params=args), with_indices=True)
    test_dataset = ds.map(function=make_map_fn("test", params=args), with_indices=True)

    train_file_name = os.path.join(args.local_dir, create_file_name(args, "train"))
    test_file_name = os.path.join(args.local_dir, create_file_name(args, "test"))
    train_dataset.to_parquet(train_file_name)
    test_dataset.to_parquet(test_file_name)
    print("Done")
