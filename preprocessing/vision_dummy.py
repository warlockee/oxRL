import argparse
import os
import datasets
import io
import base64
from PIL import Image

def create_prompt(question, model_slug):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

def make_map_fn(split, params):
    def process_fn(example, idx):
        # We just create a dummy red image for testing
        img = Image.new('RGB', (100, 100), color = 'red')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        question = "What is the result of 1+1?"
        answer = "2"
        data = {
            "prompt": create_prompt(question, params.run_id),
            "answer": answer,
            "image_base64": img_str, # Add image
            "split": split,
            "index": idx,
        }
        return data
    return process_fn

def create_file_name(params, split):
    return f"vision_dummy_{params.run_id}_wsp_{split}.parquet"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="dummy")
    parser.add_argument("--use_system_prompt", default=False)
    args = parser.parse_args()
    
    # Just make 2 examples
    ds = datasets.Dataset.from_dict({"dummy": [0, 1]})
    
    train_dataset = ds.map(function=make_map_fn("train", params=args), with_indices=True)
    test_dataset = ds.map(function=make_map_fn("test", params=args), with_indices=True)

    train_file_name = os.path.join(args.local_dir, create_file_name(args, "train"))
    test_file_name = os.path.join(args.local_dir, create_file_name(args, "test"))
    train_dataset.to_parquet(train_file_name)
    test_dataset.to_parquet(test_file_name)
    print("Done")
