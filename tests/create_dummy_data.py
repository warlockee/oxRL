import pandas as pd
import os

def create_dummy_preference_data(file_path):
    data = [
        {
            "prompt": [{"role": "user", "content": "What is the capital of France?"}],
            "chosen": "The capital of France is Paris.",
            "rejected": "The capital of France is Berlin."
        },
        {
            "prompt": [{"role": "user", "content": "Who wrote Hamlet?"}],
            "chosen": "William Shakespeare wrote Hamlet.",
            "rejected": "Charles Dickens wrote Hamlet."
        },
        {
            "prompt": [{"role": "user", "content": "What is the formula for water?"}],
            "chosen": "The chemical formula for water is H2O.",
            "rejected": "The chemical formula for water is CO2."
        }
    ]
    df = pd.DataFrame(data)
    df.to_parquet(file_path, index=False)
    print(f"Dummy data created at {file_path}")

if __name__ == "__main__":
    os.makedirs("tests/data", exist_ok=True)
    create_dummy_preference_data("tests/data/dummy_preference.parquet")
