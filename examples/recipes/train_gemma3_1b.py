import subprocess
import os
from oxrl import Trainer

def main():
    model_name = "google/gemma-3-1b-it"
    
    # 1. Preprocess the reasoning data first
    print(f"--- Preprocessing data for {model_name} ---")
    preprocess_cmd = [
        "python", "oxrl/preprocessing/openr1_math.py",
        "--local_dir", "./data",
        "--run_id", "gemma-3-1b-it"
    ]
    subprocess.run(preprocess_cmd, check=True)

    # 2. Kick off training
    print(f"--- Starting Post-training for {model_name} ---")
    trainer = Trainer(model=model_name)
    
    # We use reasonable defaults for a 1B model
    trainer.train(
        task="reasoning",
        epochs=3,
        steps_per_epoch=100,
        # Config overrides if needed
        trust_remote_code=True,
        attn_implementation="eager"
    )

if __name__ == "__main__":
    main()
