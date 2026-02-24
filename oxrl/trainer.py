
import os
import sys
import yaml
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

class Trainer:
    """
    oxRL High-level Trainer for minimal configuration.
    Example:
        trainer = Trainer(model="google/gemma-3-1b-it")
        trainer.train(data="train.jsonl", gpus=2)
    """
    def __init__(self, model: str, experiment_id: str = "oxrl_run"):
        self.model = model
        self.experiment_id = experiment_id
        
    def train(self, 
              train_file: str, 
              val_file: str = None, 
              training_gpus: int = 2, 
              rollout_gpus: int = 2,
              alg: str = "sgrpo",
              epochs: int = 3,
              steps_per_epoch: int = 10,
              batch_size: int = 2):
        """Run RL training with simplified parameters."""
        
        # 1. Prepare minimal config
        config = {
            "run": {
                "experiment_id": self.experiment_id,
                "training_gpus": training_gpus,
                "rollout_gpus": rollout_gpus,
            },
            "train": {
                "alg_name": alg,
                "total_number_of_epochs": epochs,
                "train_steps_per_epoch": steps_per_epoch,
                "train_batch_size_per_gpu": batch_size,
            },
            "model": {
                "name": self.model,
            },
            "data": {
                "train_dnames": ["custom_data"],
                "train_ratios": {"custom_data": 1.0},
                "train_files_path": str(Path(train_file).absolute()),
                "val_files_path": str(Path(val_file or train_file).absolute()),
            },
        }
        
        # 2. Save config to a temporary location
        config_dir = PROJECT_ROOT / "onboarded" / self.experiment_id
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"[oxRL] Config generated: {config_path}")
        
        # 3. Call main_rl.main()
        from main_rl import main as run_rl
        run_rl(config_file=str(config_path), experiment_id=self.experiment_id)
