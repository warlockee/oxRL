import os
import sys
import yaml
import subprocess
import torch
from pathlib import Path
from typing import Optional, Any, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swarm.config_generator import generate_config, save_config

class Trainer:
    """
    oxRL High-level Trainer for minimal configuration.
    
    Fulfils the promise: "Post-train any model under 10 lines of code."
    
    Example:
        from oxrl import Trainer
        trainer = Trainer(model="Qwen/Qwen2.5-0.5B-Instruct")
        trainer.train(task="math") 
    """
    def __init__(self, model: str, experiment_id: Optional[str] = None):
        self.model = model
        self.experiment_id = experiment_id
        
    def _get_param_count(self) -> float:
        """Estimate param count from model string if possible, or use a safe default."""
        import re
        # Try to find numbers followed by B or b
        match = re.search(r"(\d+\.?\d*)[Bb]", self.model)
        if match:
            return float(match.group(1))
        
        # Default to 7B for safety (will trigger LoRA and 2+2 GPU)
        return 7.0

    def train(self, 
              task: str = "math",
              dataset: Optional[str] = None,
              epochs: int = 1,
              steps_per_epoch: int = 10,
              **kwargs):
        """
        Run RL training with minimal boilerplate.
        
        Args:
            task: Task type ('math', 'reasoning', 'code', 'instruct', 'vision', 'audio')
            dataset: Optional dataset name override
            epochs: Number of training epochs
            steps_per_epoch: Number of optimizer steps per epoch
            **kwargs: Config overrides (e.g. lora_enabled=True, lr=1e-5)
        """
        params_b = self._get_param_count()
        
        # 1. Auto-generate config using the Swarm logic
        config_dict = generate_config(
            model_name=self.model,
            task=task,
            param_count_b=params_b,
            experiment_id=self.experiment_id
        )
        
        # 2. Apply overrides
        if dataset:
            # We assume the user has preprocessed this dataset if it is custom
            config_dict["data"]["train_dnames"] = [dataset]
            config_dict["data"]["train_ratios"] = {dataset: 1.0}
            
        config_dict["train"]["total_number_of_epochs"] = epochs
        config_dict["train"]["train_steps_per_epoch"] = steps_per_epoch
        
        # Merge other kwargs into flat sections if they match keys
        for key, val in kwargs.items():
            found = False
            for section in ["train", "run", "model", "rollout", "lora"]:
                if section in config_dict and key in config_dict[section]:
                    config_dict[section][key] = val
                    found = True
                    break
            if not found:
                # Add to train by default or keep as is
                config_dict["train"][key] = val

        # 3. Save config
        config_dir = PROJECT_ROOT / "onboarded" / config_dict["run"]["experiment_id"]
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        
        save_config(config_dict, str(config_path))
        print(f"[oxRL] Starting training for {self.model} on task '{task}'")
        print(f"[oxRL] Config: {config_path}")

        # 4. Launch Training
        # We use subprocess to ensure clean environment and Ray isolation
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "main_rl.py"),
            "--config-file", str(config_path),
            "--experiment_id", config_dict["run"]["experiment_id"]
        ]
        
        # Propagate environment variables (HF tokens, etc.)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[oxRL] Training failed with exit code {e.returncode}")
            raise
