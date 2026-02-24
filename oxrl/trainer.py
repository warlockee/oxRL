import os
import sys
import yaml
import subprocess
import torch
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class Trainer:
    """
    oxRL High-level Trainer for minimal configuration.
    
    Example:
        trainer = Trainer(model="google/gemma-3-1b-it")
        trainer.train(dataset="gsm8k") # Auto-downloads, preps, and trains
    """
    def __init__(self, model: str, experiment_id: Optional[str] = None):
        self.model = model
        self.model_slug = model.split("/")[-1].lower()
        self.experiment_id = experiment_id or f"run_{self.model_slug}"
        
    def _detect_gpus(self) -> int:
        """Detect number of available GPUs."""
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0

    def _prepare_dataset(self, dataset_name: str) -> tuple[str, str]:
        """Auto-prepare dataset using preprocessing scripts."""
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        
        train_file = data_dir / f"{dataset_name}_{self.model_slug}_wsp_train.parquet"
        test_file = data_dir / f"{dataset_name}_{self.model_slug}_wsp_test.parquet"
        
        if train_file.exists() and test_file.exists():
            print(f"[oxRL] Dataset {dataset_name} already prepared at {train_file}")
            return str(train_file), str(test_file)
            
        script_map = {
            "gsm8k": "preprocessing/gsm8k.py",
            "math_hard": "preprocessing/math_hard.py",
            "mbpp": "preprocessing/mbpp.py",
            "ultrafeedback": "preprocessing/ultrafeedback.py",
        }
        
        if dataset_name not in script_map:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: {list(script_map.keys())}")
            
        script_path = PROJECT_ROOT / script_map[dataset_name]
        print(f"[oxRL] Preparing dataset {dataset_name}...")
        
        cmd = [
            sys.executable, str(script_path),
            "--local_dir", str(data_dir),
            "--run_id", self.model_slug
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[oxRL] Preprocessing failed:\n{result.stderr}")
            raise RuntimeError(f"Failed to prepare dataset {dataset_name}")
            
        return str(train_file), str(test_file)

    def train(self, 
              dataset: Optional[str] = None,
              train_file: Optional[str] = None, 
              val_file: Optional[str] = None, 
              training_gpus: Optional[int] = None, 
              rollout_gpus: Optional[int] = None,
              alg: str = "sgrpo",
              epochs: int = 3,
              steps_per_epoch: int = 10,
              batch_size: int = 2):
        """Run RL training with simplified parameters."""
        
        # 1. Hardware Detection
        available_gpus = self._detect_gpus()
        if training_gpus is None:
            training_gpus = max(1, available_gpus // 2)
        if rollout_gpus is None:
            rollout_gpus = max(1, available_gpus - training_gpus)
            
        print(f"[oxRL] Using {training_gpus} training GPUs and {rollout_gpus} rollout GPUs (Total: {available_gpus})")

        # 2. Data Preparation
        if dataset:
            train_path, val_path = self._prepare_dataset(dataset)
        elif train_file:
            train_path = str(Path(train_file).absolute())
            val_path = str(Path(val_file or train_file).absolute())
        else:
            raise ValueError("Either 'dataset' name or 'train_file' path must be provided.")

        # 3. Prepare minimal config
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
                "train_files_path": train_path,
                "val_files_path": val_path,
            },
        }
        
        # 4. Save config to a temporary location
        config_dir = PROJECT_ROOT / "onboarded" / self.experiment_id
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"[oxRL] Config generated: {config_path}")
        
        # 5. Call main_rl.main()
        from main_rl import main as run_rl
        run_rl(config_file=str(config_path), experiment_id=self.experiment_id)
