
from oxrl import Trainer

# 1. Initialize Trainer
trainer = Trainer(
    model="google/gemma-3-1b-it",
    experiment_id="simple_run"
)

# 2. Run Training
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1,
    steps_per_epoch=2
)
