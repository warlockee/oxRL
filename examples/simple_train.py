
from oxrl import Trainer

# 1. Initialize Trainer
trainer = Trainer(
    model="google/gemma-3-1b-it",
    experiment_id="simple_run"
)

# 2. Run Training
trainer.train(
    dataset="gsm8k",
    epochs=1,
    steps_per_epoch=2
)
