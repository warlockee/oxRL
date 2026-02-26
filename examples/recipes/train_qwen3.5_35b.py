from oxrl import Trainer

trainer = Trainer(model="Qwen/Qwen3.5-35B-A3B")
trainer.train(
    task="reasoning",
    train_file="examples/data/train.jsonl",
    training_gpus=2,
    rollout_gpus=2,
    epochs=1
)
