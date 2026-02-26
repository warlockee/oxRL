from oxrl import Trainer

trainer = Trainer(model="Qwen/Qwen3-4B-Thinking-2507")
trainer.train(
    task="reasoning",
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1
)
