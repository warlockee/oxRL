from oxrl import Trainer

trainer = Trainer(model="Qwen/Qwen3-8B")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=2,
    rollout_gpus=2,
    epochs=1
)
