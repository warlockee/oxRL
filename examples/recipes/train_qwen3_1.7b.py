from oxrl import Trainer

trainer = Trainer(model="Qwen/Qwen3-1.7B")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1
)
