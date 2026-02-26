from oxrl import Trainer

trainer = Trainer(model="Qwen/Qwen2.5-Coder-1.5B-Instruct")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1
)
