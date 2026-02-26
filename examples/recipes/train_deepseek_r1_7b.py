from oxrl import Trainer

trainer = Trainer(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=2,
    rollout_gpus=2,
    epochs=1
)
