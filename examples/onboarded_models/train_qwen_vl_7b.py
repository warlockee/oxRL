from oxrl import Trainer

# Larger Qwen2-VL model for better vision reasoning
trainer = Trainer(model="Qwen/Qwen2-VL-7B-Instruct")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=2,
    rollout_gpus=2,
    epochs=1
)