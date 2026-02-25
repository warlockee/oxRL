from oxrl import Trainer

# Qwen2-VL is a powerful vision-language model
trainer = Trainer(model="Qwen/Qwen2-VL-2B-Instruct")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1
)