from oxrl import Trainer

# Llama 3.2 Vision model for high-quality image understanding
trainer = Trainer(model="meta-llama/Llama-3.2-11B-Vision-Instruct")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=4,
    rollout_gpus=4,
    epochs=1
)