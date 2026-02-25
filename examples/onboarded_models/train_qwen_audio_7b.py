from oxrl import Trainer

# Qwen2-Audio for audio-to-text and audio reasoning tasks
trainer = Trainer(model="Qwen/Qwen2-Audio-7B-Instruct")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=2,
    rollout_gpus=2,
    epochs=1
)