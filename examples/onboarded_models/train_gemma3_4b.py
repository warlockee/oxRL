from oxrl import Trainer

# Gemma-3 is natively multimodal (Vision/Audio/Text)
trainer = Trainer(model="google/gemma-3-4b-it")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1
)