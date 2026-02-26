from oxrl import Trainer

trainer = Trainer(model="HuggingFaceH4/zephyr-7b-beta")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1
)