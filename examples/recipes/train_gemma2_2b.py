from oxrl import Trainer

trainer = Trainer(model="google/gemma-2-2b-it")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1
)