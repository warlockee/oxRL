from oxrl import Trainer

trainer = Trainer(model="microsoft/Phi-3.5-mini-instruct")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1
)