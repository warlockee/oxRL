from oxrl import Trainer

trainer = Trainer(model="mistralai/Mistral-7B-Instruct-v0.3")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=2,
    rollout_gpus=2,
    epochs=1
)
