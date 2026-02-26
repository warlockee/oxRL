from oxrl import Trainer

trainer = Trainer(model="mistralai/Mistral-Nemo-Instruct-2407")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1
)