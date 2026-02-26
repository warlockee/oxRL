from oxrl import Trainer

trainer = Trainer(model="meta-llama/Llama-3.2-3B-Instruct")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1
)