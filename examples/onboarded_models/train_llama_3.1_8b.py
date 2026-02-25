from oxrl import Trainer

trainer = Trainer(model="meta-llama/Meta-Llama-3.1-8B-Instruct")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1
)