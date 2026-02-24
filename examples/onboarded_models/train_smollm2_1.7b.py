from oxrl import Trainer

trainer = Trainer(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=1,
    rollout_gpus=1,
    epochs=1
)
