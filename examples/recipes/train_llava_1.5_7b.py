from oxrl import Trainer

# Classic LLaVA model for vision-language tasks
trainer = Trainer(model="llava-hf/llava-1.5-7b-hf")
trainer.train(
    train_file="examples/data/train.jsonl",
    training_gpus=2,
    rollout_gpus=2,
    epochs=1
)