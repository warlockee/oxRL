from oxrl import Trainer

# 1. Initialize with your model
trainer = Trainer(model="Qwen/Qwen2.5-0.5B-Instruct")

# 2. Start training (auto-detects hardware, auto-preps data, auto-configures RL)
trainer.train(task="math", epochs=1, steps_per_epoch=2)
