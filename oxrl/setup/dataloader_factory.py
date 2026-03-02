"""
Rollout dataloader setup for RL training.
"""
from torch.utils.data import DataLoader
from oxrl.datasets.prompt_only import PromptOnlyDataset


def rollout_dataloader_setup(params, tokenizer, num_rollout_engines):
    """Create a DataLoader that feeds prompts to rollout engines."""
    prompt_ds = PromptOnlyDataset(
        prompt_key=params.data.prompt_key,
        max_seq_len=params.data.max_seq_len,
        tokenizer=tokenizer,
        data_path=params.data.train_files_path,
        return_text=False,
        answer_key=params.data.answer_key,
        model_name=params.model.name,
    )

    bsz = num_rollout_engines * params.rollout.rollout_batch_size_per_gpu
    dataloader = DataLoader(
        dataset=prompt_ds,
        batch_size=bsz,
        num_workers=params.data.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=prompt_ds.collate_fn,
    )

    return dataloader
