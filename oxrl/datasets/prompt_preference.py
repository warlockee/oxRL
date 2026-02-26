import torch
from torch.utils.data import Dataset
import os
from datasets import load_dataset
from typing import Dict, List, Any

class PromptPreferenceDataset(Dataset):
    '''
        This dataset handles (prompt, chosen, rejected) triplets for preference optimization (DPO, ORPO, etc.).
        The data should be in a parquet format.
    '''
    def __init__(self, 
                prompt_key: str,
                chosen_key: str,
                rejected_key: str,
                max_seq_len: int,
                tokenizer=None, 
                data_path: str = "",
                ):
        assert prompt_key != "", "prompt_key cannot be empty"
        assert chosen_key != "", "chosen_key cannot be empty"
        assert rejected_key != "", "rejected_key cannot be empty"
        assert max_seq_len > 0, "max_seq_len must be greater than 0"
        assert tokenizer is not None, "tokenizer cannot be None"
        assert os.path.exists(os.path.expanduser(data_path)), f"{data_path} does not exist"

        assert tokenizer.pad_token_id is not None, "tokenizer must have a pad token"
        assert tokenizer.eos_token_id is not None, "tokenizer must have an eos token"

        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        try:
            self.data = load_dataset("parquet", data_files=self.data_path, split="train")
        except Exception as e:
            raise Exception(f"Failed to load data from {self.data_path}: {str(e)}")
        self.len_data = len(self.data)

    def _tokenize_sample(self, prompt: List[Dict[str, str]], response: str) -> Dict[str, torch.Tensor]:
        # 1. Tokenize prompt
        prompt_ids = self.tokenizer.apply_chat_template(
            conversation=prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors='pt'
        )[0]
        prompt_len = len(prompt_ids)

        # 2. Tokenize response + EOS
        resp_output = self.tokenizer(response, return_tensors='pt', add_special_tokens=False)
        eos_tensor = torch.tensor([self.tokenizer.eos_token_id], dtype=resp_output['input_ids'].dtype)
        resp_ids = torch.cat([resp_output['input_ids'][0], eos_tensor], dim=0)
        resp_attn_mask = torch.cat([resp_output['attention_mask'][0], torch.tensor([1], dtype=resp_output['attention_mask'].dtype)], dim=0)

        seq_ids = torch.cat((prompt_ids, resp_ids), dim=-1).to(dtype=torch.long)
        seq_attn_mask = torch.cat((torch.ones_like(prompt_ids), resp_attn_mask), dim=-1)
        total_len = len(seq_ids)

        # Truncate if necessary
        if total_len > self.max_seq_len:
            seq_ids = seq_ids[:self.max_seq_len]
            seq_attn_mask = seq_attn_mask[:self.max_seq_len]
        elif total_len < self.max_seq_len:
            padding_len = self.max_seq_len - total_len
            seq_ids = torch.cat((seq_ids, torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=seq_ids.dtype)), dim=-1)
            seq_attn_mask = torch.cat((seq_attn_mask, torch.zeros(padding_len, dtype=seq_attn_mask.dtype)), dim=-1)

        # Loss mask (1 for response tokens, 0 for prompt/padding)
        loss_mask = seq_attn_mask[1:].clone()
        if prompt_len > 1:
            loss_mask[:prompt_len - 1] = 0

        return {
            "input_ids": seq_ids,
            "attn_mask": seq_attn_mask,
            "loss_mask": loss_mask,
        }

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample[self.prompt_key]
        chosen = sample[self.chosen_key]
        rejected = sample[self.rejected_key]

        chosen_data = self._tokenize_sample(prompt, chosen)
        rejected_data = self._tokenize_sample(prompt, rejected)

        return {
            "chosen_input_ids": chosen_data["input_ids"],
            "chosen_attn_mask": chosen_data["attn_mask"],
            "chosen_loss_mask": chosen_data["loss_mask"],
            "rejected_input_ids": rejected_data["input_ids"],
            "rejected_attn_mask": rejected_data["attn_mask"],
            "rejected_loss_mask": rejected_data["loss_mask"],
        }

    def __len__(self):
        return self.len_data
