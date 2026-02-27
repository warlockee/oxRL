import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class RewardValueHead(nn.Module):
    '''
       Linear value head that maps hidden states to scalar rewards.
       [B, T, H] -> [B, T]
    '''
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, T, H] -> [B, T, 1] -> [B, T]
        return self.linear(hidden_states).squeeze(-1)

class RM:
    '''
       Reward Model Training with Bradley-Terry preference loss.
       Loss: L = -log(sigmoid(r_chosen - r_rejected))

       Uses PromptPreferenceDataset (same batch format as DPO:
       chosen_input_ids, chosen_attn_mask, rejected_input_ids, rejected_attn_mask).
    '''
    def __init__(self,
                model_engine,
                optimizer,
                use_cache=False):

        self.model_engine = model_engine
        self.optimizer = optimizer
        self.use_cache = use_cache

        hidden_size = model_engine.module.config.hidden_size
        self.value_head = RewardValueHead(hidden_size).to(model_engine.device)
        self.value_head.to(dtype=next(model_engine.parameters()).dtype)
        self.value_head_optimizer = torch.optim.AdamW(self.value_head.parameters(), lr=1e-5)

    def forward(self, input_ids, attn_mask):
        '''
           Forward pass: runs model with output_hidden_states=True,
           extracts last hidden state, passes through value head,
           and returns the reward at the last non-padding position.

           input_ids: [B, T]
           attn_mask: [B, T]
           Returns: final_rewards [B]
        '''
        token_type_ids = torch.zeros_like(input_ids)

        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=attn_mask,
                                   token_type_ids=token_type_ids,
                                   output_hidden_states=True,
                                   use_cache=self.use_cache)

        # Last layer hidden states: [B, T, H]
        hidden_states = output.hidden_states[-1]

        # Value head: [B, T, H] -> [B, T]
        rewards = self.value_head(hidden_states)

        # Get reward at last non-padding position
        seq_lengths = attn_mask.sum(dim=-1) - 1  # [B]
        batch_indices = torch.arange(rewards.size(0), device=rewards.device)
        final_rewards = rewards[batch_indices, seq_lengths]  # [B]

        return final_rewards

    def compute_loss(self, r_chosen, r_rejected):
        '''
           Bradley-Terry preference loss.
           r_chosen: [B]
           r_rejected: [B]
           Returns: (loss, accuracy, margin)
        '''
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        with torch.no_grad():
            accuracy = (r_chosen > r_rejected).float().mean()
            margin = (r_chosen - r_rejected).mean()

        return loss, accuracy, margin

    def train_step(self, micro_batch):
        '''
           One training step for RM.
           micro_batch contains: chosen_input_ids, chosen_attn_mask,
                                 rejected_input_ids, rejected_attn_mask
        '''
        self.model_engine.train()
        self.value_head.train()

        # Forward chosen
        r_chosen = self.forward(micro_batch['chosen_input_ids'],
                                micro_batch['chosen_attn_mask'])

        # Forward rejected
        r_rejected = self.forward(micro_batch['rejected_input_ids'],
                                  micro_batch['rejected_attn_mask'])

        # Compute loss
        loss, accuracy, margin = self.compute_loss(r_chosen, r_rejected)

        # Backward and step
        self.model_engine.backward(loss)
        self.model_engine.step()

        self.value_head_optimizer.step()
        self.value_head_optimizer.zero_grad()

        return {"loss": loss.item(), "accuracy": accuracy.item(), "margin": margin.item()}

    def eval_step(self, micro_batch):
        '''
           Validation step.
        '''
        self.model_engine.eval()
        self.value_head.eval()

        with torch.no_grad():
            r_chosen = self.forward(micro_batch['chosen_input_ids'],
                                    micro_batch['chosen_attn_mask'])

            r_rejected = self.forward(micro_batch['rejected_input_ids'],
                                      micro_batch['rejected_attn_mask'])

            loss, accuracy, margin = self.compute_loss(r_chosen, r_rejected)

        return {"loss": loss.item(), "accuracy": accuracy.item(), "margin": margin.item()}
