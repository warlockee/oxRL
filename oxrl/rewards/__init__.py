# Reward functions — all share signature: (prompt_ids, response_ids, finish_reason, metadata) -> (rewards, is_per_token)
#
# Binary (1/0):  default, gsm8k, math, code, mcqa — clear right/wrong signal, fast convergence.
# Graduated:     soft_math (1.0/0.5/0.2), format (0.25 steps), reasoning (tags + correctness),
#                multimodal (correctness + modality) — denser signal, helps when binary is too sparse.
# Continuous:    rm_reward — learned reward model. Use for RLHF when no verifiable ground truth.
#
# Pick binary when you have verifiable answers. Pick graduated when binary reward is too sparse
# (model rarely gets full marks). Pick rm_reward when correctness can't be checked programmatically.

from oxrl.rewards.base import (
    default_reward_func,
    extract_answer,
    extract_math_answer,
    extract_mcqa_answer
)
from oxrl.rewards.math import (
    gsm8k_reward_func,
    math_reward_func,
    soft_math_reward_func
)
from oxrl.rewards.code import code_reward_func
from oxrl.rewards.format import format_reward_func
from oxrl.rewards.qa import mcqa_reward_func
from oxrl.rewards.reasoning import reasoning_reward_func
from oxrl.rewards.multimodal import multimodal_reward_func
from oxrl.rewards.rm_reward import rm_reward_func, load_reward_model

__all__ = [
    "default_reward_func",
    "gsm8k_reward_func",
    "math_reward_func",
    "soft_math_reward_func",
    "code_reward_func",
    "format_reward_func",
    "mcqa_reward_func",
    "reasoning_reward_func",
    "multimodal_reward_func",
    "rm_reward_func",
    "load_reward_model",
    "extract_answer",
    "extract_math_answer",
    "extract_mcqa_answer",
]
