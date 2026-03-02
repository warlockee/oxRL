"""
Backward-compatible re-export hub.

All schema classes, sync logic, and the loader are now in separate modules:
    oxrl.configs.schema  — Pydantic models (Run, Train, Model, Data, ...)
    oxrl.configs.sync    — sync_deepspeed_config() and helpers
    oxrl.configs.loader  — load_and_verify()

Existing code can continue to do:
    import oxrl.configs.load as cfg
    config = cfg.load_and_verify(...)
    from oxrl.configs.load import Train
"""

# Schema classes
from oxrl.configs.schema import (  # noqa: F401
    Run,
    Train,
    Data,
    Model,
    DeepSpeed,
    DeepSpeedRef,
    InferenceEngine,
    Lora,
    Reward,
    Rollout,
    Config,
)

# Loader
from oxrl.configs.loader import load_and_verify  # noqa: F401

# Sync (exposed for direct use if needed)
from oxrl.configs.sync import sync_deepspeed_config  # noqa: F401
