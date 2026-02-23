# oxRL Design Document

## Executive Summary

oxRL is a lightweight, modular, production-grade framework for post-training/fine-tuning Large Language Models (LLMs), Vision Language Models (VLMs), and Vision Language Actions (VLAs). The framework prioritizes clarity and debuggability over raw throughput, making it ideal for research teams needing understandable, hackable RL code.

**Key Statistics:**
- Total Python files: 24
- Total lines of code: ~4,341
- Main entry points: 2 (`main_rl.py`, `main_sl.py`)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Breakdown](#component-breakdown)
3. [Data Flow](#data-flow)
4. [Strengths](#strengths)
5. [Weaknesses & Issues](#weaknesses--issues)
6. [Potential Problems](#potential-problems)
7. [Next Steps & Recommendations](#next-steps--recommendations)

---

## Architecture Overview

### Directory Structure

```
oxRL/
├── algs/                    # Algorithm implementations
│   ├── PPO/ppo.py          # Proximal Policy Optimization (460 lines)
│   ├── SGRPO/sgrpo.py      # Stable GRPO variant (436 lines)
│   ├── CISPO/cispo.py      # Clipped IS Policy Optimization (437 lines)
│   ├── SFT/sft.py          # Supervised Fine-Tuning (142 lines)
│   └── DPO/dpo.py          # Direct Preference Optimization (STUB - 1 line)
├── configs/                 # Pydantic-based configuration
│   ├── load.py             # Config management (393 lines)
│   ├── rl_args.yaml        # RL training config
│   └── sl_args.yaml        # SL training config
├── custom_datasets/         # PyTorch dataset implementations
│   ├── prompt_only.py      # Prompts for RL (151 lines)
│   ├── prompt_response.py  # Prompt-response pairs for SL (231 lines)
│   └── mixed_ratio_sampler.py  # Multi-dataset sampling (132 lines)
├── misc/                    # Utilities
│   ├── logging.py          # Rank-aware logging + MLflow (77 lines)
│   └── utils.py            # Helper functions (64 lines)
├── preprocessing/           # Data preprocessing
│   └── gsm8k.py            # GSM8K dataset prep (95 lines)
├── rewards/                 # Reward computation
│   └── compute_score.py    # Reward functions (22 lines)
├── rollouts/                # Rollout generation
│   ├── vllm_engine.py      # vLLM-based inference (499 lines)
│   └── replay_buffer.py    # On-policy sample storage (207 lines)
├── main_rl.py              # Main RL training loop (582 lines)
├── main_sl.py              # Main SL training loop (413 lines)
├── README.md               # Project overview
├── FAQ.md                  # Frequently asked questions
└── req.txt                 # Minimal dependencies
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Core ML | PyTorch | Tensor computation |
| Distributed Training | DeepSpeed (ZeRO-3) | Memory-efficient training |
| Distributed Orchestration | Ray | Actor-based parallelism |
| Fast Inference | vLLM | High-throughput rollout generation |
| Model Loading | HuggingFace Transformers | Standardized model/tokenizer handling |
| Configuration | Pydantic | Type-safe config validation |
| Experiment Tracking | MLflow | Metrics and artifact logging |

---

## Component Breakdown

### 1. Algorithm Implementations (`/algs`)

#### PPO (Proximal Policy Optimization)
- **File:** `algs/PPO/ppo.py` (460 lines)
- **Type:** Ray remote actor
- **Key Methods:**
  - `compute_advantages()`: GAE-based advantage calculation with masking
  - `compute_policy_loss()`: PPO clipping + entropy + KL regularization
  - `compute_value_loss()`: Value function with clipping
- **Features:** NaN detection, padding hole detection, terminal state validation

#### SGRPO (Stable GRPO)
- **File:** `algs/SGRPO/sgrpo.py` (436 lines)
- **Type:** Ray remote actor with DeepSpeed
- **Key Methods:**
  - `init_training_engine()`: Per-actor DeepSpeed initialization
  - `ref_forward()`: Reference model log-probabilities
  - `compute_policy_loss()`: Policy loss with KL from reference
- **Features:** Reference model support, per-GPU DeepSpeed engine

#### CISPO (Clipped IS Policy Optimization)
- **File:** `algs/CISPO/cispo.py` (437 lines)
- **Type:** Ray remote actor with DeepSpeed
- **Key Difference:** Weighted policy gradient with importance sampling ratio clipping

#### SFT (Supervised Fine-Tuning)
- **File:** `algs/SFT/sft.py` (142 lines)
- **Type:** Standard class (single-GPU compatible)
- **Key Methods:** `compute_loss()`, `forward()`, `eval_step()`

#### DPO (Direct Preference Optimization)
- **File:** `algs/DPO/dpo.py` (1 line)
- **Status:** NOT IMPLEMENTED (stub only)

### 2. Rollout Generation (`/rollouts`)

#### VLLMRolloutEngine
- **File:** `rollouts/vllm_engine.py` (499 lines)
- **Type:** Ray remote actor
- **Key Methods:**
  - `refresh_model()`: Hot-swap policy during training
  - `generate()`: Main rollout generation
  - `extract_logprobs()`: Per-token log probability extraction
  - `score_response()`: Apply reward function
- **Features:** Tensor parallelism, CUDA cleanup, strict on-policy enforcement

#### ReplayBuffer
- **File:** `rollouts/replay_buffer.py` (207 lines)
- **Type:** PyTorch Dataset subclass
- **Stores:** input_ids, rewards, advantages, old logprobs, masks, dones
- **Features:** Batch sequence addition, automatic truncation

### 3. Dataset Handling (`/custom_datasets`)

| Class | Purpose | Key Features |
|-------|---------|--------------|
| `PromptOnlyDataset` | RL prompts | Variable-length tokenization, chat templates |
| `PromptResponseDataset` | SL pairs | Auto-regressive targets, packing support |
| `MixedRatioSampler` | Multi-dataset | Custom ratios, distributed RNG |

### 4. Configuration System (`/configs`)

Pydantic-based hierarchy:
```
Config (main)
├── Run               # Experiment metadata, distributed setup
├── Train             # Optimizer, algorithm, training loop params
├── Data              # Dataset paths, sampling ratios
├── Model             # Model name, dtype, attention implementation
├── DeepSpeed         # ZeRO optimization, activation checkpointing
├── DeepSpeedRef      # Reference model inference config
├── InferenceEngine   # vLLM settings
├── Reward            # Reward function, normalization
└── Rollout           # Generation params, sampling, tensor parallelism
```

---

## Data Flow

### RL Training Loop (`main_rl.py`)

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Setup Phase                                │
├─────────────────────────────────────────────────────────────────────┤
│  1. Initialize rank/device + random seeds                           │
│  2. Start Ray cluster                                               │
│  3. Load & validate configuration                                   │
│  4. Setup MLflow experiment tracking                                │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Engine Initialization                           │
├─────────────────────────────────────────────────────────────────────┤
│  • Create N training engines (Ray actors: SGRPO/CISPO)              │
│  • Create M rollout engines (Ray actors: vLLM)                      │
│  • Load tokenizer + reward function                                 │
│  • Initialize replay buffer                                          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Training Loop (per epoch)                         │
├──────────────────────────┬──────────────────────────────────────────┤
│   Rollout Phase          │   Training Phase                         │
├──────────────────────────┼──────────────────────────────────────────┤
│  1. Load prompts         │  1. Create train batches from buffer     │
│  2. Distribute to M      │  2. Pad for distributed consistency      │
│     rollout engines      │  3. Distribute to N training engines     │
│  3. Generate completions │  4. Forward + backward pass              │
│     (vLLM)               │  5. Aggregate metrics                    │
│  4. Extract logprobs     │                                          │
│  5. Compute rewards      ├──────────────────────────────────────────┤
│  6. Store in buffer      │   Checkpoint Phase                       │
│                          ├──────────────────────────────────────────┤
│                          │  • Collective save (all ranks)           │
│                          │  • Save tokenizer                        │
│                          │  • Sync filesystem                       │
│                          ├──────────────────────────────────────────┤
│                          │   Model Refresh                          │
│                          ├──────────────────────────────────────────┤
│                          │  • Update rollout engines with new       │
│                          │    policy version                        │
└──────────────────────────┴──────────────────────────────────────────┘
```

### SL Training Loop (`main_sl.py`)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Simpler Setup                               │
├─────────────────────────────────────────────────────────────────────┤
│  • torch.distributed (no Ray)                                       │
│  • DistributedSampler for data sharding                             │
│  • Single DeepSpeed engine per rank                                 │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Per Epoch                                       │
├─────────────────────────┬───────────────────────────────────────────┤
│   Training              │   Validation                              │
├─────────────────────────┼───────────────────────────────────────────┤
│  1. MixedRatioSampler   │  1. Evaluate on val data                  │
│  2. Forward + loss      │  2. All-reduce loss averaging             │
│  3. Backward + step     │  3. Save checkpoint                       │
│  4. Gradient accum      │                                           │
└─────────────────────────┴───────────────────────────────────────────┘
```

---

## Strengths

### 1. Clean Architecture
- **Clear separation of concerns:** Algorithms, rollouts, and training loops are independent
- **Explicit interfaces:** Ray actor boundaries make component interactions obvious
- **Modular design:** Easy to swap algorithms or modify individual components

### 2. Production-Ready Infrastructure
- **DeepSpeed ZeRO-3:** Enables training of billion-parameter models
- **Activation checkpointing:** Trades compute for memory
- **Distributed checkpointing:** Collective save/load across ranks
- **MLflow integration:** Experiment tracking without external servers

### 3. Type Safety & Validation
- **Pydantic configuration:** Catches invalid configs at load time
- **Explicit type hints:** Throughout function signatures
- **Runtime assertions:** Shape/value validation in critical paths

### 4. Comprehensive Error Checking
- **NaN detection:** In PPO advantage calculation
- **Padding hole detection:** Validates mask consistency
- **Terminal state validation:** Correct EOS vs truncation handling
- **Distributed training guards:** Barrier synchronization, rank checks

### 5. Fast Inference
- **vLLM integration:** State-of-the-art inference throughput
- **Tensor parallelism:** Scale inference across GPUs
- **Hot model refresh:** Update policy without restarting inference engines

### 6. Research-Friendly Design
- **Explicit tensor shapes:** Documented in comments
- **Prediction-aligned representation:** Avoids autoregressive index confusion
- **Debuggable:** Prioritizes correctness over throughput

---

## Weaknesses & Issues

### Critical Issues

| Issue | Location | Impact |
|-------|----------|--------|
| **DPO not implemented** | `algs/DPO/dpo.py` | Users may expect working DPO |
| **No test suite** | Entire codebase | No confidence in correctness |
| **Preprocessing only for GSM8K** | `preprocessing/` | Limited dataset support |

### Code Quality Issues

#### 1. Incomplete Gradient Checkpointing
```python
# Config has gradient_checkpointing but SGRPO/CISPO don't use it
model:
  gradient_checkpointing: True  # May not actually be enabled
```

#### 2. Inconsistent Logging
```python
# Some code uses print() instead of logger
print(f"Error deleting vllm_engine: {e}")  # Bypasses rank filtering
```

#### 3. Exception Swallowing
```python
# vllm_engine.py line ~108
except Exception as e:
    print(f"Error deleting vllm_engine: {e}")
    pass  # Silently continues - could mask real issues
```

#### 4. Inefficient Advantage Calculation
```python
# PPO: Sequential loop instead of vectorized
for t in reversed(range(T)):
    delta = rewards[:, t] + ...
    advantages[:, t] = delta + ...
# Could be vectorized for performance
```

#### 5. Multiple Device Transfers
```python
# Multiple .to(device) calls in hot paths
rewards = rewards.to(dtype=dtype, device=device)
values = values.to(dtype=dtype, device=device)
# Could batch transfers
```

### Documentation Gaps

| Missing | Impact |
|---------|--------|
| Installation instructions | Users can't set up the project |
| Algorithm explanations | PPO vs SGRPO vs CISPO differences unclear |
| Performance benchmarks | No baseline for comparison |
| API reference | Difficult to use as a library |
| Troubleshooting guide | Common errors undocumented |

---

## Potential Problems

### 1. Distributed Training Complexity

**Risk:** Ray + DeepSpeed configuration mismatches cause cryptic hangs

```yaml
# Must match exactly:
run:
  training_gpus: 2    # Must match DeepSpeed world size
deepspeed:
  zero_optimization:
    stage: 3          # Requires collective operations
```

**Mitigation:** Add config validation to check Ray actor count matches DeepSpeed world size

### 2. No Rollout-Training Overlap

**Risk:** GPU utilization is suboptimal

```
Current:  [Rollout Phase] → [Training Phase] → [Rollout Phase] → ...
Optimal:  [Rollout ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓]
          [Training        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓]
```

**Note:** This is acknowledged in FAQ as intentional trade-off for debuggability

### 3. Memory Pressure

**Risk:** Large replay buffers + model copies exhaust GPU memory

**Locations:**
- ReplayBuffer stores full sequences
- Reference model in SGRPO/CISPO
- vLLM KV cache during generation

**Mitigation:** Currently handled via ZeRO offloading, but no dynamic memory management

### 4. Reproducibility Concerns

**Risk:** Complex distributed setup makes exact reproduction difficult

**Factors:**
- Ray actor scheduling non-determinism
- DeepSpeed gradient synchronization ordering
- vLLM sampling randomness

**Current:** Seeds are set, but distributed ordering varies

### 5. Error Recovery

**Risk:** No retry logic for failures

**Scenarios:**
- Failed vLLM generations → crash
- Ray actor death → crash
- DeepSpeed rank hang → indefinite wait

**Mitigation needed:** Retry logic, timeouts, graceful degradation

### 6. Tensor Alignment Complexity

**Risk:** Prediction-aligned representation could break with non-autoregressive models

```python
# Token-aligned: token[t] at position t
# Prediction-aligned: logit[t] predicts token[t+1]
# Requires careful index management throughout
```

---

## Next Steps & Recommendations

### Priority 1: Critical (Blocks Usage)

#### 1.1 Add Test Suite
```
tests/
├── unit/
│   ├── test_ppo.py              # Advantage calculation, loss functions
│   ├── test_sgrpo.py            # Policy loss, KL divergence
│   ├── test_replay_buffer.py    # Add/retrieve sequences
│   └── test_config.py           # Validation edge cases
├── integration/
│   ├── test_distributed.py      # Multi-GPU training
│   ├── test_rollout.py          # vLLM generation
│   └── test_end_to_end.py       # Full training loop
└── fixtures/
    ├── tiny_model.py            # Small model for testing
    └── dummy_datasets.py        # Synthetic data
```

#### 1.2 Complete DPO Implementation
- Implement `algs/DPO/dpo.py`
- Add DPO-specific config options
- Test with preference datasets

#### 1.3 Add Installation Documentation
```markdown
## Installation

### Requirements
- Python 3.10+
- CUDA 12.0+
- 2+ GPUs (for distributed training)

### Steps
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install flash-attention: `pip install flash-attn --no-build-isolation`
4. Configure YAML
5. Run training
```

### Priority 2: Important (Improves Quality)

#### 2.1 Fix Gradient Checkpointing
```python
# In SGRPO/CISPO init_training_engine():
if self.cfg.model.gradient_checkpointing:
    self.model.gradient_checkpointing_enable()
```

#### 2.2 Vectorize Advantage Calculation
```python
# Replace loop with vectorized computation
def compute_advantages_vectorized(rewards, values, dones, gamma, gae_lambda):
    T = rewards.shape[1]
    advantages = torch.zeros_like(rewards)

    # Vectorized GAE computation
    deltas = rewards + gamma * values[:, 1:] * (1 - dones[:, :-1]) - values[:, :-1]
    # ... vectorized accumulation
```

#### 2.3 Add Retry Logic
```python
# In vllm_engine.py generate():
@retry(max_attempts=3, backoff=exponential)
def generate(self, prompts, ...):
    ...
```

#### 2.4 Standardize Logging
```python
# Replace all print() with logger
from misc.logging import get_logger
logger = get_logger(__name__)

# Instead of: print(f"Error: {e}")
logger.error(f"Error: {e}")
```

### Priority 3: Nice to Have (Future Improvements)

#### 3.1 Rollout-Training Pipelining
- Implement async rollout generation
- Double-buffer replay data
- Would improve GPU utilization significantly

#### 3.2 Additional Dataset Preprocessing
```
preprocessing/
├── gsm8k.py       # Existing
├── math.py        # MATH dataset
├── code.py        # Code generation datasets
├── general.py     # Generic prompt-response
└── preference.py  # DPO/RLHF preference data
```

#### 3.3 Performance Benchmarks
- Training throughput (tokens/sec)
- Memory usage vs model size
- Scaling efficiency (1 GPU → N GPUs)
- Comparison with other frameworks

#### 3.4 Algorithm Documentation
```markdown
## Algorithm Comparison

| Algorithm | Use Case | Key Difference |
|-----------|----------|----------------|
| PPO | Standard RL | Value function + clipped policy |
| SGRPO | Stability-focused | Reference KL + no value function |
| CISPO | IS-weighted | Importance sampling with clipping |
| DPO | Preference learning | Direct optimization, no RL |
```

#### 3.5 Configuration Validation
```python
# Add cross-field validation
@validator('training_gpus')
def validate_gpu_config(cls, v, values):
    if v != values.get('deepspeed', {}).get('world_size'):
        raise ValueError("training_gpus must match DeepSpeed world_size")
    return v
```

---

## Summary

### What oxRL Does Well
- Clean, modular architecture
- Production-grade distributed training
- Type-safe configuration
- Comprehensive error checking
- Research-friendly design

### What Needs Improvement
- No test coverage
- Incomplete implementations (DPO)
- Documentation gaps
- Sequential training (by design)
- Error recovery mechanisms

### Recommended Action Items

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Add pytest test suite | Medium |
| P0 | Complete DPO implementation | Medium |
| P0 | Write installation docs | Low |
| P1 | Fix gradient checkpointing | Low |
| P1 | Vectorize advantage calculation | Medium |
| P1 | Add retry logic | Low |
| P1 | Standardize logging | Low |
| P2 | Implement pipelining | High |
| P2 | Add more preprocessing | Medium |
| P2 | Create benchmarks | Medium |
| P2 | Write algorithm docs | Medium |

---

*Document generated: 2026-01-20*
*Codebase version: main branch (commit 2e60c8c)*
