# Post-Training Framework Comparison

A factual comparison of four open-source LLM post-training frameworks, as of March 2026.

---

## At a Glance

| Dimension | **TRL** | **OpenRLHF** | **veRL** | **oxRL** |
|---|---|---|---|---|
| **Focus** | Full-stack post-training (HF ecosystem) | Scalable online RL (Ray + vLLM) | High-performance RL research (FSDP/Megatron) | Lightweight post-training, algorithm breadth |
| **GitHub Stars** | ~17.5k | ~9.1k | ~19.6k | ~16 |
| **Contributors** | ~471 | ~80+ | ~100+ | 1 |
| **First Release** | 2022 | 2023 | 2024 | 2025 |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |

---

## Algorithms

| Category | **TRL** | **OpenRLHF** | **veRL** | **oxRL** |
|---|---|---|---|---|
| **Total algorithms** | ~16 | ~6 | ~10+ | 51 |
| **Online RL** | GRPO, RLOO, PPO, OnlineDPO, NashMD, XPO | PPO, REINFORCE++, REINFORCE++-baseline, RLOO, GRPO, Dr. GRPO | PPO, GRPO, REINFORCE++, RLOO, DAPO, DrGRPO, GSPO, PF-PPO, VAPO | SGRPO, GSPO, CISPO, PPO, RLHF, RLAIF |
| **Offline preference** | DPO, CPO, BCO, KTO, ORPO | DPO (via SFT pipeline) | DPO (limited focus) | DPO, IPO, SimPO, CPO, KTO, ORPO + 35 variants (see below) |
| **SFT / Other** | SFT, Reward, PRM, GKD, MiniLLM | SFT | SFT | SFT, CPT, KD, RM, RFT, SPIN, OnlineDPO |
| **DPO variant depth** | ~6 loss types within DPOTrainer (sigmoid, hinge, IPO, EXO, NCA, Robust, BCO, SPPO, AOT, APO, DiscoPOP) | Minimal | Minimal | 35+ standalone implementations: RDPO, cDPO, BetaDPO, CalDPO, DPOP, FocalPO, GPO, WPO, fDPO, HDPO, DPOShift, CPOSimPO, DrDPO, ChiPO, SPO, DPNLL, MinorDPO, C2DPO, AlphaDPO, BPO, SamPO, etc. |

**Note on TRL's algorithm count:** TRL implements several DPO variants as `loss_type` flags within a single `DPOTrainer` class (e.g., `loss_type="hinge"`, `loss_type="ipo"`). This is a valid engineering approach. oxRL implements each variant as a standalone class with its own file and test suite.

---

## Model Architecture Support

| Dimension | **TRL** | **OpenRLHF** | **veRL** | **oxRL** |
|---|---|---|---|---|
| **Base models** | Any HuggingFace causal LM | Any HuggingFace causal LM | HuggingFace models (Qwen, Llama, Gemma, DeepSeek) | Any HuggingFace `AutoModelForCausalLM` |
| **MoE models** | Supported via HF | Supported (70B+) | Supported up to 671B (DeepSeek-V3) | Supported (Qwen3-MoE, GSPO for routing noise) |
| **Vision-Language** | Supported (Qwen2-VL, etc.) | Supported (via OpenRLHF-M) | Supported (Qwen2.5-VL, Kimi-VL) | Supported (Qwen2-VL, Qwen3-VL, SmolVLM, Kimi-VL) |
| **Audio** | Limited | Not documented | Not documented | Supported (Qwen2-Audio) |
| **Verified model count** | Not tracked | Not tracked | Not tracked | 38 models explicitly verified via automated pipeline |
| **Max tested scale** | 70B+ | 70B+ | 671B | 35B (Qwen3.5-35B-A3B) |

---

## Multi-GPU / Distributed Training

| Dimension | **TRL** | **OpenRLHF** | **veRL** | **oxRL** |
|---|---|---|---|---|
| **DeepSpeed** | Yes (ZeRO 1/2/3 via Accelerate) | Yes (ZeRO-3, DeepCompile, AutoTP, RingAttention) | Yes (Ulysses sequence parallelism) | Yes (ZeRO-3, primary strategy) |
| **FSDP** | Yes (via Accelerate) | No | Yes (FSDP/FSDP2, primary strategy) | No |
| **Megatron-LM** | No | No | Yes | No |
| **Ray** | No | Yes (primary scheduler) | Yes (optional) | Yes (for RL actor scheduling) |
| **Tensor Parallelism** | Via vLLM | DeepSpeed AutoTP, vLLM AutoTP | Via vLLM/Megatron | Via vLLM |
| **Pipeline Parallelism** | No | Via vLLM | Via Megatron | No |

---

## Inference Engine

| Dimension | **TRL** | **OpenRLHF** | **veRL** | **oxRL** |
|---|---|---|---|---|
| **vLLM** | Yes (GRPO, RLOO, OnlineDPO, NashMD, XPO) | Yes (primary, with AutoTP + Pipeline Parallelism) | Yes (>=0.8.2) | Yes (rollout generation) |
| **SGLang** | No | No | Yes | No |
| **Native HF generate** | Yes (fallback for all trainers) | No | Yes (fallback) | No |
| **Co-located vs disaggregated** | Co-located (vLLM runs in-process) | Disaggregated (separate vLLM workers via Ray) | Hybrid (HybridEngine: same GPUs, or disaggregated) | Co-located (vLLM rollouts on same GPU set via Ray) |

---

## Ease of Use

| Dimension | **TRL** | **OpenRLHF** | **veRL** | **oxRL** |
|---|---|---|---|---|
| **Lines for basic DPO** | ~5 (Python) | ~20-30 (CLI with flags) | ~15-20 (config + script) | ~3 (Python API) or ~20 (YAML config + CLI) |
| **Configuration style** | Python kwargs / TrainingArguments | CLI arguments + YAML | YAML / Python config | YAML (Pydantic-validated) |
| **Dataset handling** | Auto-formats from HF datasets | Manual preparation | Manual preparation | Auto-detection with HF datasets |
| **Error messages** | Good (HF ecosystem maturity) | Moderate | Moderate | Strict Pydantic validation (`extra='forbid'` catches typos) |
| **Quick start** | `pip install trl` + 5 lines | Docker recommended | `pip install verl` + config | `pip install oxrl` + 3 lines |

### Minimal DPO Example Comparison

**TRL:**
```python
from trl import DPOTrainer
from datasets import load_dataset
trainer = DPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
)
trainer.train()
```

**oxRL:**
```python
from oxrl import Trainer
trainer = Trainer(model="Qwen/Qwen3-0.6B")
trainer.train(task="dpo")
```

**OpenRLHF:**
```bash
deepspeed --module openrlhf.cli.train_dpo \
   --pretrain Qwen/Qwen3-0.6B \
   --dataset Open-Orca/OpenOrca \
   --beta 0.1 \
   --max_len 2048 \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --learning_rate 5e-7 \
   ...
```

---

## Documentation

| Dimension | **TRL** | **OpenRLHF** | **veRL** | **oxRL** |
|---|---|---|---|---|
| **Hosted docs site** | Yes (huggingface.co/docs/trl) | README + GitHub Wiki | Yes (readthedocs) | README only |
| **API reference** | Comprehensive, auto-generated | Minimal | Moderate | Inline docstrings, no hosted docs |
| **Tutorials / Guides** | Extensive (smol-course, blog posts, conceptual guides) | Docker setup, training scripts | Quickstart, programming guides, algorithm explanations | Example YAML configs, Colab notebook |
| **Multi-language** | English | English, Chinese, Japanese | English, Chinese | English |
| **Academic paper** | No (but many blog posts) | Yes (EMNLP 2025 demo) | Yes (multiple papers) | No |
| **Blog posts** | 10+ official blog posts | Moderate | Moderate | No |

---

## Community and Ecosystem

| Dimension | **TRL** | **OpenRLHF** | **veRL** | **oxRL** |
|---|---|---|---|---|
| **GitHub stars** | ~17.5k | ~9.1k | ~19.6k | ~16 |
| **Contributors** | ~471 | ~80+ | ~100+ | 1 |
| **Forks** | ~2.5k | ~887 | ~3.4k | 1 |
| **Releases** | 76 (latest v0.29.0) | Regular | Regular | Early |
| **Corporate backing** | Hugging Face | Independent (academic origin) | ByteDance / Volcano Engine | Independent (solo developer) |
| **Institutional adopters** | Widely adopted across industry | Research groups, some industry | ByteDance, Alibaba Qwen, Anyscale, UC Berkeley, UCLA, UIUC | None documented |
| **HF Hub integration** | Native (push_to_hub, model cards) | Manual | Manual | Manual |

---

## Unique Features

| Framework | Unique Strengths |
|---|---|
| **TRL** | Deepest HuggingFace integration (Accelerate, PEFT, Hub). Multi-loss combinations (MPO). Liger Kernel integration. OpenEnv for agentic RL environments. Vision-language DPO out of the box. RapidFire experimentation engine. Unsloth integration for memory efficiency. Largest contributor ecosystem. |
| **OpenRLHF** | Agent-based execution paradigm ("token-in-token-out"). Hybrid engine scheduling (models + vLLM share GPUs). Dynamic filtering (DAPO). Async RL training pipeline. NeMo Gym integration for multi-turn environments. Production-tested at scale. |
| **veRL** | Broadest distributed strategy support (FSDP + Megatron + DeepSpeed). Hybrid-controller programming model. SGLang support. Scales to 671B parameters. Sandbox fusion for verifiable rewards. Backed by large research institutions. Fastest-growing star count. |
| **oxRL** | Widest algorithm coverage (51 algorithms, 35+ DPO variants). Automated model verification pipeline (38 verified models). Pydantic config validation with typo detection. LLM-oriented codebase design (optimized for AI-assisted development). `oxrl doctor` environment diagnostics. Autonomous bug reporting. Multimodal support (vision + audio). |

---

## Honest Assessment of oxRL

### Strengths
- **Algorithm breadth**: 51 algorithms with standalone implementations, each with its own file and test. No other framework comes close for offline preference optimization coverage.
- **Verified model matrix**: 38 models explicitly tested through an automated pipeline, across text, vision, audio, and MoE architectures.
- **Developer experience**: 3-line Python API, strict YAML validation, environment diagnostics via `oxrl doctor`.
- **LLM-friendly codebase**: Designed for AI-assisted development with clear file boundaries and minimal abstraction layers.

### Weaknesses (being honest)
- **Tiny community**: 1 contributor, ~16 stars. No external validation at scale. If the maintainer steps away, the project has no bus factor.
- **Not battle-tested at scale**: Largest verified model is 35B (Qwen3.5-35B-A3B with LoRA). No evidence of 70B+ full fine-tuning or 671B MoE training. TRL, OpenRLHF, and veRL have all been used for large-scale production training.
- **Limited distributed strategies**: DeepSpeed ZeRO-3 only. No FSDP, no Megatron-LM. This limits scaling options compared to veRL.
- **No hosted documentation**: README-only docs cannot compete with TRL's comprehensive doc site or veRL's readthedocs.
- **No academic paper**: No peer-reviewed publication establishing correctness or benchmarking results.
- **No SGLang support**: Only vLLM for inference, while veRL supports both.
- **Online RL algorithm coverage is narrower**: 4-6 RL algorithms vs OpenRLHF's and veRL's deeper online RL focus.
- **No async or disaggregated RL**: Co-located architecture only. Not suitable for agentic multi-turn RL with tool use.
- **No corporate backing**: Independent project without institutional resources for maintenance, CI infrastructure, or long-term support.

### When to choose oxRL
- You want to **experiment with many DPO/preference optimization variants** without reimplementing each one.
- You need a **lightweight framework** that works on consumer GPUs (LoRA-first for 7B+).
- You value **config validation and debuggability** over maximum throughput.
- You are doing **multimodal post-training** (vision + audio) on models up to ~35B.

### When to choose something else
- **TRL**: You want the safest, most mature choice with the largest community and HuggingFace ecosystem integration.
- **OpenRLHF**: You need production-grade online RL (PPO/GRPO) at 70B+ scale with async training.
- **veRL**: You need maximum scalability (671B), Megatron-LM support, or SGLang integration.

---

*Last updated: March 2026. Star counts and contributor numbers are approximate and change frequently. Algorithm counts include variants and aliases where noted.*
