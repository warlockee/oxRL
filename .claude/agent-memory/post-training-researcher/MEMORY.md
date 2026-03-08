# Post-Training Researcher Agent Memory

## oxRL Framework Architecture
- Version: 1.7.0 (pyproject.toml)
- RL algs: `oxrl/algs/grpo.py` (SGRPO/GSPO/CISPO/RLHF/RLAIF), `oxrl/algs/ppo.py`
- SL algs: Individual files in `oxrl/algs/` (see list below)
- Loss registry: `oxrl/algs/losses/__init__.py` with `get_loss_fn()` dispatch
- Config schema: `oxrl/configs/schema.py` (Pydantic models, extra='forbid')
- RL entry: `main_rl.py` (Ray + DeepSpeed), SL entry: `main_sl.py` (DeepSpeed only)
- SL_ALGORITHMS dict in `main_sl.py` maps alg_name -> class
- Dataset routing in `main_sl.py`: preference algs use PromptPreferenceDataset
- Tests: `tests/test_bugs.py`, `tests/test_lop_refactoring.py`, per-algorithm tests

## SL Algorithm Integration Pattern (checklist)
1. Create `oxrl/algs/{name}.py` with `train_step`, `eval_step`, `forward`, `compute_loss`
2. Add import + class to `oxrl/algs/__init__.py`
3. Add import + dict entry in `main_sl.py` SL_ALGORITHMS
4. Add alg_name to preference dataset routing list in `data_loader_setup()`
5. Add elif block for algorithm instantiation with config params
6. Add any new config fields to `oxrl/configs/schema.py` Train class
7. Write tests in `tests/test_{name}.py`
8. Create example config in `registry/examples/{name}.yaml`

## Onboarded Methods (all verified with GPU training runs)
- **CPO** (ICML 2024): Reference-free pref + BC reg. 22 tests. Train loss: 0.771->0.368
- **AlphaPO** (2025): SimPO + alpha reward shape transform. 22 tests. Train loss: 0.629->0.832
- **R-DPO** (2024): DPO + length regularization. 16 tests. Train loss: 0.724->0.689
- **cDPO** (Mitchell 2023): DPO + label smoothing for noise. 14 tests. Train loss: 0.715->0.688
- **beta-DPO** (NeurIPS 2024): Dynamic per-sample beta. 19 tests. Train loss: 0.691->~0
- **Cal-DPO** (NeurIPS 2024): BT loss + calibration regression. 19 tests. Train loss: 48.5->81
- **SPPO** (2024): Nash eq squared-error loss. 16 tests. Train loss: 49.0->112
- **AOT** (2024): Optimal transport sorted quantile DPO. 15 tests. Train loss: 0.691->~0
- **APO** (TACL 2024): Anchored pref opt (zero/down modes). 18 tests. Train loss: -0.003->-0.648
- **NCA** (NeurIPS 2024): Noise contrastive alignment. 15 tests. Train loss: 1.38->0.957
- **Hinge/SLiC** (2023): Max-margin hinge loss. 15 tests. Train loss: 0.977->0
- **Robust DPO** (ICML 2024): Unbiased DPO under label noise. 17 tests. Train loss: 0.859->0.42
- **EXO** (ICML 2024): Reverse KL pref opt. 16 tests. Train loss: 2.72->~0
- **DiscoPOP** (NeurIPS 2024): Log-ratio modulated loss. 17 tests. Train loss: 0.859->0.848
- **BCO** (2024): Binary classifier pairwise opt. 16 tests. Train loss: 1.38->0.434
- **ODPO** (ACL 2024 Findings): DPO with offset. 16 tests. Train loss: 1.30->~0
- **DPOP** (2024): DPO-Positive/Smaug, prevents chosen degradation. 16 tests. Train loss: 0.684->~0
- **FocalPO** (2025): Focal weighting on DPO. 17 tests. Train loss: 0.344->~0
- **GPO** (ICML 2024): Generalized PO with convex loss functions. 27 tests. Train loss: 0.977->~0
- **WPO** (EMNLP 2024): On-policy reweighted DPO. 19 tests. Train loss: 0.684->~0
- **f-DPO** (ICLR 2024): f-divergence DPO (fwd/rev KL, JS, alpha). 20 tests. Train loss: 0.738->0.719
- **H-DPO** (2024): Entropy controllable DPO. 16 tests. Train loss: 0.926->~0
- **DPO-Shift** (2025): Shifted rejected term to reduce likelihood displacement. 20 tests. Train loss: 0.684->~0
- **CPO-SimPO** (2024): Reference-free CPO+SimPO (len-norm + margin + BC reg). 22 tests. Train loss: 0.623->~0
- **SamPO** (EMNLP 2024): Down-sampled KL divergence DPO (length debiasing). 18 tests. Val loss: 0.684->0.218
- **Dr-DPO** (ICLR 2025): DRO log-sum-exp aggregation for robustness. 21 tests. Val loss: 0.684->0.190
- **Chi-PO** (ICLR 2025): Chi-squared divergence link phi(z)=z+log(z). 23 tests. Val loss: 0.674->0.109
- **SPO** (EMNLP 2025 Findings): SiLU replaces logsigmoid (bounded loss). 18 tests. Val loss: -1.19->-12.67
- **DPNLL** (DPO+NLL): DPO + NLL regularization on chosen. 23 tests. Val loss: 0.485->0.406
- **MinorDPO** (2024): Clamp rejected logr to non-negative. 20 tests. Val loss: 0.582->0.500
- **C2DPO** (2025): DPO + L2 penalty on (logr_w+logr_l)^2. 21 tests. Val loss: 0.297->0.204

## Total: 694 tests, all passing. 31 new SL algorithms onboarded.

## Environment Notes
- CUDA_HOME: `/ceph/workspace/erik/oxRL/cuda_env`
- `main_sl.py` accepts `--local_rank` arg for deepspeed launcher
- Test data: `tests/data/dummy_preference.parquet` (3 preference samples)
- Test model: `Qwen/Qwen2.5-0.5B-Instruct` with eager attention
- DeepSpeed ZeRO stage 2 for tests (lighter than stage 3)
- GPU test config needs `flops_profiler: enabled: false` in deepspeed section
- Use `deepspeed --include localhost:N` to select specific GPU (not CUDA_VISIBLE_DEVICES)
- GPUs 4,5,7 are often free for testing

## Model Onboarding Notes — see model-onboarding.md

## Candidate Methods — see candidates.md
