# oxRL NeurIPS 2026 Submission Strategy

## Strongest Narrative: Large-Scale Controlled Comparison of RL Post-Training Algorithms

Frame oxRL as the experimental apparatus, not the contribution. The contribution is **insights from 100+ runs** across algorithms, scales, and tasks.

### Contribution Bullets (Draft)

1. **First controlled, large-scale comparison** of 9 RL post-training algorithms across 4 model scales (0.5B-7B), 2+ tasks, and 100+ training runs.
2. **GSPO: a sequence-level policy gradient** designed for MoE models that absorbs expert-routing noise between inference and training engines.
3. **Scaling laws for RL post-training**: identify how online vs offline methods scale with model size and compute.
4. **Practical guidelines** for algorithm selection based on model architecture, task type, and compute budget.

---

## Top 5 Weaknesses → Fixes

| # | Weakness | Severity | Fix |
|---|----------|----------|-----|
| 1 | No experimental results exist | Fatal | Run 144+ training runs: 4 scales × 9 algorithms × 2 tasks × 3 seeds |
| 2 | Single-task evaluation | High | Math (GSM8K) + Code (MBPP) + ideally instruction-following (AlpacaEval) |
| 3 | No scalability analysis | High | Plot accuracy vs model scale (0.5B→7B) and accuracy per GPU-hour |
| 4 | Missing baselines | High | Add Best-of-N rejection sampling, REINFORCE/RLOO, base model (no post-training) |
| 5 | No ablations | Medium-High | n_samples, KL coeff, clip range, SGRPO vs GSPO on dense vs MoE, reward design |

---

## Experiment Plan

### Tier 1: Core Experiments (Required)

#### Experiment 1: Multi-Algorithm Comparison on Math

| Model | Algorithms | Dataset | Metric | Seeds |
|-------|-----------|---------|--------|-------|
| Qwen2.5-0.5B-Instruct | SFT, DPO, SGRPO, GSPO, CISPO, PPO, SimPO, KTO, IPO | GSM8K | Accuracy (greedy) | 3 |
| Qwen2.5-1.5B-Instruct | Same | GSM8K | Same | 3 |
| Qwen2.5-3B-Instruct | Same | GSM8K | Same | 3 |
| Qwen2.5-7B-Instruct | Same | GSM8K | Same | 3 |

Total: 4 × 9 × 3 = 108 runs

#### Experiment 2: Transfer to Code

| Model | Algorithms | Dataset | Metric | Seeds |
|-------|-----------|---------|--------|-------|
| Qwen2.5-Coder-1.5B-Instruct | SFT, DPO, SGRPO, PPO, SimPO, KTO | MBPP | pass@1 | 3 |
| Qwen2.5-Coder-7B-Instruct | Same | MBPP | pass@1 | 3 |

Total: 2 × 6 × 3 = 36 runs

#### Experiment 3: Scaling Laws

Plot using Qwen2.5 family (0.5B, 1.5B, 3B, 7B):
- Final accuracy vs model scale for each algorithm
- Final accuracy vs total training compute (GPU-hours)
- Convergence curves (reward vs epoch) at each scale

### Tier 2: Ablation Studies (Required for Strong Paper)

| Ablation | Variable | Values | Model | Runs |
|----------|----------|--------|-------|------|
| A: Rollout samples | n_samples | 1, 2, 4, 8, 16, 32 | Qwen2.5-1.5B | 18 |
| B: Dense vs MoE | SGRPO vs GSPO | token vs sequence | Qwen2.5-7B + Qwen3.5-35B-A3B | 12 |
| C: Reward design | reward_func | binary, graduated, reasoning | Qwen2.5-1.5B | 9 |
| D: KL coefficient | kl_coeff | 0.0, 0.001, 0.01, 0.1 | Qwen2.5-1.5B | 12 |

### Tier 3: Analysis (Best Paper Territory)

- Training dynamics: KL divergence, clip fraction, reward distribution, entropy over time
- Failure mode analysis: reward hacking detection, mode collapse detection
- Qualitative examples: algorithm-specific success/failure cases

---

## Compute Budget

| Experiment | Runs | GPU-hours/run | Total |
|-----------|------|--------------|-------|
| Core math (108 runs) | 108 | 2-8 | ~400 |
| Core code (36 runs) | 36 | 4-8 | ~200 |
| Ablations (51 runs) | 51 | 4-16 | ~300 |
| Evaluation (lm-eval-harness) | ~200 | 0.5 | ~100 |
| **Total** | **~395** | | **~1000 GPU-hours** |

~5-6 days on 8×H100, or ~2-3 days on 2 nodes.

---

## Implementation Plan (Phase 1: Infrastructure)

### 1. Evaluation Harness (`oxrl/eval/`)
- `evaluator.py` — wraps lm-evaluation-harness for GSM8K, MATH, MBPP
- `run_eval.py` — CLI script: `python -m oxrl.eval.run_eval --checkpoint <path> --tasks gsm8k,mbpp`
- Outputs structured JSON results

### 2. Experiment Sweep Launcher (`oxrl/sweep/`)
- `sweep.py` — generates configs for all (algorithm × model × seed × task) combinations
- `launcher.py` — launches experiments sequentially or via SLURM/Ray
- `results.py` — aggregates JSON results into CSV/tables

### 3. Best-of-N Baseline (`oxrl/algs/best_of_n.py`)
- Generate N responses per prompt using vLLM
- Select response with highest reward
- Report accuracy without any gradient updates (inference-only baseline)

### 4. Results Logging
- Structured JSON per run: `results/{experiment_id}/metrics.json`
- Aggregation script: `oxrl/sweep/results.py` → CSV with columns: model, algorithm, task, seed, accuracy, gpu_hours

---

## Paper Structure (9 pages + refs + appendix)

| Section | Pages | Content |
|---------|-------|---------|
| Abstract | 0.25 | Problem, gap, approach, key result, implication |
| Introduction | 1.25 | Proliferation of algorithms with no controlled comparison |
| Background & Related Work | 1.5 | RL post-training landscape, existing frameworks, evaluation methodology |
| Experimental Setup | 1.0 | oxRL framework, models, tasks, datasets, evaluation protocol |
| Main Results | 2.0 | Algorithm comparison table + scaling figure + cross-task analysis |
| Analysis & Ablations | 2.0 | Ablation studies, training dynamics, failure modes |
| Discussion & Limitations | 0.75 | Key findings, when to use each algorithm |
| Conclusion | 0.25 | Summary, future work |

---

## Anti-AI Detection Protocol (from review_feedback.md)

### Before Drafting
- Run all experiments and record raw numbers — never generate placeholder numbers
- Create single source-of-truth spreadsheet with all results
- Verify dataset stats against actual data files

### During Drafting
- No meme phrases: "with extra steps", "game-changing", "paradigm shift"
- No AI filler: "notably", "crucially", "importantly", "interestingly", "remarkably"
- No Hollywood narrative: no "Day N", no "eureka moment"
- Use passive voice for findings
- Verify every citation against actual paper

### Before Submission
- Grep all numbers appearing more than once for cross-section consistency
- Verify all products/sums compute correctly
- Check bounds: accuracy ≤ 100%, KL ≥ 0, clip fraction ∈ [0,1]
- Test PDF text extraction for garbled equations
- Use correct NeurIPS 2026 template (anonymous, "Submitted to")

---

## Essential Baselines and Citations

### Baselines to implement
1. SFT (supervised fine-tuning)
2. DPO (Rafailov et al., 2023)
3. GRPO/SGRPO (DeepSeek-R1 style)
4. PPO (Schulman et al., 2017)
5. SimPO (Meng et al., 2024)
6. KTO (Ethayarajh et al., 2024)
7. IPO (Azar et al., 2023)
8. Best-of-N rejection sampling (inference baseline)
9. Base model (no post-training)

### Must-cite papers
- TRL (HuggingFace), OpenRLHF
- RLHF (Ouyang et al., 2022)
- DPO (Rafailov et al., 2023)
- DeepSeek-R1 (DeepSeek, 2025)
- SimPO, KTO, ORPO, IPO
- PPO (Schulman et al., 2017)
- RLOO/REINFORCE leave-one-out

---

## What Would Make This Best Paper

1. Clean scaling law for RL post-training (e.g., "online RL benefit scales as O(n^0.7)")
2. Formal proof why GSPO is necessary for MoE (bound on log-ratio divergence under router noise)
3. Surprising algorithmic insight contradicting conventional wisdom
4. Head-to-head of >10 DPO variants showing most are statistically indistinguishable
