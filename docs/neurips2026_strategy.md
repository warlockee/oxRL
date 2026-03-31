# oxRL NeurIPS 2026 Submission Strategy

**Target venue**: NeurIPS 2026 (Datasets & Benchmarks track)
**Submission deadline**: ~May 25, 2026
**Last updated**: 2026-03-20

---

## Narrative

Frame oxRL as the experimental apparatus, not the contribution. The contribution is **insights from 200+ controlled training runs** across algorithms, model scales, and tasks.

**Core argument**: The RL post-training literature suffers from a comparability crisis. Each new algorithm (DPO, SimPO, KTO, GRPO, ...) is evaluated with its own codebase, data pipeline, hyperparameters, and evaluation protocol. We provide the first apples-to-apples comparison where all algorithms share identical data processing, tokenization, model initialization, reward signals, and evaluation, enabling conclusions about algorithms rather than implementations.

**"Why now?" argument**: The post-DeepSeek-R1 landscape has popularized GRPO and spawned dozens of variants, but no controlled study has compared them against the established offline preference methods (DPO and its variants) under matched conditions. Existing comparisons (Tajwar et al. 2024, Ivison et al. 2024) predate GRPO, cover fewer algorithms, and use fewer model scales.

### Contribution Bullets

1. **Controlled cross-family comparison**: First study to compare online RL methods (GRPO, PPO, CISPO) against offline preference methods (DPO, SimPO, KTO, IPO) and supervised baselines (SFT, Best-of-N) under identical conditions across 4 model scales (0.5B--7B Qwen2.5) and 3 tasks (GSM8K, MATH, MBPP), totaling 200+ training runs with statistical analysis.
2. **DPO variant indistinguishability**: Head-to-head comparison of 20 DPO/preference optimization variants showing the degree to which loss function modifications produce statistically distinguishable outcomes under matched conditions.
3. **Scaling laws for RL post-training**: Characterization of how the benefit of online vs. offline methods scales with model size and compute budget, reported as accuracy per GPU-hour.
4. **Practitioner decision framework**: Evidence-based guidelines mapping (model scale, task type, compute budget) to recommended algorithm, supported by ablations over n_samples, KL coefficient, and reward design.

### Decision: GSPO Demoted to Ablation

GSPO (sequence-level clipped surrogate) is demoted from a primary contribution to an ablation finding (Tier 2, Ablation B). Rationale:

- Without a formal variance bound or theorem, GSPO is a minor implementation variant of SGRPO.
- The planned experiments use dense Qwen2.5 models, not MoE. GSPO's value proposition is MoE-specific.
- Including GSPO as a contribution invites the reviewer objection "this is incremental" and dilutes the empirical narrative.
- GSPO remains in the algorithm grid as one of the 9 core algorithms. If the SGRPO-vs-GSPO ablation on dense models yields a surprising result, it can be elevated during writing.

---

## Online/Offline Comparability Protocol

**Problem**: Online RL methods (GRPO, PPO, CISPO) generate their own rollouts and score them with the reward function. Offline preference methods (DPO, SimPO, KTO, IPO) require pre-computed (chosen, rejected) pairs. Comparing them naively is unfair because online methods get to explore while offline methods are constrained to a fixed dataset.

**Solution**: Generate preference data from the base model's own rollouts, scored by the same reward function used by online methods. This ensures the reward signal, data distribution, and initial policy are identical across all algorithms.

### Preference Data Generation Pipeline

For each (model, task) pair:

1. Load the prompt-only dataset (e.g., GSM8K train prompts).
2. Generate N=16 responses per prompt using the base model via vLLM (temperature=1.0, unbiased sampling).
3. Score each response using the task reward function (gsm8k_reward_func, math_reward_func, or code_reward_func).
4. For each prompt, select the highest-scoring response as "chosen" and the lowest-scoring as "rejected." Discard prompts where all responses receive the same score (no signal).
5. Output a prompt_preference Parquet file compatible with oxrl's PromptPreferenceDataset.

This pipeline is implemented in `oxrl/data/generate_prefs.py` and produces output to `/home/ec2-user/fsx/oxrl_data/neurips2026/`.

### Compute Cost Accounting

The preference data generation step is an inference cost shared across all offline algorithms. It is reported separately in the compute budget. Online methods pay an equivalent or higher cost at each training epoch (rollout generation), so the comparison is compute-fair.

---

## Experiment Plan

### Tier 1: Core Experiments (Required)

#### Experiment 1: Multi-Algorithm Comparison on Math (GSM8K + MATH)

| Model | Algorithms | Datasets | Metrics | Seeds |
|-------|-----------|----------|---------|-------|
| Qwen2.5-0.5B-Instruct | SFT, DPO, SGRPO, GSPO, CISPO, PPO, SimPO, KTO, IPO, Best-of-N, Base | GSM8K, MATH | Accuracy (greedy) | 3 |
| Qwen2.5-1.5B-Instruct | Same 11 | GSM8K, MATH | Same | 3 |
| Qwen2.5-3B-Instruct | Same 11 | GSM8K, MATH | Same | 3 |
| Qwen2.5-7B-Instruct | Same 11 | GSM8K, MATH | Same | 3 |

Notes:
- Base = no post-training (zero-shot evaluation only, no training run).
- Best-of-N = inference-only baseline (generate 64 responses, pick highest reward). No gradient updates.
- MATH added alongside GSM8K to avoid ceiling effects at 7B scale.
- 9 trainable algorithms x 4 scales x 2 tasks x 3 seeds = 216 runs.
- Plus 4 x 2 = 8 base evaluations + 4 x 2 x 3 = 24 Best-of-N evaluations = 248 total configurations.

#### Experiment 2: Transfer to Code

| Model | Algorithms | Dataset | Metric | Seeds |
|-------|-----------|---------|--------|-------|
| Qwen2.5-Coder-1.5B-Instruct | SFT, DPO, SGRPO, GSPO, CISPO, PPO, SimPO, KTO, IPO | MBPP | pass@1 | 3 |
| Qwen2.5-Coder-7B-Instruct | Same 9 | MBPP | pass@1 | 3 |

Total: 2 x 9 x 3 = 54 runs (all 9 algorithms on both scales).

#### Experiment 3: DPO Variant Sweep

**Strongest potential finding**: If 20 DPO variants are statistically indistinguishable, this contradicts the implicit claim of each variant paper and is a high-impact empirical result.

| Model | Algorithms | Dataset | Metric | Seeds |
|-------|-----------|---------|--------|-------|
| Qwen2.5-1.5B-Instruct | DPO, IPO, SimPO, KTO, ORPO, CPO, AlphaPO, RDPO, CDPO, BetaDPO, CalDPO, SPPO, APO, Hinge, RobustDPO, EXO, ODPO, DPOP, FocalPO, GPO | GSM8K | Accuracy (greedy) | 5 |

Total: 20 x 5 = 100 runs at 1.5B scale (cheap, ~2 GPU-hours each).

Analysis: pairwise Wilcoxon signed-rank tests across all 190 pairs, with Bonferroni correction. Report the number of statistically significant differences at p < 0.05 / 190.

#### Experiment 4: Compute-Normalized Comparison

Derived from Experiments 1--2 (no additional runs). For each algorithm at each scale:
- Record wall-clock GPU-hours for training (including rollout generation for online methods, including preference data generation for offline methods).
- Plot accuracy vs. cumulative GPU-hours (Pareto frontier).
- Report accuracy-per-GPU-hour ratio.

This is a Tier 1 analysis because reviewers will demand it: "GRPO gets 85% on GSM8K, but at what cost compared to DPO?"

#### Experiment 5: Scaling Laws

Derived from Experiment 1 data. For the Qwen2.5 family (0.5B, 1.5B, 3B, 7B):
- Final accuracy vs. log(model parameters) for each algorithm, fit power law.
- Final accuracy vs. log(training compute in GPU-hours).
- Convergence curves: accuracy vs. training epoch at each scale.
- Key question: do online methods show a steeper scaling slope than offline methods?

### Tier 2: Ablation Studies (Required for Strong Paper)

| Ablation | Variable | Values | Model | Runs |
|----------|----------|--------|-------|------|
| A: Rollout samples | n_samples (SGRPO) | 1, 2, 4, 8, 16, 32 | Qwen2.5-1.5B | 6 x 3 seeds = 18 |
| B: SGRPO vs GSPO on dense | loss_variant | sgrpo, gspo | Qwen2.5-{1.5B, 7B} | 2 x 2 x 3 = 12 |
| C: Reward design | reward_func | binary, graduated, reasoning | Qwen2.5-1.5B | 3 x 3 seeds = 9 |
| D: KL coefficient | kl_coeff | 0.0, 0.001, 0.01, 0.1 | Qwen2.5-1.5B | 4 x 3 seeds = 12 |
| E: Learning rate sensitivity | lr | 5e-7, 1e-6, 5e-6 | Qwen2.5-1.5B, top 3 algs | 3 x 3 x 3 = 27 |

Total ablation runs: 78

### Tier 3: Analysis (Best Paper Territory)

- Training dynamics: KL divergence, clip fraction, reward distribution, entropy over time for all algorithms on shared axes.
- Failure mode detection: identify runs where training reward increases but eval accuracy drops (reward hacking).
- Qualitative case studies: algorithm-specific success and failure examples.
- Cross-task correlation: does algorithm ranking on GSM8K predict ranking on MATH? On MBPP?

---

## Compute Budget (Revised)

| Experiment | Runs | GPU-hours/run | Total |
|-----------|------|--------------|-------|
| Exp 1: Math (216 training runs) | 216 | 2-8 | ~600 |
| Exp 2: Code (54 runs) | 54 | 4-8 | ~300 |
| Exp 3: DPO variant sweep (100 runs) | 100 | ~2 | ~200 |
| Tier 2 ablations (78 runs) | 78 | 2-8 | ~250 |
| Preference data generation | ~12 | 2-4 | ~40 |
| Evaluation (lm-eval-harness) | ~500 | 0.5 | ~250 |
| **Total** | | | **~1640 GPU-hours** |

~8-10 days on 8xH100, or ~4-5 days on 2 nodes (16 GPUs).

---

## Implementation Plan (Phase 1: Infrastructure)

### 1. Preference Data Generation (`oxrl/data/generate_prefs.py`) -- P0
- Standalone script (no Ray dependency) using vLLM directly.
- Input: prompt-only Parquet/JSONL + model path + reward function name.
- Output: prompt_preference Parquet to `/home/ec2-user/fsx/oxrl_data/neurips2026/`.
- CLI: `python -m oxrl.data.generate_prefs --model <path> --dataset <path> --reward gsm8k_reward_func --n-responses 16 --output <path>`

### 2. Evaluation Harness (`oxrl/eval/`) -- P0
- `evaluator.py` -- wraps lm-evaluation-harness for GSM8K, MATH, MBPP.
- `run_eval.py` -- CLI: `python -m oxrl.eval.run_eval --checkpoint <path> --tasks gsm8k,math,mbpp`
- Outputs structured JSON: `{task, accuracy, num_correct, num_total, per_example_results}`.

### 3. Experiment Sweep Launcher (`oxrl/sweep/`) -- P1
- `sweep.py` -- generates configs for all (algorithm x model x seed x task) combinations.
- `launcher.py` -- launches experiments sequentially or via SLURM.
- `results.py` -- aggregates JSON results into CSV/tables and LaTeX.

### 4. Best-of-N Baseline -- P2
- Implemented via `oxrl/data/generate_prefs.py` with `--best-of-n` flag.
- Generate 64 responses per prompt, select highest reward, report accuracy.

### 5. Results Logging -- integrated into sweep/results.py
- Structured JSON per run: `results/{experiment_id}/metrics.json`.
- Aggregation: CSV with columns (model, algorithm, task, seed, accuracy, gpu_hours, ...).

---

## Timeline (Working Backward from May 25, 2026)

| Week | Dates | Deliverable |
|------|-------|-------------|
| W1 | Mar 14--20 | Infrastructure: eval harness, sweep launcher, preference data pipeline |
| W2 | Mar 21--27 | Preference data generation for all (model, task) pairs. Validate pipeline end-to-end with 1 algorithm x 1 model x 1 seed. |
| W3 | Mar 28--Apr 3 | Run Exp 1 GSM8K: 9 algs x 4 scales x 3 seeds = 108 runs |
| W4 | Apr 4--10 | Run Exp 1 MATH: 108 runs. Begin Exp 2 Code: 54 runs. |
| W5 | Apr 11--17 | Complete Exp 2. Run Exp 3 DPO variant sweep: 100 runs. |
| W6 | Apr 18--24 | Run evaluations on all checkpoints. Begin Tier 2 ablations (78 runs). |
| W7 | Apr 25--May 1 | Complete ablations. Tier 3 analysis: training dynamics, failure modes. |
| W8 | May 2--8 | Results aggregation. Generate all tables and figures. Begin writing. |
| W9 | May 9--15 | Full draft: Introduction, Methods, Results, Analysis. |
| W10 | May 16--22 | Revision, verification checklist, anti-AI detection sweep. |
| W11 | May 23--25 | Final polish, PDF rendering check, submission. |

**Slack**: W3-W6 have the most compute. If runs fail or need reruns, W7 is the buffer.

---

## Paper Structure (9 pages + refs + appendix)

| Section | Pages | Content |
|---------|-------|---------|
| Abstract | 0.25 | Problem (no controlled comparison), approach (200+ runs, unified framework), key finding (TBD after experiments), implication (practitioner guidance) |
| Introduction | 1.25 | Proliferation of algorithms, no apples-to-apples comparison, our approach, contribution bullets |
| Background & Related Work | 1.5 | RL post-training taxonomy (online RL vs offline preference vs supervised), existing frameworks (TRL, OpenRLHF), prior comparison studies and their limitations |
| Experimental Setup | 1.0 | oxRL framework, models (Qwen2.5 family), tasks (GSM8K, MATH, MBPP), preference data generation protocol, evaluation protocol, compute budget |
| Main Results | 2.0 | Algorithm comparison table, scaling figure, cross-task analysis, DPO variant sweep, compute-normalized Pareto frontier |
| Analysis & Ablations | 2.0 | Ablation studies, training dynamics plots, failure mode analysis |
| Discussion & Limitations | 0.75 | Key findings, when to use each algorithm, limitations (single model family, limited task diversity) |
| Conclusion | 0.25 | Summary, future work (more tasks, MoE models, reward model comparison) |

---

## Anti-AI Detection Protocol (from review_feedback.md)

### Before Drafting
- Run all experiments and record raw numbers -- never generate placeholder numbers
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
- Check bounds: accuracy <= 100%, KL >= 0, clip fraction in [0,1]
- Test PDF text extraction for garbled equations
- Use correct NeurIPS 2026 template (anonymous, "Submitted to")

---

## Essential Baselines and Citations

### Baselines to implement
1. Base model (no post-training, zero-shot evaluation only)
2. SFT (supervised fine-tuning on correct solutions)
3. Best-of-N rejection sampling (inference baseline, N=64)
4. DPO (Rafailov et al., 2023)
5. IPO (Azar et al., 2023)
6. SimPO (Meng et al., 2024)
7. KTO (Ethayarajh et al., 2024)
8. SGRPO (token-level clipped surrogate, DeepSeek-R1 style)
9. GSPO (sequence-level clipped surrogate)
10. CISPO (conservative indirect surrogate)
11. PPO (Schulman et al., 2017)
12. 20 DPO variants for Experiment 3 (already implemented in oxrl)

### Must-cite papers
- RLHF (Ouyang et al., 2022)
- DPO (Rafailov et al., 2023)
- IPO (Azar et al., 2023)
- PPO (Schulman et al., 2017)
- SimPO (Meng et al., 2024)
- KTO (Ethayarajh et al., 2024)
- DeepSeek-R1 (DeepSeek, 2025) -- GRPO at scale
- RLOO/REINFORCE leave-one-out (Ahmadian et al., 2024)
- TRL (HuggingFace), OpenRLHF -- competing frameworks
- Tajwar et al. (2024) -- prior preference fine-tuning comparison
- Ivison et al. (2024) -- DPO vs PPO unpacking
- Each of the 20 DPO variant papers included in Experiment 3

---

## Resolved Decisions Log

| Decision | Resolution | Rationale |
|----------|-----------|-----------|
| GSPO: promote or demote? | **Demoted to ablation** | No theorem, no MoE model in experiments, weakens narrative if elevated |
| Online/offline fairness | **Generate preference data from base model rollouts** | Same reward signal, same initial distribution, compute-fair |
| Number of tasks | **3: GSM8K + MATH + MBPP** | GSM8K alone has ceiling effects at 7B; MATH provides harder math eval |
| DPO variant sweep | **20 variants at 1.5B, 5 seeds, Tier 1** | Strongest potential finding; low compute cost; leverages oxrl's unique advantage (51 algorithms) |
| Compute-normalized comparison | **Tier 1 analysis** | Reviewers will demand it; derived from existing runs (no extra compute) |
| Seeds | **3 for core, 5 for DPO variant sweep** | 3 is minimum for core; 5 needed for statistical power with 20 variants |
| Hyperparameter sensitivity | **Added as Ablation E** | Preempts "did you tune hyperparameters per algorithm?" reviewer objection |
