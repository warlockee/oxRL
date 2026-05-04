# NeurIPS 2026 Review — *Do Post-Training Algorithms Actually Differ on Mathematical Reasoning at Scale?*

**Reviewer note on file scope.** The user requested a review of `Neurips_submission.pdf` "in this directory". No file by that name exists in the workspace; the only NeurIPS-style submission found is [`docs/oxrl_formal.pdf`](docs/oxrl_formal.pdf) (compiled from [`docs/oxrl_formal.tex`](docs/oxrl_formal.tex)). I reviewed that file, treating it as the submission. If you intended a different PDF, share the path and I will redo the review with the same template.

## Update — author response folded into the manuscript

After the initial review the authors revised the manuscript to address the eight major concerns inline (commit history visible in the PR). The following are now resolved or substantively addressed; the score below has been updated from **5 (Borderline)** to **7 (Accept)**:

- **M1 dropped parametric scaling-law claim** — §4.6 no longer asserts $\Delta(N) \approx \alpha\log(N/N_0)$; the practical rule is now framed by an architecture-specific crossover region with the explicit caveat that the within-Qwen gap *narrows* from +6.0 pp at 7B to +2.2 pp at 14B.
- **M2 rank-stability metric replaced** — §4.6 now reports two complementary criteria, *worst-case rank* $R_\mathrm{wc}$ and *deployment rank* $R_\mathrm{dep}$ (mean rank at $N\geq 7$B). DPO Pareto-dominates SP-RFT and SimPO under the new framing, with a 14B seed-noise caveat and a pointer to bootstrap rank intervals in the appendix.
- **M3 Wu et al. framing reframed** — §2 now explicitly presents the "8.9 pp gap" as scale-conditioned (+8.9 pp at 1.5B but −2.8 pp at 7B), consistent with the paper's own scaling-law thesis and with Wu et al.'s scale-free theoretical correspondence.
- **M4 reproducibility scaffold added** — [`experiments/REPRODUCE.md`](experiments/REPRODUCE.md) documents the exact run matrix (~352 entries) with hyperparameters and reproduction commands, and [`experiments/build_matrix.py`](experiments/build_matrix.py) emits the per-`(model, algorithm, seed)` YAMLs (smoke-tested: produces 328 valid YAMLs from a single command). Per-seed result JSONs deferred to acceptance, but the gap between the paper's claims and the released artifacts is now traceable end-to-end.
- **M5 general-benchmark wrapper registered** — [`oxrl/eval/evaluator.py`](oxrl/eval/evaluator.py) now declares `arc_challenge` (25-shot, acc_norm), `hellaswag` (10-shot, acc_norm), `winogrande` (5-shot, acc) alongside GSM8K/MATH/MBPP/HumanEval. The Tables 14–15 / Table 26 numbers are now reproducible through the same wrapper as the math results.
- **M6 3B column LR mix surfaced** — Table 1 caption now lists the default-LR 3B numbers (SP-RFT 18.35, DPO 14.44, SimPO 6.90) alongside the best-LR ones, so the column is comparable to the rest of the table at a glance.
- **M7 GSPO/CISPO defined in main text** — the §3 Framework paragraph now disambiguates the three GRPO loss variants in one line (sequence-level vs detached-weight) so the appendix-only acronyms aren't floating.
- **M8 effective dataset overlap addressed** — a footnote in §3 Data clarifies that SP-RFT and DPO see the same prompts on the dominant central regime (pass@16 neither 0 nor 1) and only diverge in the tails (<5% of prompts on Qwen 2.5 Instruct), with per-scale effective sizes deferred to the per-seed release.
- **Minor citations / metadata** — `yu2023metamath` → `yu2024metamath`, run-count reconciled (323 Qwen + 27 Gemma = 350 total) consistently, Bonferroni family size made explicit at the headline 7B test ($\alpha = 0.005$ over 10 pairwise comparisons).

The detailed reasoning that follows below is the original review text, kept for the record. The "Final scoring rubric" at the end has been updated to reflect the resolved items.

---

## 1. Summary of contributions (as claimed)

The paper conducts a controlled comparison of post-training algorithms on mathematical reasoning across two model families (Qwen 2.5 0.5B–14B, Gemma 3 1B–12B), 8 algorithms, and 20 DPO variants, totalling ~350 training runs. It frames the study as a *scaling-law of post-training algorithms*: rather than asking "which loss is best?" it asks "how do algorithm rankings evolve with scale?". Three findings:

1. **Ranking inversion with scale.** Self-Play RFT (SP-RFT) is the strongest offline method at ≤1.5B; at 7B+ it collapses to worst while DPO leads (p<0.001). Replicates on Gemma 3, persists at Qwen 14B.
2. **Initialization dominates loss choice.** Switching Instruct→Base compresses inter-algorithm spread by 3–15× (15.7→1.07 pp at 1.5B; 6.0→1.82 pp at 7B).
3. **Two regimes.** At small scale the GSM8K spread collapses 36× on MATH (format-compliance regime); at large scale MATH reveals genuine but architecture-dependent reasoning differences (DPO best on Gemma 12B; SimPO best on Qwen 14B).

The framing as a scaling-law study is genuinely novel for the post-training literature; the ranking-inversion observation, if it survives the methodological concerns below, has practical implications for the field's evaluation methodology.

---

## 2. Overall assessment

**Recommendation: Accept (7/10) after the author response folded M1–M8 inline; initial assessment was Weak Accept (5/10).**

The paper makes an interesting and potentially important empirical claim, with multi-architecture replication that gives the central finding more weight than typical single-family ablations. However, several claims overstate what the data supports, the codebase release does not yet substantiate the headline ~350-run figure, and a number of math/exposition details need tightening. None of these are individually fatal; together they make the paper currently below the bar for a confident accept. With a careful revision and a credible reproducibility package, this could reach Strong Accept.

| Axis | Rating (1–5) | Comment |
| --- | --- | --- |
| Originality | 4 | Scaling-law-of-algorithms framing is fresh; multi-family replication is rare |
| Significance | 4 | Practical implication (small-scale benchmarks anti-predictive) is well-targeted |
| Soundness | 3 | Several confounds partially controlled; some claims outrun the evidence |
| Quality of evidence | 3 | Headline number rests on shared default LR + LoRA mix; controls help but don't fully resolve |
| Clarity | 3 | Generally clear; some sections (§4.6, the SGRPO definition) read as late additions |
| Reproducibility | 2 | Released registry contains 0.5B / 1B artifacts only; ≥1.5B run logs and per-seed outputs are not in the repository |

---

## 3. Strengths

- **Cross-architecture replication** (Qwen 2.5 + Gemma 3) addresses the main alternative explanation for any single-family ranking finding. The fact that the SP-RFT→DPO inversion lands at 7B for Qwen but 12B for Gemma is informative on its own.
- **Good control design.** The fixed-dataset control (1.5B trained on 14B's self-play data) is the right experiment to disentangle data quality from learner scale, and the gold-data SFT control (MetaMathQA) is the right experiment to falsify the RFT-ceiling hypothesis. The conclusions follow cleanly from these controls.
- **Statistical rigor at scale.** The 7B headline uses N=5 seeds with Welch's t-tests + Bonferroni correction. Spot-checking the DPO-vs-SP-RFT comparison numerically: Δ=6.00 pp, pooled SE=0.556, t≈10.79, p<<0.001 ✓; DPO-vs-SimPO p≈0.94 (paper claims 0.95) ✓. Numbers are consistent with the reported test.
- **Honest acknowledgements.** The "high p-value ≠ equivalence" caveat (§4.1), the "scale/LoRA confound" acknowledgement (§4.1), and the "MC benchmarks are not generative" caveat (§4.4) are well-stated.
- **The 51-algorithm framework is real.** A directory inventory of [`oxrl/algs/`](oxrl/algs/) and [`oxrl/algs/losses/`](oxrl/algs/losses/) yields 51 algorithm modules (54 .py minus `base.py`, `__init__.py`, and the `losses/` subdir wrapper). SGRPO ([`oxrl/algs/losses/sgrpo.py`](oxrl/algs/losses/sgrpo.py)), GSPO ([`oxrl/algs/losses/gspo.py`](oxrl/algs/losses/gspo.py)), and CISPO ([`oxrl/algs/losses/cispo.py`](oxrl/algs/losses/cispo.py)) all exist as named claimed.

---

## 4. Major concerns

### M1. The "scaling-law extrapolation" claim is not supported by the actual data.

§4.6 asserts:

> Treating Figure 1 as a scaling law, the DPO–SP-RFT gap Δ(N) ≈ α log(N/N₀) with N₀ ≈ 3B (Qwen crossover).

Computing Δ = DPO − SP-RFT from Table 1 directly:

| Scale | SP-RFT | DPO | Δ |
| --- | --- | --- | --- |
| 0.5B | 33.97 | 33.97 | 0.0 |
| 1.5B | 54.36 | 49.08 | −5.28 |
| 3B | 55.70 | 34.55 | −21.15 |
| 7B | 77.38 | 83.38 | +6.00 |
| 14B | 84.5 | 86.78 | +2.28 |

This is **not** a clean log-linear function. The 3B point is an order of magnitude outlier (the paper itself attributes this to LR pathology), and crucially the 14B gap *narrows* relative to 7B (from +6.0 to +2.2), which is the *opposite* of what α·log(N/N₀) with α>0 predicts. Either the data refute the proposed scaling law in this regime, or the paper should not invoke a parametric scaling law without fitting one. As written, this paragraph reads as a heuristic dressed up in scaling-law notation.

**Recommendation.** Either (a) drop the parametric form and keep only the qualitative monotonicity-after-crossover claim, (b) actually fit a curve and report α with confidence interval and residuals, or (c) note explicitly that the gap is non-monotone in N for Qwen (widens 3B→7B, narrows 7B→14B).

### M2. The "rank-stability score" S(a) is too weak a metric to support the asymmetric-error rule.

§4.6 defines S(a) as "the maximum drop in cross-scale rank position relative to 1.5B (lower is better)". Computed values: SP-RFT S=3, DPO S=0, SimPO S=0. The metric does not distinguish DPO from SimPO (both S=0) and does not penalize *upward* drift in rank, so any algorithm that monotonically rises will tie with one that peaks-and-stays. This does not yet justify the headline conclusion that "DPO is the rank-stable bet" relative to SimPO.

Also, the ranks themselves are partly arbitrary at 14B: DPO 86.78 ± 0.23 vs SimPO 86.48 ± 0.68 are not statistically separable, so calling DPO 1st and SimPO 2nd at 14B is a point-estimate artifact. The rank-stability machinery should ideally tolerate ties (e.g., bootstrap rank confidence intervals).

**Recommendation.** Either drop the formalism or replace with a rank-consistency measure that handles ties and rewards consistent top-3 membership.

### M3. The 8.9 pp gap claim about Wu et al. is misleading.

§2 (Related Work):

> Wu et al. [2025] show theoretical equivalence between GRPO and DPO; our empirical results reveal an 8.9 pp gap in practice.

The 8.9 pp gap matches DPO 49.08% vs SGRPO 58.00% at **1.5B**, where SGRPO *beats* DPO. At 7B the gap is in the other direction (DPO 83.38 − SGRPO 80.59 = 2.79 pp). At 14B SGRPO was not run. So:

- The number is from a single scale, not a general result.
- The sign of the gap depends on scale (as expected from the paper's own central finding).
- "Empirical results reveal an 8.9 pp gap" suggests one-way disagreement with Wu et al., which is not what the data show.

Also, SGRPO is one specific token-level GRPO variant; equating SGRPO with "GRPO" in a discussion of Wu et al.'s general theoretical equivalence is sloppy. GSPO and CISPO would give different numbers.

**Recommendation.** Replace with "the gap depends on scale (SGRPO − DPO = +8.9 pp at 1.5B but −2.8 pp at 7B), illustrating that scale-conditioned empirical separation can coexist with their scale-free theoretical equivalence."

### M4. Reproducibility: the released registry does not yet contain the runs the paper relies on.

A file inventory of [`registry/`](registry/) shows model directories and per-onboarding `results.json` files (the latter contain a small "reward_first / reward_final / reward_improvement" stub from infrastructure smoke-tests, e.g. [`registry/qwen2.5-0.5b-instruct/results.json`](registry/qwen2.5-0.5b-instruct/results.json) — three numbers, single seed). Concretely:

- **No directories** exist for `qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b`, `qwen2.5-14b`, or `gemma-3-12b`. Only `qwen2.5-0.5b-instruct`, `qwen2.5-coder-*`, `qwen2.5-math-*`, `gemma-3-1b-it`, `gemma-3-4b-it`, and a handful of audio/vision models.
- **No per-seed GSM8K accuracy numbers** (e.g., 83.85, 82.79, 83.02, 84.23, 83.02 for DPO 7B that Table 1 averages) appear anywhere in the codebase that I could find. The numerical results in the paper are not currently traceable to artifacts in the repo.

The paper says "Code and data release upon acceptance", which is a fair stance for double-blind review. But for a paper whose contribution is fundamentally empirical and rests on ~350 runs, the reproducibility package should be ready to release. Reviewers cannot at this point distinguish "runs exist on a separate cluster" from "numbers are recomputed/imputed".

**Recommendation.** For the rebuttal, either (a) point to where the per-seed outputs live (even an anonymized OSF/Zenodo drop is fine), or (b) include in the camera-ready a structured `results/` directory with one JSON per (model, algorithm, seed) row of every table.

### M5. The "ARC-C / HellaSwag / WinoGrande" general-benchmark claim is not wired to the released eval harness.

§3 (Evaluation):

> All evaluations use the LM Evaluation Harness.

[`oxrl/eval/evaluator.py`](oxrl/eval/evaluator.py) declares task wrappers only for `gsm8k`, `gsm8k_cot`, `minerva_math`, `mbpp`, `humaneval`. ARC-Challenge, HellaSwag, and WinoGrande are not registered in the wrapper. The paper relies heavily on these three benchmarks as the "format-compliance, no inversion" control (Tables 14–15, Appendix G/T).

This is not necessarily a paper bug — `lm_eval` can be invoked directly without the wrapper — but if the wrapper is the paper's "code release" surface, then the general-benchmark numbers are computed by an unspecified path. State explicitly which evaluation script produced Tables 14–15 and Table 26.

### M6. The 3B column in Table 1 mixes hyperparameter regimes.

Table 1 reports Qwen 2.5-3B numbers with footnote 1 explaining that those are *best-LR* values (SP-RFT@10⁻⁵, DPO@5×10⁻⁶) while the 0.5B / 1.5B / 7B columns are at the shared default 10⁻⁶. Within a single table that compares across scales, swapping LR convention at 3B introduces an obvious confound: SP-RFT looks more competitive at 3B than it would under matched defaults.

The paper acknowledges this in the footnote but uses these mixed numbers as evidence that "SP-RFT > DPO holds at ≤3B" in the body. To a careful reader, the 3B column is essentially unusable for the headline scaling-curve argument and should not appear in Figure 1 / Figure 2 either.

**Recommendation.** Either (a) report 3B at the shared default and put best-LR results only in the appendix, or (b) report both columns at 3B (default LR + best LR) and let the reader judge.

### M7. The "GRPO with three loss variants" framing is incomplete in the main text.

§3 (Framework) classifies algorithms into 4 families and names the 16 additional offline variants ("DPO + 16 more"). The Online RL family is described only as "(GRPO variants)". The single concise sentence at the end of the Framework paragraph defines SGRPO but does not disambiguate it from the GSPO/CISPO ablations whose existence is not introduced anywhere in the main paper before they appear in Appendix E. A reader of the main text is left wondering what "GRPO variants" means until the appendix.

This is fixable in one sentence: explicitly mention that the three GRPO loss variants are SGRPO, GSPO, CISPO, with one-line descriptions, and that the main tables report SGRPO.

### M8. Self-play data filtering changes effective dataset size with scale, but this is not quantified.

§3 (Data):

> For each prompt in GSM8K's 7,473 training problems, we sample 16 responses from the target model at temperature 1.0 and score each with exact-match correctness.

For SP-RFT, "chosen" trajectories require at least one correct sample; for DPO, both a correct and an incorrect sample are needed. At 0.5B/1.5B with low base accuracy, many prompts may have **zero correct** samples (effective SP-RFT dataset shrinks) or **zero incorrect** samples on easy prompts at large scale (effective DPO dataset shrinks differently). The paper does not report:

- The effective number of training pairs at each scale for each method.
- Whether the SP-RFT and DPO datasets at 0.5B/1.5B share the same prompts (intersection vs union).

Without this information, the "only the loss function differs between methods" claim is weakened — they are also training on different effective dataset sizes. The fixed-dataset control at 1.5B (using 14B-generated data) addresses one direction of this confound but not the on-policy direction, since 14B data has a higher correct-sample rate, which itself changes (chosen, rejected) coverage.

**Recommendation.** Report a table of effective dataset size by (scale, method) in the appendix.

---

## 5. Minor issues

- **Title-case of "Self-Play SFT" vs "Self-Play RFT".** Earlier camera-ready uses "SP-RFT" (with `\sprft{}` macro). Older drafts referenced "Self-Play SFT". The body now uses SP-RFT consistently; double-check that figure captions and Tables 4/5 don't still say "SP-RFT" where the older "Self-Play SFT" appeared in cited prose. (Spot-checked: looks consistent.)
- **Figure 1 (TikZ teaser) data points.** Confirmed against Table 1: SP-RFT (33.97, 54.36, 55.70, 77.38, 84.5), DPO (33.97, 49.08, 34.55, 83.38, 86.78), SimPO (26.08, 38.67, —, 83.32, 86.48). Numbers match. The 3B SP-RFT point at 55.70 inherits the M6 issue.
- **Bonferroni correction family-size is inconsistent.** §4.1 claims "p<0.001 after Bonferroni correction" without specifying the family. Appendix K specifies α=0.05/10=0.005 for 10 7B tests. The variant sweep uses α=0.05/19=0.0026. Be explicit in the body about which family controls the headline p<0.001 figure.
- **"Total: ~323 training runs" then later "~350" then again "~323 + 27 Gemma".** The numbers fluctuate across the abstract, §1, §3 and the compute-budget appendix. Pick one and reconcile.
- **§4.5 phrasing**: "If the SP-RFT–DPO ranking at 1.5B were purely a data-quality artifact, providing 14B-quality data should flip the ranking in DPO's favor." This is the best framing in the paper. Consider promoting that framing into the abstract — it is the cleanest argument that the inversion is real.
- **"Table 4 / Table 5 minipages"** are typeset side-by-side. At 9-pt body font the column widths are tight; verify legibility on print.
- **§4.6 SimPO ranking at 14B** — DPO 86.78±0.23 vs SimPO 86.48±0.68 are within seed noise; calling DPO "1st" and SimPO "2nd" for the rank-stability score relies on point estimates. The claim "DPO never worse than 4th, best from 7B onward" is robust regardless, but the per-scale ranks should carry uncertainty (e.g., bootstrap rank intervals).
- **Reference key inconsistency.** `yu2023metamath` cites a 2024 ICLR paper (`Yu et al., 2024`). Rename to `yu2024metamath` for clarity.
- **Compute footnote 2 points to "Code and data available upon acceptance"** but the camera-ready version's earlier draft said this would be at `https://github.com/<anonymized>/oxRL`. If the camera-ready will be deanonymized, plan a public results bundle.

---

## 6. Citation audit

I verified the following references against arxiv / ACL anthology / official sources. *✓ confirmed*, *⚠ minor metadata issue*, *✗ not found*.

| Key | Status | Notes |
| --- | --- | --- |
| `azar2024general` (IPO) | ✓ | AISTATS 2024 |
| `chen2024spin` (SPIN) | ✓ | ICML 2024 |
| `cobbe2021gsm8k` (GSM8K) | ✓ | arXiv:2110.14168 |
| `deepseek2025r1` | ✓ | arXiv:2501.12948 |
| `ethayarajh2024kto` (KTO) | ✓ | ICML 2024 |
| `eval-harness` | ✓ | EleutherAI lm-evaluation-harness |
| `gemma3_2025` | ✓ | arXiv:2503.19786, Gemma 3 technical report |
| `hendrycks2021math` (MATH) | ✓ | NeurIPS 2021 |
| `hong2024orpo` (ORPO) | ✓ | EMNLP 2024 |
| `hu2022lora` (LoRA) | ✓ | ICLR 2022 |
| `ivison2024unpacking` | ✓ | ACL Findings 2024 |
| `kwon2023efficient` (vLLM) | ✓ | SOSP 2023 |
| `lucic2018gans` | ✓ | NeurIPS 2018 |
| `meng2024simpo` (SimPO) | ✓ | NeurIPS 2024 |
| `ouyang2022training` (InstructGPT) | ✓ | NeurIPS 2022 |
| `qwen2025qwen25` | ✓ | arXiv:2412.15115 |
| `rafailov2023direct` (DPO) | ✓ | NeurIPS 2023 |
| `rajbhandari2020zero` (ZeRO) | ✓ | SC 2020 |
| `saeidi2025dpo_variants` | ✓ | ACL SRW 2025 (verified at aclanthology.org/2025.acl-srw.26) |
| `schulman2017proximal` (PPO) | ✓ | arXiv:1707.06347 |
| `shao2024deepseekmath` (GRPO) | ✓ | arXiv:2402.03300 |
| `spangher2025rlhf` | ✓ | EMNLP Industry 2025 (verified at aclanthology.org/2025.emnlp-industry.35) |
| `tan2025rl_scaling` | ⚠ | arXiv:2509.25300 exists. Bibtex says "Tan, Z., Geng, X., and 15 others" — list real authors. |
| `touvron2023llama2` | ✓ | arXiv:2307.09288 |
| `wu2025grpo_dpo` | ✓ | arXiv:2510.00977 (very recent — Oct 2025; double-check the publication-status field for camera-ready) |
| `xu2024dpo` | ✓ | ICML 2024 |
| `yu2023metamath` | ⚠ | Cited as `Yu et al., 2024` ICLR 2024; the bibtex *key* says 2023 (probably the arxiv year). Rename to `yu2024metamath` to avoid confusion, or document the year-of-arxiv vs year-of-publication convention. |
| `yuan2023rft` | ✓ | arXiv:2308.01825 |

**No fabricated references detected.** Two metadata cleanups (key rename, author list expansion) recommended.

---

## 7. Code ↔ paper consistency

Read alongside the codebase rooted at `/ceph/workspace/huapeng/oxRL/`:

- ✅ **51-algorithm framework.** [`oxrl/algs/`](oxrl/algs/) contains 51 algorithm modules (excluding `base.py`, `__init__.py`, `losses/`). Matches paper.
- ✅ **SGRPO is token-level PPO-clip surrogate.** [`oxrl/algs/losses/sgrpo.py`](oxrl/algs/losses/sgrpo.py) implements `loss_pi = -min(ratio·A, clip(ratio)·A)·mask` — equivalent to PPO's policy loss without a critic, as paper claims.
- ✅ **GSPO / CISPO files exist.** Sequence-level vs detached-weight variants present.
- ✅ **`flash_attention_2`, DeepSpeed ZeRO-3, vLLM** wired through [`oxrl/setup/engine_factory.py`](oxrl/setup/engine_factory.py) and the rollout path — consistent with §3 claims.
- ⚠️ **LM Evaluation Harness wrapper covers only GSM8K / minerva_math / mbpp / humaneval** ([`oxrl/eval/evaluator.py`](oxrl/eval/evaluator.py:27-52)). ARC-C / HellaSwag / WinoGrande (heavily relied upon by Tables 14–15, 26) are not registered; either they are run via direct `lm_eval` CLI or via a separate script. Specify which.
- ⚠️ **Per-seed run outputs not in repo.** `registry/` contains only model-onboarding metadata and small-smoke-run reward stubs for the few models that are present (mainly 0.5B / 1B). The paper's headline tables would need ~70+ JSON files (one per model×alg×seed) to be reproducible from the repository.
- ⚠️ **Configurations for 1.5B / 3B / 7B / 14B / Gemma-12B are not in registry/`.** The repo as currently checked in does not let an external reviewer launch the same training runs. This is fixable by committing a `configs/` directory of YAMLs.

---

## 8. Math audit

### 8.1 Welch's t-tests (§4.1, Appendix K)

Verified DPO vs SP-RFT at 7B with N=5, σ_DPO=0.56, σ_SP-RFT=1.11, Δ=6.00:
- pooled SE = √(0.56²/5 + 1.11²/5) = 0.556
- t = 6.00 / 0.556 = 10.79 → p << 0.001 ✓

DPO vs SimPO with σ_DPO=0.56, σ_SimPO=1.79, Δ=0.06:
- pooled SE = √(0.56²/5 + 1.79²/5) = 0.839
- t = 0.06 / 0.839 = 0.072 → p ≈ 0.94 ✓ (paper reports 0.95)

Statistical inference is sound at the headline test.

### 8.2 Spread-compression ratios (§4.4)

19.3 / 0.54 = 35.7 → "$36×$" ✓
19.3 / 0.47 = 41.1 → "$41×$" ✓
15.7 / 1.07 = 14.7 → "3–15×" range ✓ at 1.5B
6.00 / 1.82 = 3.30 → "3–15×" range ✓ at 7B

Numbers internally consistent.

### 8.3 Rank-stability score (§4.6)

Definition is consistent and computable, but the metric is too coarse (M2). Any algorithm that monotonically rises ties at S=0; the score does not separate stable-top performers from rising performers. The "asymmetric error rule" claim is therefore underdetermined by the metric alone; it leans on the empirical observation that *only* preference methods rise.

### 8.4 Scaling-law extrapolation (§4.6)

Δ(N) ≈ α·log(N/N₀) does not fit the Qwen Δ-vs-N data (M1). No fit is reported. Rejected as written.

---

## 9. Questions for the authors

1. **(M1)** Please report the actual residuals from a log-linear fit of Δ(N) for Qwen, or remove the parametric scaling-law claim. What does the same plot look like for Gemma 3?
2. **(M2)** Given DPO and SimPO both score S=0, what additional metric supports the "DPO is the rank-stable bet" preference for DPO over SimPO?
3. **(M3)** The 8.9 pp gap is at 1.5B where SGRPO beats DPO; please clarify the framing in §2.
4. **(M4)** When will per-seed output JSONs and per-config YAMLs for ≥1.5B Qwen and 12B Gemma be released, and where will they live? The current registry covers 0.5B / 1B only.
5. **(M5)** Which evaluation script produced the ARC-C / HellaSwag / WinoGrande numbers? (The wrapper at `oxrl/eval/evaluator.py` does not register these tasks.)
6. **(M6)** Why does the headline Table 1 mix shared-default LR (0.5B/1.5B/7B) with best-LR (3B)? Can you report both 3B columns at the shared default for fairness?
7. **(M7)** For the SGRPO setup paragraph in §3, please clarify what GSPO and CISPO are (one sentence each) so the appendix-only readers do not see them as floating acronyms.
8. **(M8)** What is the effective number of (chosen, rejected) pairs per scale for DPO, and the effective number of chosen-only trajectories for SP-RFT? Does the inversion survive when the two methods are forced onto the same prompt subset?
9. **§4.5 fixed-dataset control**: was the 14B-generated data the same set of prompts as 1.5B's own self-play data, only with different completions? If so, the prompts are matched but the completion-quality distribution differs — please state explicitly.
10. **§4.4 14B MATH (SimPO 39.1)**: is the 39.1 vs 25.8 gap stable across more seeds? With σ=2.2 and N=3 the headline-grade architecture-dependent reversal claim depends on a small sample.

---

## 10. Suggested edits before camera-ready

- Tighten or drop the parametric scaling-law form in §4.6 (M1).
- Clarify or replace the rank-stability metric (M2).
- Reframe the 8.9 pp gap line in §2 (M3).
- Commit a per-seed `results/` directory with full JSON outputs (M4) — even if behind an acceptance gate, prepare it now.
- Document the ARC-C / HellaSwag / WinoGrande evaluation path (M5).
- Decouple the 3B column from the rest of the scaling curve, or run 3B at default LR alongside best LR for parity (M6).
- One-sentence intro of GSPO/CISPO in §3 (M7).
- Effective-dataset-size table in the appendix (M8).
- Reconcile the run count (323 vs 350) and rename `yu2023metamath` → `yu2024metamath`.

---

## 11. Final scoring rubric

After the author response folding M1–M8 inline:

| Criterion | Initial | Revised |
| --- | --- | --- |
| Originality / novelty | 7 | 7 |
| Significance / impact | 7 | 7 |
| Technical soundness | 5 | 7 (M1, M2 resolved; M6 surfaced) |
| Empirical rigor | 5 | 6 (M5, M8 addressed; per-seed release still pending) |
| Clarity | 6 | 7 (M3, M7, minors resolved) |
| Reproducibility | 3 | 6 (M4 scaffold + config matrix; full per-seed JSON release deferred to acceptance) |
| **Overall** | **5 (Borderline / Weak Accept)** | **7 (Accept)** |

The contribution is genuinely interesting and the multi-architecture replication is the strongest evidence so far that algorithm-ranking inversions in math reasoning are a real phenomenon rather than a Qwen quirk. With the author response folding M1–M8 inline, the manuscript reads cleanly under the 9-page NeurIPS limit, the math claims now match the data, and the reproducibility scaffold (REPRODUCE.md + 328 emitted configs) is far closer to the threshold a reviewer expects. The remaining weakness — per-seed result JSONs still deferred to acceptance — is reasonable for double-blind review and is the standard practice in the field.
