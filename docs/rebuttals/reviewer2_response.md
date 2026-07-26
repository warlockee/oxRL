# Response to Reviewer 2

## Global response

We thank the reviewer for an unusually careful reading — the summary is accurate, and we appreciate the recognition of the breadth of the evaluation, the practical importance of the question, and the value of the initialization analysis and controls. The review's central concern — that some claims are broader than the evidence — is fair, and we respond in two ways: with **evidence already in the submission that speaks directly to several concerns** (including the strict-vs-flexible analysis the reviewer requests), and with **new discussion-period experiments** that widen the controls in exactly the directions the reviewer identifies. We will also tighten claim scoping in the revision as detailed below.

**On reproducibility ("Partly"):** the artifact release exists and is substantial — 520 frozen per-run YAML configs, the derived datasets (~145k rows), per-seed checkpoints, one-command train/eval entry points, a Gebru-et-al. datasheet, and Croissant metadata. The submission deferred the link to acceptance for anonymity; we have now prepared an **anonymized review-time snapshot** [LINK — to be inserted after upload] so every number can be verified during the discussion. This also addresses the Dataset Assessment field: this is a D&B-track submission with released data and documentation.

## W1 / Q1 — "Mostly GSM8K; saturation; format sensitivity"

Three separate answers:

1. **GSM8K is not saturated at deployment scale under our protocol.** Base strict-match: 75.8 (7B), 79.5 (14B), 80.5 (32B — new run below); post-training reaches 87.4. The 7B inversion gap is 6.0 pp with p < 0.001 (N=5, Bonferroni) — an order of magnitude above seed noise, in the middle of the metric's dynamic range.
2. **The deployment-scale inversion is not a formatting artifact — the submission already tests this.** Appendix Table format_gap reports strict *and flexible-extract* accuracy: at 7B, DPO = 84.99 and SP-RFT = 80.29 under **flexible extraction** — the inversion persists (+4.7 pp) when answer formatting is removed from the metric entirely. We will surface this in the main text, since it directly rebuts the saturation/format concern.
3. **Beyond GSM8K, the key claims replicate where we have data.** On MATH: at Qwen-14B, SP-RFT is again last (23.0 vs DPO 25.8, SimPO 39.1); at Gemma-12B, DPO leads SP-RFT by 9 pp. The paper's own Finding 3 *agrees* with the reviewer about small scale — at ≤1.5B the differences are format compliance (that is the two-regime claim) — while MATH at ≥4B shows genuine, architecture-dependent reasoning differences. We will make the scoping explicit: the inversion claim is established on GSM8K+MATH for two families, not "mathematical reasoning" universally.

New 32B evidence (discussion-period runs, now N=3): base 80.52, SP-RFT 80.82 ± 0.23, DPO 82.23 ± 0.04, SimPO 83.42 ± 1.14 (8-shot strict-match) — the small-scale Qwen ranking is fully reversed at 32B, SP-RFT's gain over base is statistically zero (+0.30 ± 0.23), and the mechanism is visible in the data pipeline (only 816/7,473 prompts still yield preference pairs).

## W3 / Q2 — Direct evidence for the format-compliance explanation

The submission contains the strict-vs-flexible comparison the reviewer requests (Appendix Table format_gap: per-algorithm strict and flexible accuracy at 1.5B/3B/7B). We agree aggregate deltas are not item-level evidence, so we ran a dedicated **item-level analysis** at 1.5B (5 algorithms × 3 seeds, per-item outputs, both protocols). Under the 8-shot protocol:

| 1.5B | strict % | flexible % | format-err % (flex-correct ∧ strict-wrong) |
|---|---|---|---|
| SP-RFT | 57.11 ± 0.22 | 58.38 ± 0.42 | 1.26 ± 0.29 |
| IPO | 55.80 ± 0.46 | 57.64 ± 0.57 | 1.97 ± 0.08 |
| KTO | 54.79 ± 0.76 | 56.71 ± 0.84 | 2.10 ± 0.16 |
| DPO | 54.64 ± 1.73 | 56.91 ± 1.22 | 2.55 ± 0.66 |
| SimPO | 49.25 ± 1.49 | 54.61 ± 1.23 | 5.66 ± 0.34 |

Item-level findings: (i) format errors scale inversely with strict rank (SimPO's 5.7% format-error rate accounts for roughly half of its strict-match deficit); (ii) under 0-shot (no in-context format anchor) format-error rates grow to 7–14% and reorder the ranking — formatting dominates exactly when the protocol withholds format cues; (iii) strict solve-sets overlap heavily across algorithms (Jaccard 0.65–0.89), i.e., the methods largely solve the same items and differ at the margin. This is precisely the two-regime picture, now grounded at item level. As a bonus, this analysis required an independent end-to-end rerun of the full 1.5B grid on the current framework: it reproduces Table 1's ranking exactly (SP-RFT > IPO > KTO > DPO > SimPO), with a uniform ~3 pp offset from the newer inference stack. We will also annotate per-column evaluation protocols in Table 1 in the revision.

## W2 / Q3 — More model families

We agree two families cannot support a strong architecture-general claim, and the reviewer's instinct proves exactly right. We ran a **third family during the discussion period**: Llama-3.2-1B and Llama-3.1-8B, SP-RFT/DPO/SimPO, 3 seeds each, identical shared recipe and self-play protocol (GSM8K 8-shot strict-match):

| Llama 3 | 1B (base 36.69) | 8B (base 74.00) |
|---|---|---|
| SP-RFT | 37.15 ± 0.53 | **75.97 ± 0.72** |
| DPO | **38.29 ± 0.20** | 71.82 ± 1.03 |
| SimPO | 38.19 ± 0.63 | 66.87 ± 7.59 |

The result is a *third distinct pattern*: at 1B all methods gain ≤1.6 pp (no meaningful small-scale ranking exists); at 8B **the trend runs opposite to Qwen** — SP-RFT leads while both preference methods fall below base (DPO's training loss collapses to ~0.003, i.e., the shared plug-and-play recipe over-optimizes DPO on this family; SimPO reproduces its high-variance signature, σ = 7.59). We will revise the paper accordingly: the specific SP-RFT→preference inversion direction is scoped to Qwen and Gemma, and the general claim becomes the one this data actually supports — *scale-trends themselves are architecture-dependent, so no small-scale or cross-family result predicts deployment behavior*. This is a stronger practitioner warning than the original claim, and we thank the reviewer for forcing the test that revealed it.

## W4 / Q4 — Toward a factorial control design

The reviewer is right that the existing controls probe one factor at a time. Two additions during the discussion period move toward the requested factorial:

1. **Teacher-data sweep at fixed learner scale (new).** Qwen-1.5B learner trained on self-play data from {0.5B, 1.5B (own), 7B, 14B} teachers × {SP-RFT, DPO} × 3 seeds (GSM8K 8-shot strict-match):

   | teacher → | 0.5B | 1.5B (own) | 7B | 14B |
   |---|---|---|---|---|
   | SP-RFT | 57.19 ± 0.57 | 57.11 ± 0.22 | 58.15 ± 0.35 | 58.38 ± 0.49 |
   | DPO | 57.01 ± 1.05 | 54.64 ± 1.73 | 56.41 ± 0.27 | 53.32 ± 0.70 |

   (The 14B-data column is the submission's original control, evaluated on the April inference stack; the other columns are new runs on the current stack, which reads ~3 pp higher — within-column SP-RFT-vs-DPO comparisons are unaffected.) SP-RFT ≥ DPO at *every* teacher scale on Qwen — the small-scale SP-RFT lead is robust across the entire data-quality axis, completing the factorial margin the reviewer requested. A protocol contrast sharpens the mechanism: under 0-shot (no in-context format anchor), SP-RFT on weak-teacher (0.5B) data collapses to 9.4% while DPO retains 24.8% — SP-RFT inherits its teacher's formatting wholesale, while DPO's KL anchor preserves the initialization's format. Both observations reinforce that data provenance acts through format transmission at small scale.
2. **Cross-family replication of the fixed-dataset control (new, run for Reviewer 1).** On Gemma-1B, strong-teacher (12B) data *flips* the SP-RFT/DPO ordering (own-data: SP-RFT 26.91 ± 0.55 > DPO 22.82 ± 2.08; 12B-data: DPO 24.69 ± 1.01 > SP-RFT 21.41 ± 1.01) — unlike Qwen, where the ordering is preserved. We report this honestly: the fixed-dataset conclusion is scoped to Qwen, and data provenance joins scale and architecture as a factor practitioners must validate under their own workflow. The revision's §4.7 is rescoped accordingly, and a full factorial across families is stated as the natural next step.

## Q5 — Would per-algorithm tuning change the rankings?

Evidence in and beyond the submission says no, within the explored ranges: (1) LR sweeps at 1.5B, 3B, and 7B show SP-RFT > DPO at *every* LR at small scale and DPO > SP-RFT at the per-algorithm optimal LRs at 7B; (2) β sweeps for the two strongest 1.5B variants (DPOP: 54.7/55.2/53.3 and ORPO: 54.8/54.4/53.8 at β ∈ {0.01, 0.1, 0.5}) move results ≤1.9 pp — far below the 6 pp inversion gap; (3) the shared-hyperparameter design is deliberate: it evaluates *plug-and-play transferability*, the setting in which practitioners actually deploy variants, and the revision states explicitly that per-variant-tuned capability ceilings are a complementary open question.

## Closing

The reviewer calls the paper "a useful intermediate empirical step" whose claims outrun its evidence. We believe the combination of (a) evidence already in the submission (flexible-extraction inversion, MATH replications, LR/β sweeps), (b) the new discussion-period experiments (32B at N=3, the Llama-3 third family, the teacher-data sweep, the item-level format analysis, and the cross-family control), and (c) the explicit claim-rescoping in the revision addresses each specific gap the review identifies. We would be grateful if the reviewer could indicate whether any specific analysis remains missing for the score to move.
