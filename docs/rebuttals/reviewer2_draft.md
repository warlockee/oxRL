# Response to Reviewer 2 (working draft — ⏳ slots pending experiments)

## Global response

We thank the reviewer for an unusually careful reading — the summary is accurate, and we appreciate the recognition of the breadth of the evaluation, the practical importance of the question, and the value of the initialization analysis and controls. The review's central concern — that some claims are broader than the evidence — is fair, and we respond in two ways: with **evidence already in the submission that speaks directly to several concerns** (including the strict-vs-flexible analysis the reviewer requests), and with **new discussion-period experiments** that widen the controls in exactly the directions the reviewer identifies. We will also tighten claim scoping in the revision as detailed below.

**On reproducibility ("Partly"):** the artifact release exists and is substantial — 520 frozen per-run YAML configs, the derived datasets (~145k rows), per-seed checkpoints, one-command train/eval entry points, a Gebru-et-al. datasheet, and Croissant metadata. The submission deferred the link to acceptance for anonymity; we have now prepared an **anonymized review-time snapshot** [LINK — to be inserted after upload] so every number can be verified during the discussion. This also addresses the Dataset Assessment field: this is a D&B-track submission with released data and documentation.

## W1 / Q1 — "Mostly GSM8K; saturation; format sensitivity"

Three separate answers:

1. **GSM8K is not saturated at deployment scale under our protocol.** Base strict-match: 75.8 (7B), 79.5 (14B), 80.5 (32B — new run below); post-training reaches 87.4. The 7B inversion gap is 6.0 pp with p < 0.001 (N=5, Bonferroni) — an order of magnitude above seed noise, in the middle of the metric's dynamic range.
2. **The deployment-scale inversion is not a formatting artifact — the submission already tests this.** Appendix Table format_gap reports strict *and flexible-extract* accuracy: at 7B, DPO = 84.99 and SP-RFT = 80.29 under **flexible extraction** — the inversion persists (+4.7 pp) when answer formatting is removed from the metric entirely. We will surface this in the main text, since it directly rebuts the saturation/format concern.
3. **Beyond GSM8K, the key claims replicate where we have data.** On MATH: at Qwen-14B, SP-RFT is again last (23.0 vs DPO 25.8, SimPO 39.1); at Gemma-12B, DPO leads SP-RFT by 9 pp. The paper's own Finding 3 *agrees* with the reviewer about small scale — at ≤1.5B the differences are format compliance (that is the two-regime claim) — while MATH at ≥4B shows genuine, architecture-dependent reasoning differences. We will make the scoping explicit: the inversion claim is established on GSM8K+MATH for two families, not "mathematical reasoning" universally.

New 32B evidence (from the parallel discussion-period runs): base 80.52, SP-RFT 80.59, DPO 82.26, SimPO 84.61 (8-shot strict-match) — the small-scale ranking is fully reversed at 32B and SP-RFT's gain over base collapses to +0.07 pp, with the mechanism visible in the data pipeline (only 816/7,473 prompts still yield preference pairs).

## W3 / Q2 — Direct evidence for the format-compliance explanation

The submission contains the strict-vs-flexible comparison the reviewer requests (Appendix Table format_gap: per-algorithm strict and flexible accuracy at 1.5B/3B/7B). We agree aggregate deltas are not item-level evidence, so we have run a dedicated **item-level analysis** at 1.5B (5 algorithms × 3 seeds, per-item outputs logged under both 0-shot-CoT and 8-shot protocols): ⏳ [format-error rate per algorithm (flexible-correct ∧ strict-wrong); item-level solve-set overlap across algorithms; strict/flexible per-item agreement]. [Fill: table + 2-sentence conclusion.]

## W2 / Q3 — More model families

We agree two families cannot support a strong architecture-general claim, and the MATH reversal between Gemma-12B and Qwen-14B is exactly why the paper frames large-scale differences as *architecture-dependent* (Finding 3) — we will further soften any wording that suggests generality beyond the tested families. ⏳ [If Llama-3 third-family runs complete in time: report SP-RFT/DPO/SimPO at Llama-3.2-1B and Llama-3.1-8B, 3 seeds — does the small-scale SP-RFT lead and deployment-scale preference lead replicate?] [Else: state as camera-ready commitment.]

## W4 / Q4 — Toward a factorial control design

The reviewer is right that the existing controls probe one factor at a time. Two additions during the discussion period move toward the requested factorial:

1. **Teacher-data sweep at fixed learner scale (new).** Qwen-1.5B learner trained on self-play data from {0.5B, 1.5B (own), 7B, 14B} teachers × {SP-RFT, DPO} × 3 seeds: ⏳ [table]. Combined with the existing learner-scale axis at fixed data, this gives both margins of the (learner scale × data source) factorial for the two claim-bearing algorithms.
2. **Cross-family replication of the fixed-dataset control (new, run for Reviewer 1).** On Gemma-1B, strong-teacher (12B) data *flips* the SP-RFT/DPO ordering (own-data: SP-RFT 26.91 ± 0.55 > DPO 22.82 ± 2.08; 12B-data: DPO 24.69 ± 1.01 > SP-RFT 21.41 ± 1.01) — unlike Qwen, where the ordering is preserved. We report this honestly: the fixed-dataset conclusion is scoped to Qwen, and data provenance joins scale and architecture as a factor practitioners must validate under their own workflow. The revision's §4.7 is rescoped accordingly, and a full factorial across families is stated as the natural next step.

## Q5 — Would per-algorithm tuning change the rankings?

Evidence in and beyond the submission says no, within the explored ranges: (1) LR sweeps at 1.5B, 3B, and 7B show SP-RFT > DPO at *every* LR at small scale and DPO > SP-RFT at the per-algorithm optimal LRs at 7B; (2) β sweeps for the two strongest 1.5B variants (DPOP: 54.7/55.2/53.3 and ORPO: 54.8/54.4/53.8 at β ∈ {0.01, 0.1, 0.5}) move results ≤1.9 pp — far below the 6 pp inversion gap; (3) the shared-hyperparameter design is deliberate: it evaluates *plug-and-play transferability*, the setting in which practitioners actually deploy variants, and the revision states explicitly that per-variant-tuned capability ceilings are a complementary open question.

## Closing

The reviewer calls the paper "a useful intermediate empirical step" whose claims outrun its evidence. We believe the combination of (a) evidence already in the submission (flexible-extraction inversion, MATH replications, LR/β sweeps), (b) the new discussion-period experiments (32B, teacher-data sweep, item-level format analysis, cross-family control ⏳), and (c) the explicit claim-rescoping in the revision addresses each specific gap the review identifies. We would be grateful if the reviewer could indicate whether any specific analysis remains missing for the score to move.
