# Reviewer 2 (Rating: 3 — Borderline reject)

[Archived verbatim from OpenReview]

Summary: rankings transfer/invert with scale on math reasoning; three findings (inversion from
Instruct; initialization explains most variance; two regimes). "Useful intermediate empirical step
... current evidence not yet sufficient to support the paper's broad publication-level claims."

Strengths: broad careful evaluation (~323 Qwen + 27 Gemma runs); practically important question;
Base-vs-Instruct insight; fixed-dataset + gold-SFT controls address natural alternatives.

Weaknesses: (1) claims broader than evidence — GSM8K-centric; possible saturation at 7B/14B;
exact-match format sensitivity. (2) two families insufficient for architecture-level claims; MATH
winner differs sharply between families. (3) format-compliance explanation under-analyzed — wants
item-level error analysis, format-error rates, strict-vs-flexible, difficulty controls.
(4) controls not factorial — one factor at a time, limited scale; wants full (learner scale x data
source x algorithm) x families x benchmarks.

Reproducibility: Partly — code/configs/outputs not accessible during review.
Dataset Assessment: NA — no dataset included. Ethics: no/very minor.

Questions: Q1 GSM8K saturated at 7B/14B? more benchmarks. Q2 direct evidence for format-compliance
(item-level, format-error rates, strict-vs-flexible). Q3 more model families. Q4 expand fixed-dataset
control (learner scale x data source x algorithm). Q5 would per-algorithm tuning change rankings?