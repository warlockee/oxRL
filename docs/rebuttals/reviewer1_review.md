# Reviewer 1 (Rating: 2 — Reject; Confidence: 4)

[Archived verbatim from OpenReview]

Summary: studies whether post-training algorithms (DPO, SimPO, IPO, SP-RFT) maintain consistent
rankings across scales; ~350 runs, Qwen2.5 0.5B-14B + Gemma3 1B-12B; rankings invert; SP-RFT
dominates <=1.5B, DPO leads at 7B; initialization dominates; scale is the main driver.

Weaknesses: (1) presentation — missing full names/citations in intro; abstract <=1.5B vs intro <=3B;
"capable base model" vs "useful one" unclear; "formatting capabilities" (S4.2) unclear; "Qwen 14B"
naming. (2) missing evidence — ARC-c results claimed but not found in main text; Table 1 SGRPO worse
than base and IPO>SP-RFT unexplained; S4.4 format-compliance claim without analysis. (3) 14B not
large — run 32B; involve GRPO; fixed-dataset control only on Qwen2.5-1.5B.

Questions: Q1 missing values in Table 1; Q2 will inversion hold for Qwen3; Q3 why 5 vs 3 seeds;
Q4 zero variance at 0.5B and no variance for base.

Reproducibility: Yes. Dataset Assessment: Partly (responsible AI fields missing).
Ethics: no/very minor.