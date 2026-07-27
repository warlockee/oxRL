# Rebuttals — NeurIPS 2026 discussion period

One directory per review round. Naming convention: `reviewerN_response.md` is the
OpenReview paste (hard limit: **10,000 characters per rebuttal** — verify with
`wc -m` before posting); `reviewerN_full.md` is the extended reference version
with everything that didn't fit. `results_appendix.md` holds the per-seed numbers
for all discussion-period experiments and is **shared across reviewers** — extend
it, don't fork it.

| File | Purpose | Status |
|---|---|---|
| `reviewer1_response.md` | Reviewer 1 (rating 2): presentation, ARC-C, 32B N=3, GRPO, controls, zero-variance | ready to post (9,983 chars; insert anon link) |
| `reviewer1_full.md` | Extended version | reference |
| `reviewer2_response.md` | Reviewer 2 (rating 3): scope, format analysis, factorial, 3rd family | ready to post (9,583 chars; insert anon link) |
| `reviewer2_draft.md` | Same content, working copy | reference |
| `results_appendix.md` | Per-seed results: 0.5B rerun, Gemma control, 3B fill, 32B N=3, Llama-3, factorial, item-level | complete |

Supporting artifacts:
- Raw eval JSONs: `oxrl_results/eval_r6_*` on FSx (release with artifacts).
- Manuscript fixes for reviewer 1 are applied in `docs/oxrl_formal.tex`
  (commit `ee6a12c`): revised Table 1 (0.5B multi-seed column, SGRPO (GRPO)),
  abstract ≤3B, naming, and the "Discussion-Period Additions" appendix.
- 32B run configs: session scratchpad `qwen32b_configs/` (r6_{sft,dpo,simpo}_qwen32b_gsm8k_s{42,123,456});
  DPO at 32B uses the precomputed ref-logp cache (oxRL commit `a46f308`).
