# Follow-up comment for Reviewer 1 (paste-ready, 32B seeds complete)

As promised in our response, the additional 32B seeds have completed. All Qwen2.5-32B cells are now **N=3** (GSM8K 8-shot strict-match, seeds 42/123/456):

| Qwen2.5-32B | Mean ± σ (%) | Δ vs Base |
|---|---|---|
| Base (Instruct) | 80.52 | — |
| SP-RFT | 80.82 ± 0.23 | +0.30 |
| DPO | 82.23 ± 0.04 | +1.71 |
| **SimPO** | **83.42 ± 1.14** | **+2.90** |

The single-seed ordering reported in our response is unchanged and now seed-robust: preference methods > SP-RFT ≈ base in every seed, with DPO > SP-RFT significant at p < 0.01 (Welch). Two auxiliary observations: SP-RFT's gain over base is statistically zero (+0.30 ± 0.23), completing the RFT-ceiling trajectory (+23.2 pp at 1.5B → +1.6 at 7B → +0.3 at 32B); and the per-method seed variances reproduce the paper's variance signatures at a new scale (DPO σ = 0.04, the most stable method; SimPO σ = 1.14, the least). The revised manuscript's 32B table now reports these N=3 values, and per-seed evaluation JSONs are included in the released artifacts.
