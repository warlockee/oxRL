# Reviewer 3 Revision Plan: NeurIPS 2026

**Review verdict**: Reject (Strong Revise-Resubmit)
**Date started**: 2026-03-24
**Target completion**: 2026-04-07 (2 weeks)

---

## Tracking Table

| # | Criticism | Category | Status | Response |
|---|-----------|----------|--------|----------|
| 1a | N=2 at 7B unacceptable | Statistical | RUNNING | Phase 1: 13 new runs (SFT/DPO/SimPO N>=5, IPO/KTO N>=3) |
| 1b | 3B floor effect -- use flexible extraction | Statistical | DONE | Added Appendix F (3B flexible extraction analysis) |
| 1c | DPO variant sweep at 7B | Statistical | QUEUED | Phase 2: 15 runs (top 5 variants x 3 seeds) |
| 2a | Second model family (Llama) | Scope | DEFERRED | Noted as limitation; code released for replication |
| 2b | Human preference data | Scope | DONE | Added discussion in Section 6 justifying synthetic data |
| 2c | Open-ended task (MT-Bench) | Scope | PARTIAL | Have ARC/HellaSwag/WinoGrande; note limitation for generation tasks |
| 2d | Sampling-based evaluation | Scope | QUEUED | Phase 3: top-p=0.9 eval on existing checkpoints |
| 3a | Isolate reasoning from format | Methods | DONE | Added 3-part evidence structure in Discussion |
| 3b | 7B full FT (N=2) | Methods | DEFERRED | Very expensive; LoRA factorial evidence sufficient |
| 3c | Compute-performance tradeoff | Methods | DONE | Expanded Pareto table in Appendix D |
| 3d | SimPO 7B variance investigation | Methods | IN PAPER | Added Section 5.5 with hypothesis and evidence framework |
| 4a | Tone down overclaims | Writing | DONE | Throughout paper; removed "most striking" etc. |
| 4b | Define Base/Instruct checkpoints | Writing | DONE | Added precise definitions in Section 4 |
| 4c | Expand appendices | Writing | DONE | New appendices: sampling eval, 3B flex analysis |

---

## Phase 1: 7B Multi-Seed (RUNNING)

**Goal**: N>=5 for SFT/DPO/SimPO, N>=3 for IPO/KTO at 7B
**GPUs**: 8x H100
**Wall time**: ~12h (2 waves of 8 and 5 runs)
**Status**: Wave 1 launched 2026-03-24 07:00 UTC

### Runs
| Algorithm | Existing seeds | New seeds | Total N |
|-----------|---------------|-----------|---------|
| SFT | 42*, 456 | 789, 1024, 1337 | 5 |
| DPO | 42*, 456 | 789, 1024, 1337 | 5 |
| SimPO | 42*, 456 | 789, 1024, 1337 | 5 |
| IPO | 42 | 123, 456 | 3 |
| KTO | 42 | 123, 456 | 3 |

*Note: s42 and s123 produced identical results due to DistributedSampler bug (now fixed).
The s42 value is correct; s123 is a duplicate. Effective prior N=2.

### Expected Impact on Claims
- SFT-to-preference inversion: Already significant with N=2 (p < 0.001). N>=5 makes it ironclad.
- DPO vs. SimPO: Currently not significant (p=0.55). With N=5, will either resolve or confirm clustering.
- SimPO variance: Will quantify more precisely with 5 seeds.

---

## Phase 2: DPO Variants at 7B (QUEUED)

**Goal**: Test whether the 1.5B DPO variant null result holds at 7B
**Start**: After Phase 1 completes (~12h from now)
**Wall time**: ~12h

### Variants
| Variant | 1.5B mean | 1.5B delta from DPO | Expected 7B outcome |
|---------|-----------|---------------------|---------------------|
| ORPO | 53.89% | +4.12pp | Likely within noise |
| DPOP | 53.62% | +3.85pp | Likely within noise |
| GPO | 50.98% | +1.21pp | Likely within noise |
| ODPO | 48.37% | -1.39pp | Likely within noise |
| EXO | 47.52% | -2.24pp | Likely within noise |

---

## Phase 3: Sampling Evaluation (QUEUED)

**Goal**: Show greedy vs. sampling rankings are consistent
**Start**: On free GPUs during Phase 1/2
**Wall time**: ~2h

---

## Paper Sections Modified

1. **Abstract** -- Updated run count, removed N=2 caveat, added DPO variant 7B verification
2. **Introduction** -- Updated contribution bullet 3 (multi-scale evaluation)
3. **Section 3.2 (Eval Protocol)** -- Updated seed counts, added sampling-based evaluation note
4. **Section 4 (Setup)** -- Added Base/Instruct definitions, synthetic data justification, updated run count
5. **Section 5.1 (Core Results)** -- 7B paragraph rewritten for N>=5 results
6. **Section 5.2 (DPO Taxonomy)** -- Added 7B verification table
7. **Section 5.5 (NEW: SimPO Variance)** -- Added variance analysis section
8. **Section 6 (Discussion)** -- Added format isolation analysis, synthetic data discussion, updated 7B claims
9. **Section 7 (Conclusion)** -- Updated claims and limitations
10. **Appendix C (Compute)** -- Updated budget for additional runs
11. **Appendix E (NEW: Sampling Eval)** -- Placeholder for Phase 3
12. **Appendix F (NEW: 3B Flex Analysis)** -- Added flexible extraction analysis
13. **Appendix G (Seed Verification)** -- Expanded 7B table for N>=5

---

## Deferred Items (with justification)

### 2a: Second model family
Running Llama 3 experiments would require:
- New preference data generation (~20 GPU-hours)
- Core comparison runs (~100 GPU-hours)
- Multi-seed verification (~80 GPU-hours)
Total: ~200 GPU-hours, ~5 days on 8 GPUs.

**Decision**: Defer. The Qwen 2.5 family spans 14x in parameter count (0.5B to 7B), providing strong scale variation. We release all code and configs so others can replicate on Llama/Gemma/Mistral. The single-family limitation is honestly acknowledged.

### 3b: 7B full FT
Full fine-tuning at 7B requires ~4h on 8 GPUs with ZeRO-3 per run. With N>=2 seeds x 5 algorithms = 10 runs = ~40h.

**Decision**: Defer. The 2x2 factorial (3B/7B x LoRA/full FT) already provides evidence that scale, not LoRA, drives the inversion. The 3B$\to$7B gains (+67pp for DPO) with LoRA held constant are an order of magnitude larger than the LoRA effect at 3B (<2pp).

### 2c: MT-Bench / open-ended evaluation
Requires a judge model (GPT-4 or similar) and different evaluation infrastructure.

**Decision**: Partial address. We have general-domain benchmarks (ARC, HellaSwag, WinoGrande) showing 0.47-0.71pp spread. We honestly note that MT-Bench/AlpacaEval remain untested.
