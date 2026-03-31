# Reviewer 5 Revision Plan: NeurIPS 2026

**Review verdict**: Reject (Revise-Resubmit with specific roadmap)
**Date**: 2026-03-28
**Target completion**: 2026-04-18 (21 days, leaving 18 days buffer before May 6 deadline)
**Resources**: 8x H100 80GB, ~39 days to deadline

---

## Executive Summary

Reviewer 5's criticisms are the most existential: they demand **structurally new experiments** (different architectures, larger scales, human preference data). The word "fatal" appears in criticism #1.

**Key strategic insight**: Using **Gemma 3** (March 2025) instead of Llama 3 (April 2024) kills multiple birds:
- Gemma 3 1B / 4B / 12B maps cleanly to Qwen 0.5-1.5B / 3B / 7B
- 12B simultaneously addresses criticism #3 (>7B scale)
- March 2025 release = maximally current for NeurIPS 2026
- oxRL already has Gemma 3 1B and 4B recipes registered
- Different architecture (no GQA at 1B, different tokenizer, different training mix)

---

## Phase 1: Gemma 3 Cross-Architecture Validation (CRITICAL)
**Addresses**: Criticism #1 (fatal gap) + Criticism #3 (>7B scale)
**Timeline**: Days 1-10
**GPU allocation**: 4-8 GPUs

### Models
| Model | Params | Maps to Qwen | FT Method | HuggingFace ID |
|-------|--------|-------------|-----------|----------------|
| Gemma 3 1B-IT | 1B | 0.5B/1.5B | Full FT | google/gemma-3-1b-it |
| Gemma 3 4B-IT | 4B | 3B | Full FT | google/gemma-3-4b-it |
| Gemma 3 12B-IT | 12B | 7B (+ >7B!) | LoRA r=16 | google/gemma-3-12b-it |

### Experiment Design (core 3 algorithms: SFT, DPO, SimPO)

| Model | Algorithms | FT Method | Seeds | Est. GPU-h/run | Runs |
|-------|-----------|-----------|-------|----------------|------|
| Gemma 3 1B-IT | SFT, DPO, SimPO | Full FT | 3 | ~0.5 | 9 |
| Gemma 3 4B-IT | SFT, DPO, SimPO | Full FT | 3 | ~2 | 9 |
| Gemma 3 12B-IT | SFT, DPO, SimPO | LoRA r=16 | 3 | ~10 | 9 |
| **Subtotal** | | | | | **27 runs, ~115 GPU-h** |

### Base Initialization Ablation (1 seed each)

| Model | Init | Algorithms | Runs |
|-------|------|-----------|------|
| Gemma 3 1B (Base) | Base | SFT, DPO, SimPO | 3 |
| Gemma 3 12B (Base) | Base | SFT, DPO, SimPO | 3 |
| **Subtotal** | | | **6 runs, ~30 GPU-h** |

### Data Pipeline
1. Sample 16 responses per GSM8K training prompt from each Gemma model (vLLM, temp=1.0)
2. Score with exact-match reward (same as Qwen pipeline)
3. Select chosen/rejected pairs
4. ~2-4 GPU-h per model for rollout generation

### What We Need to Show
Cross-architecture validation succeeds if we observe **any two** of:
1. **Initialization dominance**: from Base init, inter-algorithm spread collapses
2. **Scale-dependent inversion**: SFT leads at 1B/4B but DPO/SimPO lead at 12B
3. **DPO/SimPO clustering at large scale**: at 12B, DPO and SimPO produce similar accuracy

**Phase 1 total: 33 runs + data gen, ~145 GPU-h, ~5 days wall time**

### Bonus: 12B addresses >7B concern
Gemma 3 12B is genuinely larger than Qwen 7B. If DPO/SimPO > SFT holds at 12B on a different architecture, this simultaneously resolves:
- Criticism #1 (cross-architecture generalization)
- Criticism #3 (deployment-scale >7B evaluation)

---

## Phase 2: 7B Full Fine-Tuning (CRITICAL — resolves LoRA confound)
**Addresses**: Criticism #2
**Timeline**: Days 3-8 (overlaps with Phase 1)
**GPU allocation**: 8 GPUs (ZeRO-3)

### Experiment Design (N=1 seed, as reviewer suggests)

| Model | Algorithm | FT Method | GPUs | Est. time |
|-------|-----------|-----------|------|-----------|
| Qwen2.5-7B-Instruct | SFT | Full FT (ZeRO-3) | 8 | ~8h |
| Qwen2.5-7B-Instruct | DPO | Full FT (ZeRO-3) | 8 | ~10h |
| Qwen2.5-7B-Instruct | SimPO | Full FT (ZeRO-3) | 8 | ~8h |

**Total: 3 runs, ~54 GPU-h, ~26h wall time (sequential on 8 GPUs)**

### What We Need to Show
- If DPO > SFT under full FT → LoRA is ruled out as driver. Inversion is scale-driven.
- If DPO ≈ SFT under full FT → LoRA confound is real (still publishable: "inversion is LoRA-mediated")

### Contingency
If OOM: use gradient checkpointing + CPU optimizer offload. If still fails: run DPO + SFT only (the critical comparison).

---

## Phase 3: Qwen 14B Deployment Scale (HIGH)
**Addresses**: Criticism #3 (reinforces Gemma 12B evidence)
**Timeline**: Days 8-15
**GPU allocation**: 2-3 GPUs

**Note**: If Gemma 3 12B results are strong, this phase becomes reinforcement rather than critical. Can be deprioritized if time is tight.

| Model | Algorithms | FT Method | Seeds | Est. time | Runs |
|-------|-----------|-----------|-------|-----------|------|
| Qwen2.5-14B-Instruct | SFT, DPO, SimPO | LoRA r=16 | 3 | ~12h each | 9 |

**Total: 9 runs, ~126 GPU-h, ~2.5 days on 3 GPUs**

If GSM8K ceiling (>90% base): shift primary eval to MATH.

---

## Phase 4: UltraFeedback Human Preference Exploratory (MEDIUM)
**Addresses**: Criticism #4
**Timeline**: Days 10-14
**GPU allocation**: 1-2 GPUs

| Model | Algorithms | Dataset | Seeds | Est. time | Runs |
|-------|-----------|---------|-------|-----------|------|
| Qwen2.5-1.5B-Instruct | SFT, DPO, SimPO | UltraFeedback | 3 | ~20 min | 9 |

**Total: 9 runs, ~3 GPU-h (trivially cheap)**

Evaluate with MT-Bench or AlpacaEval for instruction-following quality.

Either outcome is publishable:
- Init dominance holds → strengthens core claim
- Init dominance fails → interesting boundary condition for verifiable tasks

---

## Phase 5: Minor Fixes (LOW)
**Addresses**: Criticism #5
**Timeline**: Days 14-20

| Fix | GPU-h | Description |
|-----|-------|-------------|
| 5a: DPO/SimPO LR sweep 7B | ~36 | 3 LR values × 2 algorithms × 1 seed |
| 5b: MATH error analysis | 0 | Manual/GPT-4 analysis of wrong answers |
| 5c: Table simplification | 0 | Split dense tables, reorganize infrastructure section |
| 5d: 7B MATH eval | ~1 | Eval-only on existing checkpoints |

---

## Total Resource Budget

| Metric | Value |
|--------|-------|
| New training runs | ~90 |
| New GPU-hours | ~365 |
| Wall time (8x H100) | ~7-10 days |
| Buffer before deadline | 18+ days |

---

## Master Timeline

```
WEEK 1 (Mar 28 - Apr 3): PHASE 1 + PHASE 2
================================================================
Day 1-2: Generate Gemma preference data + launch 1B/4B runs
Day 3-4: Launch Gemma 12B runs (wave 1)
Day 3:   Start Qwen 7B Full FT - SFT (overnight, 8 GPUs)
Day 4:   Qwen 7B Full FT - DPO
Day 5:   Qwen 7B Full FT - SimPO
Day 6-7: Gemma 12B remaining + evaluate all Phase 1/2

WEEK 2 (Apr 4 - Apr 10): PHASE 3 + PHASE 4
================================================================
Day 8:   Generate Qwen 14B preference data
Day 8:   Launch UltraFeedback runs (done in hours)
Day 9-13: Qwen 14B runs (3 parallel on 3 GPUs)
Day 10-13: Phase 5a LR sweep (parallel on spare GPUs)

WEEK 3 (Apr 11 - Apr 18): ANALYSIS + WRITING
================================================================
Day 14-16: Phase 5b/5c/5d, compile all results
Day 17-21: Paper revision with new sections + tables

BUFFER (Apr 19 - May 6): 17 DAYS
================================================================
Reruns, polish, anti-AI sweep, final submission
```

---

## Tracking Table

| # | Criticism | Phase | Status | Priority | Est. GPU-h |
|---|-----------|-------|--------|----------|-----------|
| 1 | Cross-architecture (Gemma 3) | Phase 1 | NOT STARTED | CRITICAL | 145 |
| 2 | LoRA confound (7B full FT) | Phase 2 | NOT STARTED | CRITICAL | 54 |
| 3 | >7B scale (Qwen 14B) | Phase 3 | NOT STARTED | HIGH | 126 |
| 4 | Human preference data | Phase 4 | NOT STARTED | MEDIUM | 3 |
| 5a | LR sweep DPO/SimPO 7B | Phase 5 | NOT STARTED | LOW | 36 |
| 5b | MATH error analysis | Phase 5 | NOT STARTED | LOW | 0 |
| 5c | Table simplification | Phase 5 | NOT STARTED | LOW | 0 |
| 5d | 7B MATH evaluation | Phase 5 | NOT STARTED | LOW | 1 |

---

## Models to Download

```
google/gemma-3-1b-it          (already in oxRL registry)
google/gemma-3-1b-pt          (Base, for init ablation)
google/gemma-3-4b-it          (already in oxRL registry)
google/gemma-3-12b-it         (need to add to registry)
google/gemma-3-12b-pt         (Base, for init ablation)
Qwen/Qwen2.5-14B-Instruct    (for Phase 3)
openbmb/UltraFeedback         (dataset, for Phase 4)
```
