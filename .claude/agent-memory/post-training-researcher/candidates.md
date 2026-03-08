# Candidate Methods for Future Onboarding

## High Priority
- **RLOO (REINFORCE Leave-One-Out)**: ACL 2024. RL method simpler than PPO. Would need RL integration (main_rl.py).
- ~~LRPO / SamPO~~: Onboarded (EMNLP 2024, down-sampled KL divergence)

## Medium Priority
- **MPO (Mixed Preference Optimization)**: Combines DPO + BCO + SFT losses. Multi-loss approach. TRL supports loss_type as list.
- **DAPO**: ByteDance RL method (2025). Clip-Higher + Dynamic Sampling + Token-Level PG. Needs RL integration.
- **TDPO**: Token-level DPO (ICML 2024). Per-token forward KL divergence -- complex, modifies loss fundamentally.
- ~~SamPO~~ (EMNLP 2024): Onboarded (down-sampled KL divergence, stochastic token sampling)
- **MallowsPO** (ICLR 2025): Dispersion-scaled DPO. Needs per-prompt entropy pre-computation, not just a loss mod.
- **LMPO** (2025): Length-controlled margin-based PO. Needs probability (not logprob), Z-score normalization.
- **LPO** (2025): Linear PO with absolute diff + STE. Requires stop-gradient detach ops, complex.
- **alpha-DPO** (ICLR 2025): Adaptive margin with per-sample Z-score. Moderate complexity.
- **TIS-DPO** (ICLR 2025): Token importance sampling. Needs contrastive LLM pairs for weight estimation.

## Lower Priority (need online generation or external data)
- **DNO**: Direct Nash Optimization -- needs online generation + GPT-4 as oracle.
- **HyPO**: Hybrid PO -- needs online samples for KL regularization.
- **D2PO**: Discriminator-guided DPO -- needs online collection + discriminator.
- **REBEL**: Regressing relative rewards -- needs iterative data collection.
- **LiPO-lambda**: Listwise ranking -- needs K>2 ranked responses (not pairwise).
- **sDPO**: Stepwise DPO -- data scheduling (not a loss variant), updates ref model between steps.

## Completed (moved from candidate list)
- ~~cDPO~~: Onboarded
- ~~Cal-DPO~~: Onboarded
- ~~beta-DPO~~: Onboarded
- ~~SPPO~~: Onboarded
- ~~NCA~~: Onboarded (pairwise case; InfoNCA = DPO for K=2)
- ~~AOT~~: Onboarded
- ~~APO~~: Onboarded (both zero and down modes)
- ~~Hinge/SLiC~~: Onboarded
- ~~Robust DPO~~: Onboarded (unbiased with 1/(1-2*eps) normalization)
- ~~EXO-pair~~: Onboarded (reverse KL preference optimization)
- ~~DiscoPOP~~: Onboarded (log-ratio modulated LRML loss)
- ~~BCO~~: Onboarded (binary classifier pairwise optimization)
- ~~ODPO~~: Onboarded (DPO with offset for preference strength)
- ~~DPOP~~: Onboarded (DPO-Positive / Smaug penalty)
- ~~FocalPO~~: Onboarded (focal weighting on DPO loss)
- ~~GPO~~: Onboarded (exponential, truncated_quadratic, savage losses from ICML 2024)
- ~~WPO~~: Onboarded (on-policy importance weighting, EMNLP 2024)
- ~~f-DPO~~: Onboarded (forward KL, JS divergence, alpha divergence, ICLR 2024)
- ~~H-DPO~~: Onboarded (entropy controllable DPO via alpha coefficient, 2024)
- ~~DPO-Shift~~: Onboarded (shifted rejected term for likelihood displacement, 2025)
- ~~CPO-SimPO~~: Onboarded (CPO + SimPO combined, reference-free, 2024)
- ~~Dr-DPO~~: Onboarded (Distributionally Robust DPO, ICLR 2025, log-sum-exp aggregation)
- ~~Chi-PO~~: Onboarded (Chi-squared pref opt, ICLR 2025, phi(z)=z+log(z) link function)
- ~~SPO~~: Onboarded (Self Pref Opt, EMNLP 2025, SiLU replaces logsigmoid for bounded loss)
- ~~DPNLL~~: Onboarded (DPO+NLL, DPO with NLL regularization on chosen responses)
- ~~MinorDPO~~: Onboarded (arXiv 2408.09834, clamped rejected log-ratio for robustness)
- ~~C2DPO~~: Onboarded (arXiv 2502.17507, constrained DPO with L2 penalty on sum of log-ratios)
- ~~GPO~~: Onboarded (exponential, truncated_quadratic, savage losses from ICML 2024)
- ~~WPO~~: Onboarded (on-policy importance weighting, EMNLP 2024)

## Relationship Notes
- SPPO loss = Cal-DPO calibration loss (same formula, different theoretical motivation)
- NCA pairwise differs from InfoNCA pairwise: InfoNCA = DPO for K=2, but NCA != DPO
- APO-zero increases chosen + decreases rejected; APO-down decreases both (rejected more)
- Robust DPO vs cDPO: same numerator, robust adds 1/(1-2*eps) normalization
- GPO logistic loss = DPO; GPO hinge = SLiC; GPO adds exponential/truncated_quadratic/savage
- f-DPO reverse_kl = DPO; f-DPO forward_kl is mode-covering; f-DPO JS is balanced
- WPO reduces to DPO when all data is on-policy (weights=1)
- TRL DPO trainer supports: sigmoid, hinge, ipo, exo_pair, nca_pair, robust, bco_pair, sppo_hard, aot, aot_unpaired, apo_zero, apo_down, discopop, sft
- ALL TRL DPO loss_types are now covered by oxRL implementations
- TRL CPO trainer supports: sigmoid, hinge, ipo, simpo, alphapo -- ALL covered by oxRL
- CPO-SimPO is reference-free; reuses simpo_gamma config field from SimPO
- DPO-Shift: shift_lambda=1 reduces to DPO; <1 reduces rejected influence
- R-DPO and iLR-DPO have equivalent loss formulas (same alpha * length_diff term)
- SamPO: When len_chosen == len_rejected, reduces to standard DPO. Uses stochastic downsampling.
- SamPO eval uses deterministic non-sampled DPO for stable metrics
- Dr-DPO: -bp*log(mean(exp(-L/bp))) aggregation; bp->inf = DPO, bp->0 = min loss (worst-case focus)
- Dr-DPO: Single sample always equals DPO regardless of beta_prime
- Chi-PO: When logr_w == logr_l (symmetric), phi_w - phi_l = 0, same as DPO
- Chi-PO: No new hyperparameters; same init as DPO. Uses torch.clamp on logr to prevent exp overflow
- SPO: SiLU(x)=x*sigmoid(x) >= logsigmoid(x), so -SiLU <= -logsigmoid (SPO loss <= DPO loss)
- SPO: -SiLU has bounded minimum ~0.278 (self-regularizing), but can go negative for large positive x
- SPO: No new hyperparameters; identical interface to DPO. One-line change: F.logsigmoid -> F.silu
- DPNLL: DPO + alpha*NLL_chosen; combines DPO reference-model loss with CPO-style BC regularization
- DPNLL: alpha=0 reduces to DPO; returns 5 values (total, dpo, nll, margin, reward_acc)
- DPNLL: Uses forward_ref (logps only) and forward (logps + raw logits) for efficiency
- MinorDPO: clamp(logr_l, min=0) stops rejected penalty once pi_l < pi_ref_l
- MinorDPO: No new hyperparameters; same interface as DPO. When logr_l >= 0, identical to DPO
- MinorDPO: When logr_l < 0, loss >= DPO loss; no gradient flows through pi_logps_l
- C2DPO: Constraint = (logr_w + logr_l)^2; penalizes total deviation from reference
- C2DPO: lambda=0 reduces to DPO; default lambda=2e-4 (from paper)
- C2DPO: When logr_w + logr_l = 0, constraint is zero (balanced deviations are OK)
