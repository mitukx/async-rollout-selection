# Async Rollout Selection

A small, auditable starting point for studying **completion-dependent sampling in asynchronous policy optimization**. The current result concerns a *single fixed-policy update with deadline-based inclusion*, not a full asynchronous training convergence theorem.

## Verified baseline

With a two-action logistic policy, deterministic rewards, and action-dependent completion probabilities, training on completed rollouts can **reverse the expected REINFORCE update direction**. In a concrete example, the true gradient is `-0.2`, while the completed-only expected score is `+0.045455`. Known, strictly positive inclusion probabilities admit an unbiased per-launch inverse-probability-weighted estimator, with an exact variance formula. If completion depends on the *unobserved reward*, the original gradient need not be identifiable—even with strictly positive completion probabilities.

See [theory.md](theory.md) for the initial results and [audit_theory.md](audit_theory.md) for sharp identification intervals, follow-up-audit estimators, exact variance and bounded-budget audit allocation. Those results are classical statistical baselines, **not new frontier theory**. These are elementary selection/missing-data results, **not claimed as novel research theorems**.

```bash
python selection.py
python audit.py
python -m unittest discover -s tests -v
```

Python standard library only. The synthetic experiment is a check of the formulas, not evidence of large-model training gains.

## Research question (open within this project)

When inclusion depends on evolving queue state, policy version, and trajectory-dependent completion time, what can be identified and estimated from **logged launches, completions, and a limited independent audit budget**? The intended target is a *wall-clock-constrained* gradient error bound or lower bound that does not follow immediately from classical inverse-probability weighting. This question is a proposal, **not a certified open problem**.

## Prior work / differentiation

- Dong, Goel & Yu (2026), [*Predicting and Controlling Staleness in Fully Asynchronous RL Training*](https://www.appliedcompute.com/research/staleness-in-fully-async-rl): analyzes queue staleness and notes `queue-max` response-length selection bias; therefore, observing length bias is **not** a contribution here.
- He et al. (2026), [*Asynchronous LLM Post-Training: Group-Mass Capping and Convergence Analysis*](https://arxiv.org/abs/2610.01896): analyzes asynchronous GRPO staleness with conditional unbiasedness assumptions on returned samples. Deadline-filtered rollouts may fall outside those assumptions. Their convergence results must not be presented as missing-data guarantees.
- Classical missing-data / Horvitz–Thompson estimation already explains inverse-inclusion weighting and nonidentifiability without assumptions. Extending these facts to a queue **alone** is insufficient for a novelty claim.

A substantive next result needs a new constraint, matching upper/lower bound, or operationally useful theorem beyond these baselines. No such result is claimed in this version.
