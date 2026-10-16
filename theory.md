# Exact baseline: completion-selected policy gradients

Fix a logistic policy `P(A=1)=p=sigmoid(theta)`, with `0<p<1`. Rewards `r0,r1 in [0,1]` are deterministic. A launched rollout is selected with probability `q_a=P(S=1|A=a)`, independently across launches. The policy does not change while samples are collected. Put `Z=(1-p)q0+p q1>0`.

**Proposition 1 (sign reversal).** The launch-distribution gradient of expected reward is
`G=p(1-p)(r1-r0)`.
The expected completed-only REINFORCE score is
`G_selected=E[r_A(A-p)|S=1]=p(1-p)(q1*r1-q0*r0)/Z`.
Thus, for nonzero terms, the update directions disagree iff
`(r1-r0)(q1*r1-q0*r0)<0`.
*Proof:* Conditional action probabilities among selected rollouts are `p*q1/Z` and `(1-p)*q0/Z`; substitute Bernoulli scores `1-p` and `-p`. QED.

Example: `p=0.5,r0=1,r1=0.2,q0=0.1,q1=1` gives `G=-0.2` and `G_selected=1/22=+0.0454545...`. **Important:** `G_selected` is the expectation of an estimator using the *original-policy score*; it is not claimed to be the derivative of a selected-data objective.

**Proposition 2 (inverse-probability correction).** If `q0,q1>0` are known, for `n` independent *launched* rollouts define
`G_HT=(1/n) sum_i S_i*r_{A_i}*(A_i-p)/q_{A_i}`.
Then `E[G_HT]=G` and
`Var(G_HT)=[(1-p)*p^2*r0^2/q0+p*(1-p)^2*r1^2/q1-G^2]/n`.
*Proof:* Conditional on action, `E[S/q_A|A]=1`; compute the second moment using `S^2=S`, then divide variance by `n`. QED. This is classical Horvitz–Thompson weighting; small inclusion probabilities entail high variance.

**Proposition 3 (outcome-dependent selection is not identified from observed rewards alone).** Keep the reward/selection mechanism for action 1 fixed in two worlds. For action 0, consider binary rewards with the following mechanisms:

| World | P(R=1) | P(S=1 | R=1) | P(S=1 | R=0) |
|---|---:|---:|---:|
| I | 1/2 | 1/2 | 1/2 |
| II | 1/4 | 1 | 1/3 |

In either world, conditional on action 0, the observable probabilities are
`P(S=1,R=1)=1/4`, `P(S=1,R=0)=1/4`, `P(S=0)=1/2`.
Even observing all launched actions and completion flags, any finite collection of logs has identical distributions across worlds, although their mean rewards differ by `1/4`. The two true policy gradients therefore differ in absolute value by `D=p(1-p)/4`. For every estimator based on these logs,
`max(E_I[(Ghat-G_I)^2],E_II[(Ghat-G_II)^2]) >= D^2/4`.
*Proof:* Observed-data distributions are identical; average the two squared errors and use `[(x-a)^2+(x-b)^2]/2 >= (a-b)^2/4`. QED. This is a classical missing-not-at-random obstruction; known reward-conditioned propensities or an independent audit can alter identification.

## Scope and novelty boundary

These are exact elementary baseline facts, **not new theorems**. The selection variable stands for inclusion in one update, not permanent censoring; there is no queue interaction, changing policy, correlated rollout, wall-clock objective, or staleness correction. The inverse-weighted estimator is normalized by launches, never by completions. The nonidentifiability example cannot be fixed by action-only weights.

Prior work: [Applied Compute (2026)](https://www.appliedcompute.com/research/staleness-in-fully-async-rl) already analyzes length selection in asynchronous queue policies; [He et al. (2026)](https://arxiv.org/abs/2610.01896) analyze asynchronous LLM post-training with conditional unbiasedness assumptions. Classical sampling theory handles action-dependent missingness. Merely rediscovering these effects is insufficient.

**Next research target:** mathematically define an adaptive queue where inclusion depends on time, policy version, and reward-correlated trajectory length; determine an identifiable gradient estimand using logged launches plus finite audit budget; seek a nontrivial *wall-clock-constrained* upper/lower bound beyond classical missing-data results. No claim of novelty or a solved frontier problem is made yet.
