# Reward-dependent completion: identification and audits

This document extends [the baseline](theory.md). Everything here is a classical missing-data / two-phase sampling consequence in a **fixed-policy, independent-launch, two-action** model; it is **not** claimed as a novel result or asynchronous-RL convergence guarantee.

Let `P(A=1)=p`, `0<p<1`. A bounded reward `R in [0,1]` is defined for *every launched rollout*, including one not returned by the training deadline. `S=1` means the original reward is observed before that deadline. Missingness may depend arbitrarily on the reward, conditional on action. Define observable `u_a=E[S R | A=a]` and `m_a=P(S=0 | A=a)`. The target is the launch-population policy gradient `G=p(1-p)(E[R|A=1]-E[R|A=0])`; there is no parameter drift.

## Proposition 4: sharp identified interval without audits

Writing `v_a=E[(1-S)R|A=a]`, bounded rewards imply `0 <= v_a <= m_a`. Therefore the identified interval is

```
G_low  = p(1-p)*(u1-u0-m0)
G_high = p(1-p)*(u1+m1-u0).
```

Both endpoints and every point between them are attainable by choosing the unobserved conditional reward distributions appropriately, leaving the observed data unchanged. Hence this is a *sharp population-level interval*, not a confidence interval. In particular, a positive sign is identifiable only when `G_low>0`; a negative sign only when `G_high<0`. If `m0+m1>0`, the interval width is `p(1-p)(m0+m1)`. Proof: bound each unobserved mean by its probability mass; realize arbitrary hidden means with constant conditional rewards on `S=0`.

## Proposition 5: independent follow-up audit identifies the gradient

For an unfinished rollout (`S=0`), draw an independent audit flag `B|A=a,S=0 ~ Bernoulli(h_a)`, where `0<h_a<=1` is set independently of its *unobserved reward*. If audited, the **original** hidden reward becomes observable. In particular, merely rerunning the task to obtain a fresh reward does not necessarily reveal the same latent outcome.

For each launch let `X=R(A-p)[S+(1-S)B/h_A]`. Conditional on `A,R,S` the bracket has mean one; thus `E[X]=G`. Independent launches give an unbiased average `Ghat=n^{-1} sum X_i`. With `v_a=E[(1-S)R^2|A=a]` and `w_a=E[S R^2|A=a]`,

```
Var(X) = (1-p)*p^2*(w0+v0/h0)
       + p*(1-p)^2*(w1+v1/h1) - G^2.
```

This is the exact per-launch variance; the average's variance is `Var(X)/n`. It requires neither knowing `P(S=1|A,R)` nor independence of `S` from `R`. Audits are Bernoulli follow-ups, and `n` counts **launches** (including unfinished, unaudited ones).

## Proposition 6: optimal action-specific audit rates within this estimator class

Suppose missing fractions `m0,m1` are known and audits have equal cost, with expected audits per launch limited to `b>0`: `(1-p)m0*h0 + p*m1*h1 <= b`. Over arbitrary reward laws in `[0,1]` consistent with those missing fractions, the worst-case `Var(X)` is

```
Vmax = (1-p)*p^2*(1-m0+m0/h0)
     + p*(1-p)^2*(1-m1+m1/h1).
```

To see exact attainability choose `R=1` for both actions: the true gradient is zero, and all squared-reward terms reach their maxima. Hence minimizing this expression is a well-defined **worst-case-variance objective for this uncentered Horvitz–Thompson estimator**. For nonempty missing strata, the KKT solution is

```
h0 = min(1, lambda*p)
h1 = min(1, lambda*(1-p)),
```

where `lambda>0` uses the available expected budget. If `b >= (1-p)m0+p*m1` both rates equal one. A stratum with `m_a=0` requires no audit and its `h_a` is arbitrary. Proof: discard terms constant in `h`; minimize `sum_a P(A=a)*m_a*(A-p)^2/h_a` with linear cost. Differentiation gives `h_a proportional to |A-p|` before clipping; convexity proves global optimality. The formula is **not** minimax-optimal among all estimators (centered, model-assisted, or adaptive estimators may do better). The budget is in *expectation*, not a hard per-run cap. The solution assumes known missing fractions, not finite-sample estimates of them.

## Finite-sample confidence bound

For fixed `p,h0,h1` and independent launches, `X in [-p/h0,(1-p)/h1]`. Hoeffding's bound yields, with probability at least `1-delta`,

```
|Ghat-G| <= (p/h0+(1-p)/h1)*sqrt(log(2/delta)/(2*n)).
```

This conservative bound holds even with reward-dependent censoring. It does not give an anytime-valid confidence sequence. A varying policy would change the target and requires a separate argument.

## Novelty gate and operational limits

- Nonrandom delayed feedback and outcome-dependent censoring already occur in delayed-bandit and missing-data research. Sharp partial identification, inverse-probability follow-up, variance bounds, and Neyman-like audit allocation are standard *types* of results; the propositions above are a controlled baseline rather than original research.
- A useful new theorem would explicitly model evolving queue state and policy versions, and derive a **wall-clock** performance or estimation trade-off with controlled auditing. It must outperform or prove a limitation of established queue scheduling / delayed feedback methods, not simply rename the expressions above.
- This study assumes that an independently randomized follow-up can expose the **original** otherwise missing outcome. For truly killed rollouts or unrecoverable trajectories this may be false; in that case identification from this audit model is inapplicable.
- The present code uses a single fixed deadline and fixed policy, not an asynchronous learner. It does not model execution costs of audits or an actual queue.

Relevant references: [Applied Compute 2026](https://www.appliedcompute.com/research/staleness-in-fully-async-rl), [He et al. 2026](https://arxiv.org/abs/2610.01896), and general Horvitz–Thompson / two-phase sampling and missing-not-at-random literature.
