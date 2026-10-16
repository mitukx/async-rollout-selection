"""Missing-outcome audit calculations for a fixed two-action policy.

An audit independently reveals the *same* reward of an unfinished rollout.
This assumption is essential; an independent rerun is not interchangeable.
"""
from dataclasses import dataclass
from random import Random
from math import log, sqrt


def gradient_interval(p, observed0, observed1, missing0, missing1):
    """Sharp identified interval using E[S*R|A=a] and P(S=0|A=a)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0,1)")
    for mass, missing in ((observed0, missing0), (observed1, missing1)):
        if mass < 0 or missing < 0 or mass + missing > 1 + 1e-12:
            raise ValueError("incompatible observed reward mass / missing probability")
    factor = p * (1 - p)
    return (factor * (observed1 - observed0 - missing0),
            factor * (observed1 + missing1 - observed0))


def audit_cost(p, missing0, missing1, h0, h1):
    return (1 - p) * missing0 * h0 + p * missing1 * h1


def optimal_audit_probabilities(p, missing0, missing1, budget):
    """Minimize worst-case variance of the *uncentered* audit-HT score.

    budget is expected audits per launched rollout. Missing rates are assumed
    known; this does NOT enforce a hard finite-run audit cap.
    """
    if not 0 < p < 1 or not all(0 <= x <= 1 for x in (missing0, missing1)):
        raise ValueError("invalid policy probability or missing rate")
    if budget <= 0:
        raise ValueError("positive audit budget required")
    total_missing = (1 - p) * missing0 + p * missing1
    if budget >= total_missing:
        return 1.0, 1.0
    # KKT water-filling with at most two strata; no iterative optimizer.
    mass = ((1-p)*missing0, p*missing1)
    score = (p, 1-p)
    h = [1.0, 1.0]
    active = {a for a in (0, 1) if mass[a] > 0}
    remaining = budget
    while active:
        scale = remaining / sum(mass[a]*score[a] for a in active)
        capped = {a for a in active if scale*score[a] >= 1}
        if not capped:
            for a in active:
                h[a] = scale*score[a]
            break
        for a in capped:
            h[a] = 1.0
            remaining -= mass[a]
        active -= capped
    return h[0], h[1]


def worst_case_audit_variance(p, missing0, missing1, h0, h1):
    """Exact sup of one-launch variance over R in [0,1] for fixed missing rates.

    Applies *only* to the uncentered audit Horvitz-Thompson estimator;
    not an information-theoretic minimax lower bound over all estimators.
    """
    if not 0 < p < 1 or not all(0 <= x <= 1 for x in (missing0, missing1)):
        raise ValueError("invalid policy probability or missing rate")
    if not 0 < h0 <= 1 or not 0 < h1 <= 1:
        raise ValueError("both audit probabilities must be positive and <=1")
    return ((1-p) * p*p * (1-missing0 + missing0/h0) +
            p * (1-p)**2 * (1-missing1 + missing1/h1))


def audit_confidence_radius(n, p, h0, h1, delta):
    """Hoeffding radius for sequential, nonanticipating sampling and audit."""
    if n < 1 or not 0 < p < 1 or not (0 < h0 <= 1 and 0 < h1 <= 1) or not 0 < delta < 1:
        raise ValueError("invalid confidence-bound parameters")
    span = p/h0 + (1-p)/h1
    return span * sqrt(log(2/delta) / (2*n))


@dataclass(frozen=True)
class DelayedBandit:
    p: float
    reward_prob: tuple[float, float]
    complete_if_reward: tuple[float, float]
    complete_if_failure: tuple[float, float]

    def __post_init__(self):
        if not 0 < self.p < 1:
            raise ValueError("p must be in (0,1)")
        if any(len(x) != 2 for x in (self.reward_prob, self.complete_if_reward, self.complete_if_failure)):
            raise ValueError("two entries are needed for every action-specific parameter")
        if any(not 0 <= v <= 1 for x in (self.reward_prob, self.complete_if_reward, self.complete_if_failure) for v in x):
            raise ValueError("probabilities must lie in [0,1]")

    def observed_masses(self):
        return tuple(self.reward_prob[a] * self.complete_if_reward[a] for a in (0, 1))

    def missing_probabilities(self):
        return tuple(1 - (self.reward_prob[a]*self.complete_if_reward[a] +
                           (1-self.reward_prob[a])*self.complete_if_failure[a]) for a in (0, 1))

    def true_gradient(self):
        return self.p * (1-self.p) * (self.reward_prob[1]-self.reward_prob[0])

    def completed_only_gradient(self):
        mass0, mass1 = self.observed_masses()
        missing0, missing1 = self.missing_probabilities()
        selected_fraction = 1 - ((1-self.p)*missing0 + self.p*missing1)
        if selected_fraction == 0:
            raise ValueError("no completions")
        return self.p*(1-self.p)*(mass1-mass0)/selected_fraction

    def audit_gradient_variance(self, h0, h1):
        if not (0 < h0 <= 1 and 0 < h1 <= 1):
            raise ValueError("positive audit probabilities required")
        observed0, observed1 = self.observed_masses()
        hidden0 = self.reward_prob[0] - observed0
        hidden1 = self.reward_prob[1] - observed1
        second = ((1-self.p)*self.p**2*(observed0 + hidden0/h0)
                  + self.p*(1-self.p)**2*(observed1 + hidden1/h1))
        return second - self.true_gradient()**2

    def simulate(self, launches, h0, h1, seed=0):
        """Returns (completed-only mean, audited per-launch mean, completions, audits)."""
        if launches < 1 or not (0 < h0 <= 1 and 0 < h1 <= 1):
            raise ValueError("positive launches and audit probabilities required")
        rng = Random(seed)
        completed_sum = audit_sum = 0.0
        completions = audits = 0
        for _ in range(launches):
            a = int(rng.random() < self.p)
            reward = int(rng.random() < self.reward_prob[a])
            completed = rng.random() < (self.complete_if_reward[a] if reward else self.complete_if_failure[a])
            score = reward * (a - self.p)
            if completed:
                completed_sum += score
                audit_sum += score
                completions += 1
            elif rng.random() < (h1 if a else h0):
                audits += 1
                audit_sum += score / (h1 if a else h0)
        if not completions:
            raise ValueError("no completed rollouts")
        return completed_sum/completions, audit_sum/launches, completions, audits


if __name__ == "__main__":
    model = DelayedBandit(0.5, (0.8, 0.4), (0.1, 0.9), (0.9, 0.1))
    t0, t1 = model.missing_probabilities()
    h0, h1 = optimal_audit_probabilities(model.p, t0, t1, budget=0.1)
    naive, audit, completed, audited = model.simulate(200_000, h0, h1, seed=7)
    print(f"true gradient: {model.true_gradient():+.6f}")
    print(f"completed-only expected: {model.completed_only_gradient():+.6f}")
    print(f"completed-only observed: {naive:+.6f} ({completed} completions)")
    print(f"audited estimate: {audit:+.6f} ({audited} audits)")
    print(f"audit probabilities: ({h0:.4f}, {h1:.4f})")
