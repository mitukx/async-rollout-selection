"""Exact two-action completion-selection calculations; standard library only."""
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Bandit:
    p: float
    r0: float
    r1: float
    q0: float
    q1: float

    def __post_init__(self):
        if not 0 < self.p < 1:
            raise ValueError("p must be in (0,1)")
        if not all(0 <= x <= 1 for x in (self.r0, self.r1, self.q0, self.q1)):
            raise ValueError("rewards and inclusion probabilities must be in [0,1]")

    def true_gradient(self):
        return self.p * (1 - self.p) * (self.r1 - self.r0)

    def selected_gradient(self):
        """E[R(A-p)|S=1], not the gradient of selected-data expected reward."""
        z = (1 - self.p) * self.q0 + self.p * self.q1
        if z == 0:
            raise ValueError("no rollouts selected")
        return self.p * (1 - self.p) * (self.q1 * self.r1 - self.q0 * self.r0) / z

    def ipw_variance(self, launches):
        """Variance of per-launch Horvitz-Thompson gradient mean."""
        if launches < 1 or min(self.q0, self.q1) <= 0:
            raise ValueError("launches and both inclusion probabilities must be positive")
        second = ((1 - self.p) * self.p**2 * self.r0**2 / self.q0
                  + self.p * (1 - self.p)**2 * self.r1**2 / self.q1)
        return (second - self.true_gradient()**2) / launches


def simulate(model, launches, seed=0):
    """Return selected score mean, inverse-probability mean per launch, selected count."""
    if launches < 1 or min(model.q0, model.q1) <= 0:
        raise ValueError("positive launches and inclusion probabilities required")
    rng = Random(seed)
    selected_sum = corrected_sum = 0.0
    selected = 0
    for _ in range(launches):
        a = int(rng.random() < model.p)
        q = model.q1 if a else model.q0
        if rng.random() < q:
            score = (model.r1 if a else model.r0) * (a - model.p)
            selected_sum += score
            corrected_sum += score / q
            selected += 1
    if not selected:
        raise ValueError("no rollouts selected")
    return selected_sum / selected, corrected_sum / launches, selected


if __name__ == "__main__":
    model = Bandit(0.5, 1.0, 0.2, 0.1, 1.0)
    selected, ipw, count = simulate(model, 200_000, seed=7)
    print(f"true: {model.true_gradient():+.6f}")
    print(f"selected expected: {model.selected_gradient():+.6f}")
    print(f"selected observed: {selected:+.6f} ({count} completed)")
    print(f"IPW observed: {ipw:+.6f}")
