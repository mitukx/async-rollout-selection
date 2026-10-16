import unittest
from selection import Bandit, simulate


class SelectionTests(unittest.TestCase):
    def test_sign_reversal(self):
        m = Bandit(0.5, 1, 0.2, 0.1, 1)
        self.assertAlmostEqual(m.true_gradient(), -0.2)
        self.assertAlmostEqual(m.selected_gradient(), 1 / 22)

    def test_equal_inclusion(self):
        m = Bandit(0.3, 0.9, 0.1, 0.4, 0.4)
        self.assertAlmostEqual(m.true_gradient(), m.selected_gradient())

    def test_exact_variance(self):
        m = Bandit(0.4, 0.7, 0.2, 0.25, 0.8)
        outcomes = [
            (0, (1 - m.p) * (1 - m.q0) + m.p * (1 - m.q1)),
            (-m.p * m.r0 / m.q0, (1 - m.p) * m.q0),
            ((1 - m.p) * m.r1 / m.q1, m.p * m.q1),
        ]
        mean = sum(v * prob for v, prob in outcomes)
        var = sum((v - mean)**2 * prob for v, prob in outcomes)
        self.assertAlmostEqual(mean, m.true_gradient())
        self.assertAlmostEqual(var, m.ipw_variance(1))
        self.assertAlmostEqual(var / 10, m.ipw_variance(10))

    def test_simulation(self):
        m = Bandit(0.5, 1, 0.2, 0.1, 1)
        naive, corrected, count = simulate(m, 200_000, seed=7)
        self.assertGreater(count, 0)
        self.assertAlmostEqual(naive, m.selected_gradient(), delta=0.01)
        self.assertAlmostEqual(corrected, m.true_gradient(), delta=0.01)

    def test_zero_inclusion_cannot_be_weighted(self):
        m = Bandit(0.5, 1, 0.2, 0, 1)
        with self.assertRaises(ValueError):
            m.ipw_variance(10)

    def test_outcome_dependent_missingness(self):
        # Two distinct reward distributions, same observed outcomes and flags.
        def observed(mu, q1, q0):
            return (mu*q1, (1-mu)*q0, mu*(1-q1)+(1-mu)*(1-q0))
        for a, b in zip(observed(0.5, 0.5, 0.5), observed(0.25, 1, 1/3)):
            self.assertAlmostEqual(a, b)


if __name__ == "__main__":
    unittest.main()
