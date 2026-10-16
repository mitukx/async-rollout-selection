import unittest
from audit import (DelayedBandit, audit_cost, gradient_interval,
                   optimal_audit_probabilities, worst_case_audit_variance,
                   audit_confidence_radius)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.model = DelayedBandit(0.5, (0.8, 0.4), (0.1, 0.9), (0.9, 0.1))

    def test_sharp_identification_interval_contains_target(self):
        m0, m1 = self.model.observed_masses()
        t0, t1 = self.model.missing_probabilities()
        lo, hi = gradient_interval(self.model.p, m0, m1, t0, t1)
        self.assertLessEqual(lo, self.model.true_gradient())
        self.assertGreaterEqual(hi, self.model.true_gradient())
        self.assertAlmostEqual(hi-lo, self.model.p*(1-self.model.p)*(t0+t1))

    def test_identification_violated_inputs(self):
        with self.assertRaises(ValueError):
            gradient_interval(.5, .9, .1, .3, .1)

    def test_sign_reversal_with_reward_dependent_completion(self):
        self.assertAlmostEqual(self.model.true_gradient(), -.1)
        self.assertAlmostEqual(self.model.completed_only_gradient(), 0.07/0.34)

    def test_audit_budget_and_variance_improvement(self):
        t0, t1 = self.model.missing_probabilities()
        b = 0.1
        h0, h1 = optimal_audit_probabilities(self.model.p, t0, t1, b)
        self.assertAlmostEqual(audit_cost(self.model.p, t0, t1, h0, h1), b, places=11)
        h_equal = b / ((1-self.model.p)*t0 + self.model.p*t1)
        self.assertLessEqual(worst_case_audit_variance(self.model.p, t0, t1, h0, h1),
                             worst_case_audit_variance(self.model.p, t0, t1, h_equal, h_equal)+1e-12)

    def test_asymmetric_allocation_and_cap(self):
        p, t0, t1, b = .2, .8, .3, .15
        h0, h1 = optimal_audit_probabilities(p, t0, t1, b)
        self.assertGreater(h1, h0)
        self.assertAlmostEqual(audit_cost(p, t0, t1, h0, h1), b, places=11)
        full0, full1 = optimal_audit_probabilities(p, t0, t1, 1.0)
        self.assertEqual((full0, full1), (1, 1))

    def test_no_missing_in_stratum(self):
        h0, h1 = optimal_audit_probabilities(.5, 0, .6, .05)
        self.assertEqual(h0, 1)
        self.assertAlmostEqual(.5*.6*h1, .05)

    def test_bound_attained_by_constant_unit_rewards(self):
        p, t0, t1, h0, h1 = .3, .8, .2, .2, .6
        unit_reward = DelayedBandit(p, (1,1), (1-t0, 1-t1), (0,0))
        self.assertAlmostEqual(unit_reward.audit_gradient_variance(h0, h1),
                               worst_case_audit_variance(p, t0, t1, h0, h1))

    def test_monte_carlo_matches_true_gradient(self):
        t0, t1 = self.model.missing_probabilities()
        h0, h1 = optimal_audit_probabilities(.5, t0, t1, .1)
        naive, audited, completed, audits = self.model.simulate(300_000, h0, h1, seed=17)
        self.assertGreater(naive, 0)
        self.assertAlmostEqual(naive, self.model.completed_only_gradient(), delta=.015)
        self.assertAlmostEqual(audited, self.model.true_gradient(), delta=.015)
        self.assertAlmostEqual(audits/300_000, .1, delta=.003)

    def test_confidence_radius_shrinks_with_samples_and_audit(self):
        r = audit_confidence_radius(1000, .5, .2, .3, .05)
        self.assertAlmostEqual(audit_confidence_radius(4000, .5, .2, .3, .05), r/2)
        self.assertLess(audit_confidence_radius(1000, .5, .4, .6, .05), r)

    def test_invalid_audit_parameters(self):
        with self.assertRaises(ValueError):
            optimal_audit_probabilities(.5, .3, .5, 0)
        with self.assertRaises(ValueError):
            self.model.simulate(100, 0, .1)


if __name__ == "__main__":
    unittest.main()
