"""Unit tests for the Bayesian A/B testing engine."""
from __future__ import annotations

import pytest

from src.experiments import (
    ExperimentData,
    BayesianResult,
    _posterior_params,
    expected_lift,
    prob_b_beats_a,
    recommend_sample_size,
    run_bayesian_ab_test,
)


# ── Posterior parameter tests ─────────────────────────────────────────────────

class TestPosteriorParams:
    def test_uninformative_prior_no_data(self):
        a, b = _posterior_params(0, 0, prior_alpha=1.0, prior_beta=1.0)
        assert a == 1.0
        assert b == 1.0

    def test_correct_update(self):
        # 10 successes out of 100 trials, prior Beta(1,1)
        a, b = _posterior_params(10, 100, prior_alpha=1.0, prior_beta=1.0)
        assert a == pytest.approx(11.0)
        assert b == pytest.approx(91.0)

    def test_informative_prior(self):
        # Prior encodes belief of ~30% base rate: Beta(30, 70)
        a, b = _posterior_params(5, 20, prior_alpha=30.0, prior_beta=70.0)
        assert a == pytest.approx(35.0)
        assert b == pytest.approx(85.0)

    def test_all_successes(self):
        a, b = _posterior_params(50, 50)
        assert a == 51.0
        assert b == 1.0


# ── prob_b_beats_a tests ──────────────────────────────────────────────────────

class TestProbBBeatsA:
    def test_symmetric_posteriors_approx_half(self):
        # Identical Beta distributions → P(B > A) ≈ 0.5
        p = prob_b_beats_a(50.0, 50.0, 50.0, 50.0, n_samples=100_000)
        assert abs(p - 0.5) < 0.04

    def test_clearly_superior_treatment(self):
        # A has low rate (~2%), B has high rate (~80%) → P(B>A) ≈ 1
        p = prob_b_beats_a(2.0, 98.0, 80.0, 20.0, n_samples=100_000)
        assert p > 0.999

    def test_clearly_inferior_treatment(self):
        # A has high rate (~80%), B has low rate (~2%) → P(B>A) ≈ 0
        p = prob_b_beats_a(80.0, 20.0, 2.0, 98.0, n_samples=100_000)
        assert p < 0.001

    def test_output_is_probability(self):
        p = prob_b_beats_a(10.0, 10.0, 12.0, 8.0)
        assert 0.0 <= p <= 1.0


# ── expected_lift tests ───────────────────────────────────────────────────────

class TestExpectedLift:
    def test_positive_when_treatment_better(self):
        lift = expected_lift(2.0, 98.0, 80.0, 20.0, n_samples=100_000)
        assert lift > 0.0

    def test_negative_when_treatment_worse(self):
        lift = expected_lift(80.0, 20.0, 2.0, 98.0, n_samples=100_000)
        assert lift < 0.0

    def test_near_zero_when_equal(self):
        lift = expected_lift(50.0, 50.0, 50.0, 50.0, n_samples=100_000)
        assert abs(lift) < 0.05

    def test_lift_magnitude_reasonable(self):
        # A ≈ 40%, B ≈ 60% → lift ≈ 50%
        lift = expected_lift(40.0, 60.0, 60.0, 40.0, n_samples=200_000)
        assert 0.30 < lift < 0.80


# ── recommend_sample_size tests ───────────────────────────────────────────────

class TestRecommendSampleSize:
    def test_returns_positive_integer(self):
        n = recommend_sample_size(0.20)
        assert isinstance(n, int)
        assert n > 0

    def test_larger_mde_needs_less_data(self):
        n_small_mde = recommend_sample_size(0.20, mde=0.02)
        n_large_mde = recommend_sample_size(0.20, mde=0.10)
        assert n_small_mde > n_large_mde

    def test_higher_power_needs_more_data(self):
        n_low = recommend_sample_size(0.20, power=0.70)
        n_high = recommend_sample_size(0.20, power=0.95)
        assert n_high > n_low

    def test_clips_edge_rates(self):
        # Should not raise for boundary rates
        n = recommend_sample_size(0.001)
        assert n > 0
        n = recommend_sample_size(0.999)
        assert n > 0


# ── run_bayesian_ab_test integration tests ────────────────────────────────────

class TestRunBayesianABTest:
    def _make_exp(
        self,
        ctrl_conv: int = 80,
        ctrl_trials: int = 400,
        trt_conv: int = 100,
        trt_trials: int = 400,
    ) -> ExperimentData:
        return ExperimentData(
            name="test_exp",
            description="Test",
            segment="High",
            control_description="Control",
            treatment_description="Treatment",
            control_conversions=ctrl_conv,
            control_trials=ctrl_trials,
            treatment_conversions=trt_conv,
            treatment_trials=trt_trials,
        )

    def test_result_is_bayesian_result(self):
        result = run_bayesian_ab_test(self._make_exp())
        assert isinstance(result, BayesianResult)

    def test_prob_is_probability(self):
        result = run_bayesian_ab_test(self._make_exp())
        assert 0.0 <= result.prob_treatment_beats_control <= 1.0

    def test_high_prob_when_treatment_clearly_better(self):
        # ctrl: 40/400 (10%), trt: 120/400 (30%)
        result = run_bayesian_ab_test(self._make_exp(ctrl_conv=40, trt_conv=120))
        assert result.prob_treatment_beats_control > 0.95

    def test_low_prob_when_treatment_clearly_worse(self):
        # ctrl: 120/400 (30%), trt: 40/400 (10%)
        result = run_bayesian_ab_test(self._make_exp(ctrl_conv=120, trt_conv=40))
        assert result.prob_treatment_beats_control < 0.05

    def test_rates_are_valid(self):
        result = run_bayesian_ab_test(self._make_exp())
        assert 0.0 <= result.control_rate <= 1.0
        assert 0.0 <= result.treatment_rate <= 1.0

    def test_posterior_samples_populated(self):
        result = run_bayesian_ab_test(self._make_exp())
        assert len(result.control_posterior["samples"]) == 500
        assert len(result.treatment_posterior["samples"]) == 500

    def test_zero_data_uses_prior(self):
        # No data: posterior equals prior Beta(1,1) for both arms
        exp = self._make_exp(ctrl_conv=0, ctrl_trials=0, trt_conv=0, trt_trials=0)
        result = run_bayesian_ab_test(exp)
        # Both posteriors are identical → P ≈ 0.5
        assert abs(result.prob_treatment_beats_control - 0.5) < 0.1

    def test_sample_size_recommendation_positive(self):
        result = run_bayesian_ab_test(self._make_exp())
        assert result.recommended_sample_size > 0

    def test_lift_direction_matches_rates(self):
        result = run_bayesian_ab_test(self._make_exp(ctrl_conv=80, trt_conv=120))
        assert result.expected_lift > 0  # treatment rate > control rate
