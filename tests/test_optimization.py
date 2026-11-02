"""
Unit tests for lending allocation optimisation.

Optimisation tests use a small N=50 portfolio (opt_portfolio fixture) to keep
SLSQP runtimes within a few seconds.  The equal-weight and metrics tests use
the full N=1000 portfolio only where size-dependent properties are checked.

Tests verify:
  - Budget constraint: allocations sum to K.
  - Concentration constraint: x_i <= kappa * K for all i.
  - Equal-weight allocation produces x_i = K/N.
  - Fixed-PD linear optimisation concentrates on top-coefficient obligors.
  - ECL computation is non-negative and bounded by LGD * K.
  - Metrics dictionary contains all required keys.
"""

import numpy as np
import pytest

from src.config import K_BUDGET, N_OBLIGORS, KAPPA, LGD, SCENARIOS
from src.optimization import (
    solve_climate_aware,
    solve_conventional,
    solve_equal_weight,
    solve_fixed_pd_linear,
    compute_metrics,
)
from src.pd_engine import calibrate_thresholds, compute_pd


class TestEqualWeight:
    def test_sum_to_budget(self):
        x = solve_equal_weight()
        assert abs(x.sum() - K_BUDGET) < 1e-8

    def test_uniform(self):
        x = solve_equal_weight()
        assert np.allclose(x, K_BUDGET / N_OBLIGORS)

    def test_length(self):
        x = solve_equal_weight()
        assert len(x) == N_OBLIGORS


class TestConventionalOptimisation:
    """Uses N=50 opt_portfolio for fast SLSQP convergence."""

    def test_budget_constraint(self, opt_portfolio, opt_thresholds):
        x, _, _ = solve_conventional(opt_portfolio, opt_thresholds)
        assert abs(x.sum() - K_BUDGET) < 1e-4

    def test_concentration_constraint(self, opt_portfolio, opt_thresholds):
        x, _, _ = solve_conventional(opt_portfolio, opt_thresholds)
        assert (x <= KAPPA * K_BUDGET + 1e-6).all()

    def test_non_negative(self, opt_portfolio, opt_thresholds):
        x, _, _ = solve_conventional(opt_portfolio, opt_thresholds)
        assert (x >= -1e-8).all()

    def test_objective_positive(self, opt_portfolio, opt_thresholds):
        _, obj, _ = solve_conventional(opt_portfolio, opt_thresholds)
        assert obj > 0


class TestClimateAwareOptimisation:
    """Uses N=50 opt_portfolio for fast SLSQP convergence."""

    @pytest.fixture(scope="class")
    def climate_aware_alloc(self, opt_portfolio, opt_thresholds, opt_shocks):
        """Climate-aware allocations for HWTP (scenario 1)."""
        S_s = opt_shocks[1]  # HWTP
        x, obj, success = solve_climate_aware(opt_portfolio, opt_thresholds, S_s)
        return x, obj, success

    def test_budget_constraint(self, climate_aware_alloc):
        x, _, _ = climate_aware_alloc
        assert abs(x.sum() - K_BUDGET) < 1e-4

    def test_concentration_constraint(self, climate_aware_alloc):
        x, _, _ = climate_aware_alloc
        assert (x <= KAPPA * K_BUDGET + 1e-6).all()

    def test_non_negative(self, climate_aware_alloc):
        x, _, _ = climate_aware_alloc
        assert (x >= -1e-8).all()

    def test_objective_positive(self, climate_aware_alloc):
        _, obj, _ = climate_aware_alloc
        assert obj > 0


class TestFixedPDOptimisation:
    """Uses N=50 opt_portfolio."""

    def test_budget_constraint(self, opt_portfolio, opt_thresholds, opt_shocks):
        S_s = opt_shocks[0]
        x, _ = solve_fixed_pd_linear(opt_portfolio, opt_thresholds, S_s,
                                      x_ref=np.full(len(opt_portfolio), K_BUDGET / len(opt_portfolio)))
        assert abs(x.sum() - K_BUDGET) < 1e-6

    def test_concentration_constraint(self, opt_portfolio, opt_thresholds, opt_shocks):
        S_s = opt_shocks[0]
        x, _ = solve_fixed_pd_linear(opt_portfolio, opt_thresholds, S_s,
                                      x_ref=np.full(len(opt_portfolio), K_BUDGET / len(opt_portfolio)))
        assert (x <= KAPPA * K_BUDGET + 1e-8).all()

    def test_concentrates_on_few(self, opt_portfolio, opt_thresholds, opt_shocks):
        """Fixed-PD should concentrate at the cap; few obligors funded."""
        S_s = opt_shocks[0]
        x, _ = solve_fixed_pd_linear(opt_portfolio, opt_thresholds, S_s,
                                      x_ref=np.full(len(opt_portfolio), K_BUDGET / len(opt_portfolio)))
        n_funded = int(np.sum(x > 1e-6))
        # Greedy fill at cap kappa*K per obligor; expect ~1/kappa = 20 obligors
        assert n_funded <= int(1.0 / KAPPA) + 1, \
            f"Fixed-PD should fund ~{int(1/KAPPA)} obligors, got {n_funded}"


class TestMetrics:
    """Uses full_portfolio for HHI identity test; others use opt_portfolio."""

    def test_metrics_keys(self, opt_portfolio, opt_thresholds, opt_shocks):
        n = len(opt_portfolio)
        x = np.full(n, K_BUDGET / n)
        S_s = opt_shocks[0]
        m = compute_metrics(opt_portfolio, x, opt_thresholds, S_s)
        for key in ["objective", "ecl", "n_funded", "max_share", "hhi"]:
            assert key in m

    def test_ecl_non_negative(self, opt_portfolio, opt_thresholds, opt_shocks):
        n = len(opt_portfolio)
        x = np.full(n, K_BUDGET / n)
        S_s = opt_shocks[0]
        m = compute_metrics(opt_portfolio, x, opt_thresholds, S_s)
        assert m["ecl"] >= 0

    def test_ecl_bounded(self, opt_portfolio, opt_thresholds, opt_shocks):
        """ECL <= LGD * K."""
        n = len(opt_portfolio)
        x = np.full(n, K_BUDGET / n)
        S_s = opt_shocks[0]
        m = compute_metrics(opt_portfolio, x, opt_thresholds, S_s)
        assert m["ecl"] <= LGD * K_BUDGET + 1e-6

    def test_hhi_in_range(self, opt_portfolio, opt_thresholds, opt_shocks):
        """HHI in [1/N, 1]."""
        n = len(opt_portfolio)
        x = np.full(n, K_BUDGET / n)
        S_s = opt_shocks[0]
        m = compute_metrics(opt_portfolio, x, opt_thresholds, S_s)
        assert 1.0 / n - 1e-8 <= m["hhi"] <= 1.0 + 1e-8

    def test_n_funded_positive(self, opt_portfolio, opt_thresholds, opt_shocks):
        n = len(opt_portfolio)
        x = np.full(n, K_BUDGET / n)
        S_s = opt_shocks[0]
        m = compute_metrics(opt_portfolio, x, opt_thresholds, S_s)
        assert m["n_funded"] > 0

    def test_hhi_equal_weight_is_inverse_n(self, opt_portfolio, opt_thresholds, opt_shocks):
        n = len(opt_portfolio)
        x = np.full(n, K_BUDGET / n)
        S_s = opt_shocks[0]
        m = compute_metrics(opt_portfolio, x, opt_thresholds, S_s)
        expected_hhi = 1.0 / n
        assert abs(m["hhi"] - expected_hhi) < 1e-8
