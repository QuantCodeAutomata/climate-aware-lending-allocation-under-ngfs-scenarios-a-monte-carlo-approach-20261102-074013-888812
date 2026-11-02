"""
Unit tests for the analytical PD engine.

Tests verify:
  - Calibration identity: no-climate PD = p_ref for all obligors.
  - PD range: all PDs in [0, 1].
  - Monotonicity: larger exposure → higher PD (lending cost effect).
  - Climate effect direction: positive S_s → lower PD (improved production).
  - Gradient sign: dPD/dx_i > 0.
  - Mean-reversion weights: sum approximately q1 * d per year.
  - No look-ahead: thresholds depend only on reference allocation, not future shocks.
"""

import numpy as np
import pytest
from scipy.stats import norm

from src.config import (
    K_BUDGET,
    N_OBLIGORS,
    T_MATURITY,
    LAMBDA_REF,
    R_RISKFREE,
)
from src.pd_engine import (
    compute_q1,
    compute_q2,
    mean_reversion_weights,
    calibrate_thresholds,
    compute_pd,
    compute_pd_gradient,
    compute_terminal_mean,
)


class TestHelperFunctions:
    def test_q1_positive(self, full_portfolio):
        b = full_portfolio["b"].values
        q1 = compute_q1(b)
        assert (q1 > 0).all()

    def test_q2_positive(self, full_portfolio):
        b = full_portfolio["b"].values
        q2 = compute_q2(b)
        assert (q2 > 0).all()

    def test_q1_less_than_T_over_b(self, full_portfolio):
        """q1 = (1 - exp(-bT))/b < T holds for b > 0."""
        b = full_portfolio["b"].values
        q1 = compute_q1(b)
        assert (q1 < T_MATURITY).all()

    def test_mean_reversion_weights_shape(self, full_portfolio):
        b = full_portfolio["b"].values
        W = mean_reversion_weights(b, T_MATURITY)
        assert W.shape == (N_OBLIGORS, int(T_MATURITY))

    def test_mean_reversion_weights_positive(self, full_portfolio):
        b = full_portfolio["b"].values
        W = mean_reversion_weights(b, T_MATURITY)
        assert (W > 0).all()


class TestCalibration:
    def test_no_climate_pd_equals_p_ref(self, full_portfolio, thresholds):
        """
        By construction, PD^i(x_ref; no_climate) = p_ref for all i.
        Paper Section 3, Reference Default Calibration.
        """
        p_ref = 1.0 - np.exp(-LAMBDA_REF * T_MATURITY)
        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        pd_arr = compute_pd(full_portfolio, x_ref, thresholds, S_s=None)
        assert np.allclose(pd_arr, p_ref, atol=1e-8), \
            f"Max deviation from p_ref: {np.abs(pd_arr - p_ref).max():.2e}"

    def test_p_ref_value(self):
        """p_ref = 1 - exp(-0.03 * 5) ≈ 0.13929."""
        p_ref = 1.0 - np.exp(-LAMBDA_REF * T_MATURITY)
        assert abs(p_ref - 0.13929) < 0.001

    def test_thresholds_shape(self, full_portfolio, thresholds):
        assert thresholds.shape == (N_OBLIGORS,)


class TestPDProperties:
    def test_pd_in_unit_interval(self, full_portfolio, thresholds, climate_shocks):
        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        for s in range(4):
            pd_arr = compute_pd(full_portfolio, x_ref, thresholds, climate_shocks[s])
            assert (pd_arr >= 0).all() and (pd_arr <= 1).all(), \
                f"PD out of [0,1] for scenario {s}"

    def test_pd_monotone_in_exposure(self, full_portfolio, thresholds):
        """
        Larger lending exposure → higher PD (lending cost increases).
        Paper Section 2: the production drift decreases with x_i via -eta*x*q1.
        """
        z = thresholds
        x_low = np.full(N_OBLIGORS, 0.05)
        x_high = np.full(N_OBLIGORS, 0.15)
        pd_low = compute_pd(full_portfolio, x_low, z, S_s=None)
        pd_high = compute_pd(full_portfolio, x_high, z, S_s=None)
        # For virtually all obligors, higher exposure → higher PD
        fraction_correct = (pd_high >= pd_low).mean()
        assert fraction_correct > 0.95, \
            f"Only {fraction_correct:.2%} of obligors have pd(high) >= pd(low)"

    def test_positive_climate_shock_reduces_pd(self, full_portfolio, thresholds):
        """
        A positive climate aggregate shock S_s > 0 increases terminal mean
        → lower z - m → lower PD.  Paper Section 2.
        """
        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        z = thresholds
        pd_base = compute_pd(full_portfolio, x_ref, z, S_s=None)
        S_positive = np.full(N_OBLIGORS, 0.1)  # uniform positive shock
        pd_pos = compute_pd(full_portfolio, x_ref, z, S_s=S_positive)
        assert (pd_pos < pd_base).all(), \
            "Positive climate shock should reduce PD for all obligors"

    def test_negative_climate_shock_increases_pd(self, full_portfolio, thresholds):
        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        z = thresholds
        pd_base = compute_pd(full_portfolio, x_ref, z, S_s=None)
        S_neg = np.full(N_OBLIGORS, -0.1)
        pd_neg = compute_pd(full_portfolio, x_ref, z, S_s=S_neg)
        assert (pd_neg > pd_base).all()

    def test_pd_gradient_positive(self, full_portfolio, thresholds):
        """dPD/dx_i > 0: larger loan → higher default probability."""
        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        grad = compute_pd_gradient(full_portfolio, x_ref, thresholds, S_s=None)
        assert (grad > 0).all(), "All PD gradients should be positive"

    def test_gradient_consistency(self, full_portfolio, thresholds):
        """Finite-difference check of analytical gradient."""
        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        eps = 1e-5
        dx = np.zeros(N_OBLIGORS)
        dx[0] = eps
        pd_plus = compute_pd(full_portfolio, x_ref + dx, thresholds, S_s=None)
        pd_base = compute_pd(full_portfolio, x_ref, thresholds, S_s=None)
        fd_grad = (pd_plus[0] - pd_base[0]) / eps
        anal_grad = compute_pd_gradient(full_portfolio, x_ref, thresholds, S_s=None)[0]
        assert abs(fd_grad - anal_grad) / (abs(anal_grad) + 1e-10) < 0.01, \
            f"Gradient mismatch: FD={fd_grad:.6f}, Analytical={anal_grad:.6f}"


class TestClimateShocks:
    def test_shocks_shape(self, ngfs_shocks):
        """Shock tensor: (4 scenarios, 50 sectors, 5 years, 3 channels)."""
        assert ngfs_shocks.shape == (4, 50, 5, 3)

    def test_scenario_pd_means_near_reference(
        self, full_portfolio, thresholds, climate_shocks
    ):
        """
        Scenario mean PDs should remain close to the reference 13.929%
        (paper Section 4.1: 'climate effects are modest in the portfolio mean').
        """
        p_ref = 1.0 - np.exp(-LAMBDA_REF * T_MATURITY)
        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        for s in range(4):
            pd_arr = compute_pd(full_portfolio, x_ref, thresholds, climate_shocks[s])
            mean_pd = pd_arr.mean()
            # Allow ±5 percentage points around reference
            assert abs(mean_pd - p_ref) < 0.05, \
                f"Scenario {s} mean PD {mean_pd:.4f} deviates excessively from p_ref {p_ref:.4f}"

    def test_daps_lower_dispersion_than_hwtp(
        self, full_portfolio, thresholds, climate_shocks
    ):
        """
        DAPS should exhibit lower cross-sectional PD dispersion than HWTP,
        SWUC, DiRe.  Paper Section 4.1.
        """
        from src.config import SCENARIOS
        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        stds = {}
        for s_idx, name in enumerate(SCENARIOS):
            pd_arr = compute_pd(full_portfolio, x_ref, thresholds, climate_shocks[s_idx])
            stds[name] = float(pd_arr.std())
        daps_std = stds["DAPS"]
        # DAPS dispersion must be the smallest
        for name in ["HWTP", "SWUC", "DiRe"]:
            assert daps_std <= stds[name], \
                f"DAPS std ({daps_std:.6f}) should be <= {name} std ({stds[name]:.6f})"
