"""
Unit tests for Monte Carlo simulation components.

Tests verify:
  - Systemic covariance matrix is symmetric and PSD.
  - PCA eigenvalues are non-negative and in descending order.
  - Variance explained by q=5 components exceeds 99.99%.
  - Gaussian latent variables have approximately zero mean and unit variance.
  - Default indicators are binary (0/1).
  - Portfolio losses are non-negative and bounded by LGD * K.
  - ES_99 >= mean loss.
  - Student-t copula transformation preserves the approximate latent
    distribution shape while altering the tail.
  - Simulated marginal PDs agree with analytical PDs within Monte Carlo error.
"""

import numpy as np
import pytest
from scipy.stats import norm

from src.config import (
    K_BUDGET,
    N_OBLIGORS,
    LGD,
    Q_PCA,
    M_DRAWS,
    MC_SEED,
    T_MATURITY,
    LAMBDA_REF,
)
from src.simulation import (
    build_systemic_covariance,
    pca_decompose,
    compute_idiosyncratic_variance,
    simulate_gaussian_latents,
    apply_t_copula,
    default_indicators,
    portfolio_losses,
    expected_shortfall,
    paired_loss_comparison,
)
from src.pd_engine import calibrate_thresholds, compute_pd


class TestSystemicCovariance:
    def test_shape(self, full_portfolio):
        N = len(full_portfolio)
        sigma = build_systemic_covariance(full_portfolio)
        assert sigma.shape == (N, N)

    def test_symmetric(self, systemic_cov):
        assert np.allclose(systemic_cov, systemic_cov.T, atol=1e-12)

    def test_diagonal_non_negative(self, systemic_cov):
        assert (np.diag(systemic_cov) >= 0).all()

    def test_psd(self, systemic_cov):
        """Eigenvalues of a PSD matrix are all >= 0 (up to numerical noise)."""
        eigs = np.linalg.eigvalsh(systemic_cov)
        assert (eigs >= -1e-8).all(), f"Min eigenvalue: {eigs.min():.2e}"


class TestPCADecomposition:
    def test_eigenvalues_descending(self, systemic_cov):
        eigs, _, _ = pca_decompose(systemic_cov, Q_PCA)
        assert (np.diff(eigs) <= 0).all(), "Eigenvalues not in descending order"

    def test_eigenvalues_non_negative(self, systemic_cov):
        eigs, _, _ = pca_decompose(systemic_cov, Q_PCA)
        assert (eigs >= 0).all()

    def test_eigenvectors_shape(self, systemic_cov):
        eigs, uvecs, _ = pca_decompose(systemic_cov, Q_PCA)
        assert uvecs.shape == (N_OBLIGORS, Q_PCA)

    def test_var_explained_high(self, systemic_cov):
        """q = 5 components should explain > 99.99% of systemic variance."""
        _, _, var_exp = pca_decompose(systemic_cov, Q_PCA)
        assert var_exp > 0.9999, \
            f"Expected > 99.99% var explained with q={Q_PCA}, got {var_exp*100:.6f}%"

    def test_var_explained_increases_with_q(self, systemic_cov):
        for q1, q2 in [(2, 5), (5, 10)]:
            _, _, ve1 = pca_decompose(systemic_cov, q1)
            _, _, ve2 = pca_decompose(systemic_cov, q2)
            assert ve2 >= ve1


class TestLatentSimulation:
    @pytest.fixture(scope="class")
    def latents(self, full_portfolio, systemic_cov):
        eigs, uvecs, _ = pca_decompose(systemic_cov, Q_PCA)
        delta = compute_idiosyncratic_variance(systemic_cov, eigs, uvecs)
        Y = simulate_gaussian_latents(
            full_portfolio, eigs, uvecs, delta, M=1000, seed=42
        )
        return Y

    def test_shape(self, latents):
        assert latents.shape == (1000, N_OBLIGORS)

    def test_approximately_zero_mean(self, latents):
        """
        Column means should be near zero.

        With M=1000 draws and N=1000 obligors the Monte Carlo SE per column is
        1/sqrt(1000) ≈ 0.032; the expected maximum over 1000 columns is roughly
        3.5 SE ≈ 0.112.  We allow up to 0.15 (≈4.7 SE) to avoid spurious
        failures from random seed variation.
        """
        col_means = latents.mean(axis=0)
        assert np.abs(col_means).max() < 0.15, \
            f"Max column mean: {np.abs(col_means).max():.4f}"

    def test_approximately_unit_variance(self, latents):
        """Column variances should be near 1."""
        col_vars = latents.var(axis=0)
        # Allow some deviation due to finite sample size
        assert np.abs(col_vars - 1).max() < 0.3, \
            f"Max column variance deviation from 1: {np.abs(col_vars - 1).max():.4f}"


class TestDefaultIndicators:
    def test_binary(self, full_portfolio, thresholds):
        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        pd_arr = compute_pd(full_portfolio, x_ref, thresholds, S_s=None)
        Y = np.random.default_rng(1).standard_normal((100, N_OBLIGORS))
        D = default_indicators(Y, pd_arr)
        unique = np.unique(D)
        assert set(unique).issubset({0.0, 1.0})

    def test_rate_matches_pd(self, full_portfolio, thresholds):
        """With many paths, empirical default rate should match analytical PD."""
        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        pd_arr = compute_pd(full_portfolio, x_ref, thresholds, S_s=None)
        rng = np.random.default_rng(77)
        Y = rng.standard_normal((10000, N_OBLIGORS))
        D = default_indicators(Y, pd_arr)
        sim_rates = D.mean(axis=0)
        # For independent normals, empirical rates should be close to analytical
        # Tolerance: 2 standard errors
        se = np.sqrt(pd_arr * (1 - pd_arr) / 10000)
        within_2se = np.abs(sim_rates - pd_arr) <= 2 * se
        fraction_ok = within_2se.mean()
        # By 95% rule, expect ≥ 90% within 2 SE
        assert fraction_ok >= 0.90, f"Only {fraction_ok:.2%} within 2 SE"


class TestPortfolioLosses:
    def test_non_negative(self, full_portfolio, thresholds):
        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        pd_arr = compute_pd(full_portfolio, x_ref, thresholds, S_s=None)
        Y = np.random.default_rng(2).standard_normal((100, N_OBLIGORS))
        D = default_indicators(Y, pd_arr)
        L = portfolio_losses(D, x_ref, lgd=LGD)
        assert (L >= 0).all()

    def test_bounded_by_lgd_times_K(self, full_portfolio, thresholds):
        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        pd_arr = compute_pd(full_portfolio, x_ref, thresholds, S_s=None)
        Y = np.random.default_rng(3).standard_normal((100, N_OBLIGORS))
        D = default_indicators(Y, pd_arr)
        L = portfolio_losses(D, x_ref, lgd=LGD)
        assert (L <= LGD * K_BUDGET + 1e-8).all()


class TestExpectedShortfall:
    def test_es99_ge_mean(self):
        rng = np.random.default_rng(5)
        losses = rng.exponential(scale=1.0, size=5000)
        es99 = expected_shortfall(losses, level=0.99)
        assert es99 >= losses.mean(), "ES_99 must be >= mean loss"

    def test_es99_ge_var(self):
        rng = np.random.default_rng(6)
        losses = rng.exponential(scale=1.0, size=5000)
        var99 = float(np.quantile(losses, 0.99))
        es99 = expected_shortfall(losses, level=0.99)
        assert es99 >= var99, "ES_99 must be >= VaR_99"

    def test_es99_in_range(self):
        rng = np.random.default_rng(7)
        losses = rng.uniform(0, 10, size=5000)
        es99 = expected_shortfall(losses, level=0.99)
        assert 0 <= es99 <= 10


class TestTCopula:
    def test_shape_preserved(self, full_portfolio, systemic_cov):
        eigs, uvecs, _ = pca_decompose(systemic_cov, Q_PCA)
        delta = compute_idiosyncratic_variance(systemic_cov, eigs, uvecs)
        Y_g = simulate_gaussian_latents(
            full_portfolio, eigs, uvecs, delta, M=200, seed=42
        )
        Y_t = apply_t_copula(Y_g, nu=4, seed=99)
        assert Y_t.shape == Y_g.shape

    def test_t_copula_has_heavier_tails(self, full_portfolio, systemic_cov):
        """
        With t-copula, extreme joint defaults should be more common.
        ES_99 of t-copula losses should exceed Gaussian (on average).
        """
        eigs, uvecs, _ = pca_decompose(systemic_cov, Q_PCA)
        delta = compute_idiosyncratic_variance(systemic_cov, eigs, uvecs)
        Y_g = simulate_gaussian_latents(
            full_portfolio, eigs, uvecs, delta, M=2000, seed=42
        )
        Y_t = apply_t_copula(Y_g, nu=4, seed=99)

        x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
        z = calibrate_thresholds(full_portfolio)
        pd_arr = compute_pd(full_portfolio, x_ref, z, S_s=None)

        D_g = default_indicators(Y_g, pd_arr)
        D_t = default_indicators(Y_t, pd_arr)
        L_g = portfolio_losses(D_g, x_ref)
        L_t = portfolio_losses(D_t, x_ref)
        es_g = expected_shortfall(L_g)
        es_t = expected_shortfall(L_t)
        assert es_t > es_g, \
            f"t-copula ES_99 ({es_t:.4f}) should exceed Gaussian ({es_g:.4f})"


class TestPairedComparison:
    def test_output_keys(self):
        rng = np.random.default_rng(10)
        la = rng.exponential(1.0, 1000)
        lb = rng.exponential(1.2, 1000)
        result = paired_loss_comparison(la, lb)
        for key in ["mean_diff", "ci_lower", "ci_upper", "se"]:
            assert key in result

    def test_ci_contains_mean(self):
        rng = np.random.default_rng(11)
        la = rng.exponential(1.0, 1000)
        lb = rng.exponential(1.0, 1000)  # same distribution
        result = paired_loss_comparison(la, lb)
        assert result["ci_lower"] <= result["mean_diff"] <= result["ci_upper"]
