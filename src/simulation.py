"""
Monte Carlo joint default and portfolio loss simulation.

Implements the PCA-based Gaussian latent-factor model described in paper
Section 2.4–2.5, plus the Student-t copula robustness transformation
(Section 5.2).

Systemic covariance (paper Section 2.4):
    K_{X_T}^{i,j} = rho^i rho^j (1 - exp(-(b^i + b^j) T)) / (b^i + b^j)

PCA decomposition:
    Retain q leading eigenpairs; form approximation K^(q) and residual
    delta_i = K^{i,i}_{X_T} - K^{(q),i,i}_{X_T}.

Latent variable construction per path m (paper Section 2.5):
    Y_i = [sum_{k=1}^q sqrt(nu^k) u_i^k G^k
           + sqrt((1 - rho_i^2) q2^i + delta_i) Z_i] / sqrt(q2^i)

Default indicator:
    D_i = 1{Y_i <= Phi^{-1}(PD_T^i)}

Portfolio loss:
    L = LGD * sum_i x_i D_i

Student-t copula robustness (paper Section 5.2):
    T_i = Y_i^{(G)} / sqrt(W/nu),   W ~ chi^2_nu
    U_i = F_{t_nu}(T_i)
    Y_i^{(t)} = Phi^{-1}(U_i)

Paper: "Climate aware lending allocation under NGFS scenarios –
        A Monte Carlo approach", Sections 2.4, 2.5, 5.2.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.stats import norm, t as t_dist, chi2

from src.config import (
    T_MATURITY,
    M_DRAWS,
    Q_PCA,
    NU_T_COPULA,
    LGD,
    ES_LEVEL,
    MC_SEED,
)
from src.pd_engine import compute_q2


# ── Covariance and PCA ───────────────────────────────────────────────────────

def build_systemic_covariance(portfolio: "pd.DataFrame") -> np.ndarray:
    """
    Build the N×N systemic covariance matrix K_{X_T}.

        K_{X_T}^{i,j} = rho^i rho^j (1 - exp(-(b^i + b^j) T)) / (b^i + b^j)

    Paper Section 2.4.

    Parameters
    ----------
    portfolio : DataFrame from generate_portfolio()

    Returns
    -------
    Sigma : (N, N) symmetric positive-semi-definite matrix
    """
    T = T_MATURITY
    b = portfolio["b"].values  # (N,)
    rho = portfolio["rho"].values  # (N,)

    # Broadcasting: b_i + b_j
    b_sum = b[:, None] + b[None, :]  # (N, N)
    cov = (rho[:, None] * rho[None, :]) * (1.0 - np.exp(-b_sum * T)) / b_sum
    return cov


def pca_decompose(
    sigma: np.ndarray, q: int = Q_PCA
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Eigenvalue decomposition of the systemic covariance matrix, retaining
    the q leading components (paper Section 2.4).

    Parameters
    ----------
    sigma : (N, N) systemic covariance matrix
    q     : number of components to retain

    Returns
    -------
    eigenvalues  : (q,) array of retained eigenvalues (descending)
    eigenvectors : (N, q) array of corresponding eigenvectors
    var_explained: fraction of total systemic variance explained by q components
    """
    # numpy.linalg.eigh returns eigenvalues in ascending order
    eigenvalues_all, eigenvectors_all = np.linalg.eigh(sigma)

    # Sort descending
    idx = np.argsort(eigenvalues_all)[::-1]
    eigenvalues_all = eigenvalues_all[idx]
    eigenvectors_all = eigenvectors_all[:, idx]

    # Clip tiny negative eigenvalues from numerical noise
    eigenvalues_all = np.maximum(eigenvalues_all, 0.0)

    total_var = eigenvalues_all.sum()
    eigenvalues_q = eigenvalues_all[:q]
    eigenvectors_q = eigenvectors_all[:, :q]  # (N, q)

    var_explained = eigenvalues_q.sum() / total_var if total_var > 0 else 0.0

    return eigenvalues_q, eigenvectors_q, var_explained


def compute_idiosyncratic_variance(
    sigma: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """
    Compute the per-obligor discarded systemic variance delta_i.

        delta_i = K_{X_T}^{i,i} - sum_{k=1}^q nu^k (u_i^k)^2

    Parameters
    ----------
    sigma        : (N, N) full systemic covariance
    eigenvalues  : (q,) retained eigenvalues
    eigenvectors : (N, q) retained eigenvectors

    Returns
    -------
    delta : (N,) residual systemic variance per obligor (clipped at 0)
    """
    full_diag = np.diag(sigma)  # K^{i,i}
    approx_diag = np.sum(eigenvalues[None, :] * eigenvectors ** 2, axis=1)
    delta = np.maximum(full_diag - approx_diag, 0.0)
    return delta


# ── Latent-variable simulation ───────────────────────────────────────────────

def simulate_gaussian_latents(
    portfolio: "pd.DataFrame",
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    delta: np.ndarray,
    M: int = M_DRAWS,
    seed: int = MC_SEED,
) -> np.ndarray:
    """
    Simulate M standardised Gaussian latent variables Y_i^{(m)}.

    Construction (paper Section 2.5):
        Y_i = [sum_{k=1}^q sqrt(nu^k) u_i^k G^k
               + sqrt((1-rho_i^2) q2^i + delta_i) Z_i] / sqrt(q2^i)

    Parameters
    ----------
    portfolio    : obligor DataFrame
    eigenvalues  : (q,) PCA eigenvalues
    eigenvectors : (N, q) PCA eigenvectors
    delta        : (N,) residual systemic variance
    M            : number of paths
    seed         : random seed

    Returns
    -------
    Y : (M, N) array of standardised latent variables
    """
    rng = np.random.default_rng(seed)
    q = len(eigenvalues)
    N = len(portfolio)
    T = T_MATURITY

    b = portfolio["b"].values
    rho = portfolio["rho"].values
    q2 = compute_q2(b, T)

    sqrt_nu = np.sqrt(np.maximum(eigenvalues, 0.0))  # (q,)

    # Systematic component matrix: (N, q); each column scaled by sqrt(nu^k)
    A = eigenvectors * sqrt_nu[None, :]  # (N, q)

    # G: (M, q) common factors; Z: (M, N) idiosyncratic
    G = rng.standard_normal((M, q))   # (M, q)
    Z = rng.standard_normal((M, N))   # (M, N)

    # Systematic part: (M, N) = G @ A.T
    systematic = G @ A.T  # (M, N)

    # Idiosyncratic variance: (1-rho_i^2) q2^i + delta_i
    idio_var = (1.0 - rho ** 2) * q2 + delta  # (N,)
    idio_std = np.sqrt(np.maximum(idio_var, 0.0))  # (N,)

    # Numerator: systematic + idiosyncratic noise
    num = systematic + idio_std[None, :] * Z  # (M, N)

    # Normalise by sqrt(q2^i) to get unit-variance latent
    Y = num / np.sqrt(q2)[None, :]  # (M, N)

    return Y


def apply_t_copula(
    Y_gauss: np.ndarray,
    nu: int = NU_T_COPULA,
    seed: int = MC_SEED + 1,
) -> np.ndarray:
    """
    Transform Gaussian latent draws to preserve marginal PDs but use a
    Student-t copula (paper Section 5.2).

    Steps:
      1. Draw W^{(m)} ~ chi^2_nu independently for each path.
      2. T_i^{(m)} = Y_i^{(G,m)} / sqrt(W^{(m)} / nu)  →  t_nu marginals.
      3. U_i^{(m)} = F_{t_nu}(T_i^{(m)})               →  uniform marginals.
      4. Y_i^{(t,m)} = Phi^{-1}(U_i^{(m)})             →  standard-normal
                                                             marginals with
                                                             t-copula dependence.

    Parameters
    ----------
    Y_gauss : (M, N) Gaussian latent array from simulate_gaussian_latents
    nu      : degrees of freedom
    seed    : additional seed for chi-squared draws

    Returns
    -------
    Y_t : (M, N) transformed latents with t-copula dependence
    """
    rng = np.random.default_rng(seed)
    M, N = Y_gauss.shape

    # One shared chi^2 scaling variable per path (common W)
    W = rng.chisquare(nu, size=M)  # (M,)

    # t-distributed values
    T_vals = Y_gauss / np.sqrt(W[:, None] / nu)  # (M, N)

    # Map to uniform via t_nu CDF, then back to normal via Phi^{-1}
    U = t_dist.cdf(T_vals, df=nu)  # (M, N)
    # Clip to avoid Phi^{-1}(0) = -inf or Phi^{-1}(1) = +inf
    U = np.clip(U, 1e-10, 1.0 - 1e-10)
    Y_t = norm.ppf(U)  # (M, N)

    return Y_t


# ── Default indicators and portfolio losses ──────────────────────────────────

def default_indicators(
    Y: np.ndarray,
    pd_arr: np.ndarray,
) -> np.ndarray:
    """
    Compute default indicator matrix.

        D_i^{(m)} = 1 { Y_i^{(m)} <= Phi^{-1}(PD_T^i) }

    Parameters
    ----------
    Y      : (M, N) latent draws
    pd_arr : (N,) analytical terminal default probabilities

    Returns
    -------
    D : (M, N) boolean (0/1) default indicators
    """
    thresholds = norm.ppf(np.clip(pd_arr, 1e-10, 1.0 - 1e-10))  # (N,)
    D = (Y <= thresholds[None, :]).astype(np.float32)
    return D


def portfolio_losses(
    D: np.ndarray,
    x: np.ndarray,
    lgd: float = LGD,
) -> np.ndarray:
    """
    Compute portfolio credit loss on each Monte Carlo path.

        L^{(m)} = LGD * sum_i x_i D_i^{(m)}

    Parameters
    ----------
    D   : (M, N) default indicator matrix
    x   : (N,) exposure allocation
    lgd : loss given default

    Returns
    -------
    L : (M,) loss array
    """
    return lgd * (D @ x)  # (M,)


def expected_shortfall(losses: np.ndarray, level: float = ES_LEVEL) -> float:
    """
    Estimate Expected Shortfall at confidence level `level` from an empirical
    loss distribution using tail averaging (paper Section 2.5).

    ES_{level} = mean(L | L >= quantile_{level}(L))

    Parameters
    ----------
    losses : (M,) simulated loss array
    level  : confidence level (e.g. 0.99)

    Returns
    -------
    es : float
    """
    q = np.quantile(losses, level)
    tail = losses[losses >= q]
    return float(np.mean(tail)) if len(tail) > 0 else float(q)


# ── High-level simulation runner ─────────────────────────────────────────────

def run_simulation(
    portfolio: "pd.DataFrame",
    pd_arr: np.ndarray,
    x: np.ndarray,
    q: int = Q_PCA,
    use_t_copula: bool = False,
    nu: int = NU_T_COPULA,
    M: int = M_DRAWS,
    seed: int = MC_SEED,
    Y_precomputed: Optional[np.ndarray] = None,
) -> dict:
    """
    Run a complete Monte Carlo simulation for one policy/scenario pair.

    Parameters
    ----------
    portfolio      : obligor DataFrame
    pd_arr         : (N,) analytical terminal PDs for this policy/scenario
    x              : (N,) exposure allocation
    q              : number of PCA components
    use_t_copula   : if True, apply Student-t copula transformation
    nu             : t-copula degrees of freedom
    M              : number of Monte Carlo paths
    seed           : random seed
    Y_precomputed  : optional pre-simulated (M, N) Gaussian latent array
                     (for paired comparisons across policies)

    Returns
    -------
    dict with keys:
      Y            : (M, N) Gaussian latent draws (for paired reuse)
      losses_gauss : (M,) Gaussian losses
      losses_t     : (M,) t-copula losses (or None if use_t_copula=False)
      mean_loss    : float
      es99         : float  Expected Shortfall at 99%
      es99_t       : float or None
      sim_pd       : (N,) simulated marginal PDs (for validation)
      eigenvalues  : (q,) retained eigenvalues
      var_explained: fraction of systemic variance explained
    """
    # 1. Build covariance and PCA
    sigma = build_systemic_covariance(portfolio)
    eigenvalues, eigenvectors, var_explained = pca_decompose(sigma, q)
    delta = compute_idiosyncratic_variance(sigma, eigenvalues, eigenvectors)

    # 2. Simulate Gaussian latents (or reuse precomputed for paired comparison)
    if Y_precomputed is not None:
        Y = Y_precomputed
    else:
        Y = simulate_gaussian_latents(
            portfolio, eigenvalues, eigenvectors, delta, M=M, seed=seed
        )

    # 3. Gaussian default indicators and losses
    D_gauss = default_indicators(Y, pd_arr)
    losses_gauss = portfolio_losses(D_gauss, x)

    # 4. Simulated marginal PDs (for validation)
    sim_pd = D_gauss.mean(axis=0)

    result = {
        "Y": Y,
        "losses_gauss": losses_gauss,
        "losses_t": None,
        "mean_loss": float(losses_gauss.mean()),
        "es99": expected_shortfall(losses_gauss, ES_LEVEL),
        "es99_t": None,
        "sim_pd": sim_pd,
        "eigenvalues": eigenvalues,
        "var_explained": var_explained,
    }

    # 5. Student-t copula robustness
    if use_t_copula:
        Y_t = apply_t_copula(Y, nu=nu, seed=seed + 1000)
        D_t = default_indicators(Y_t, pd_arr)
        losses_t = portfolio_losses(D_t, x)
        result["losses_t"] = losses_t
        result["es99_t"] = expected_shortfall(losses_t, ES_LEVEL)

    return result


def paired_loss_comparison(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    ci_level: float = 0.95,
) -> dict:
    """
    Paired comparison of two loss distributions using common latent draws.

    Computes the mean pathwise loss difference (a - b) and a two-sided
    confidence interval via t-based inference on the differences.

    Parameters
    ----------
    losses_a : (M,) loss array for policy A (e.g. climate-aware)
    losses_b : (M,) loss array for policy B (e.g. conventional)
    ci_level : confidence level for the interval

    Returns
    -------
    dict with keys: mean_diff, ci_lower, ci_upper, se
    """
    diff = losses_b - losses_a  # positive means A has lower loss
    M = len(diff)
    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(M))
    # t-critical value (M large → z-approx accurate)
    from scipy.stats import t as scipy_t
    tc = float(scipy_t.ppf(0.5 + ci_level / 2, df=M - 1))
    return {
        "mean_diff": mean_diff,
        "ci_lower": mean_diff - tc * se,
        "ci_upper": mean_diff + tc * se,
        "se": se,
    }
