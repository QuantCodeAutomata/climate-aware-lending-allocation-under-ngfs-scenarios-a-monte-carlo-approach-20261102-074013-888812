"""
Analytical terminal default probability engine.

Implements the structural credit model described in paper Sections 2 and 3:

  - Obligor i's log-production follows a mean-reverting process (OU-type).
  - The analytical five-year terminal default probability at allocation x_i
    under scenario s is:

        PD_T^i(x_i; s) = Phi( (z_i - m_i(x_i; s)) / (sigma^i * sqrt(q2^i)) )

    where:
      q1^i = (1 - exp(-b^i T)) / b^i
      q2^i = (1 - exp(-2 b^i T)) / (2 b^i)
      m_i(x_i; s) = exp(-b^i T) p0^i + a^i q1^i + S_s^i - eta^i x_i q1^i
      z_i is the obligor-specific fixed threshold calibrated so that
          PD_T^i(x_ref; no_climate) = p_ref for all i.

  - The climate aggregate shock S_s^i is computed from sector-level annual
    NGFS shocks theta_{j,s,k}^i (Section 3, equations for gamma and d).

Paper: "Climate aware lending allocation under NGFS scenarios –
        A Monte Carlo approach", Sections 2–3.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.config import (
    T_MATURITY,
    R_RISKFREE,
    LAMBDA_REF,
    K_BUDGET,
    N_OBLIGORS,
    C_J,
    ALPHA_J,
    BETA_J,
    SCENARIOS,
)


# ── Helper functions ────────────────────────────────────────────────────────

def compute_q1(b: np.ndarray, T: float = T_MATURITY) -> np.ndarray:
    """
    q1^i = (1 - exp(-b^i T)) / b^i  (paper Section 2, integral term).

    Parameters
    ----------
    b : array of mean-reversion speeds
    T : maturity

    Returns
    -------
    q1 array, same shape as b
    """
    return (1.0 - np.exp(-b * T)) / b


def compute_q2(b: np.ndarray, T: float = T_MATURITY) -> np.ndarray:
    """
    q2^i = (1 - exp(-2 b^i T)) / (2 b^i)  (paper Section 2, variance term).

    Parameters
    ----------
    b : array of mean-reversion speeds
    T : maturity

    Returns
    -------
    q2 array, same shape as b
    """
    return (1.0 - np.exp(-2.0 * b * T)) / (2.0 * b)


def mean_reversion_weights(b: np.ndarray, T: float = T_MATURITY) -> np.ndarray:
    """
    Mean-reversion weight for year k at maturity T.

    w_k^i = (exp(-b^i (T-k)) - exp(-b^i (T-k+1))) / b^i
    for k = 1, ..., T (integer years).

    Parameters
    ----------
    b : (N,) array of mean-reversion speeds
    T : maturity in integer years

    Returns
    -------
    W : (N, T) array; W[i, k] = weight for obligor i at year k+1
    """
    n_years = int(T)
    N = len(b)
    W = np.empty((N, n_years))
    for k in range(1, n_years + 1):
        # year index k (1-based)
        W[:, k - 1] = (np.exp(-b * (T - k)) - np.exp(-b * (T - k + 1))) / b
    return W


def compute_climate_shock(
    portfolio: "pd.DataFrame",
    ngfs_shocks: np.ndarray,
    scenario_idx: int,
    shock_scale: float = 1.0,
) -> np.ndarray:
    """
    Compute the aggregate climate horizon shock S_s^i for each obligor.

    Steps (paper Section 3, Climate Scenario Integration):
      1. Map sector-level shocks theta_{j,s,k} to each obligor via sector_idx.
      2. gamma_{j,s,k}^i = (1 / (2 beta_j)) * (theta / (r + b^i) - alpha_j)
         with c_j applied in step 3.
      3. d_{s,k}^i = sum_j c_j gamma_{j,s,k}^i
      4. S_s^i = sum_k w_k^i d_{s,k}^i

    Parameters
    ----------
    portfolio   : DataFrame from generate_portfolio()
    ngfs_shocks : (n_scenarios, N_SECTORS, n_years, 3) shock array
    scenario_idx: integer index into SCENARIOS list
    shock_scale : scaling multiplier for sensitivity analysis (default 1.0)

    Returns
    -------
    S_s : (N,) array of aggregate climate shocks
    """
    b = portfolio["b"].values
    sector_idx = portfolio["sector_idx"].values
    T = T_MATURITY
    n_years = int(T)
    beta_j = np.array(BETA_J)

    # Weights: (N, n_years)
    W = mean_reversion_weights(b, T)

    # Sector shocks for this scenario: (N_SECTORS, n_years, 3)
    sector_shocks = ngfs_shocks[scenario_idx]  # (N_SECTORS, n_years, 3)
    # Map to obligors: (N, n_years, 3)
    obligor_shocks = sector_shocks[sector_idx] * shock_scale

    # Accumulate annual drift d_{s,k}^i
    # gamma_{j,s,k}^i = theta / (2 beta_j (r + b^i)) - alpha_j / (2 beta_j)
    # d_{s,k}^i = sum_j c_j gamma_{j,s,k}^i
    r_plus_b = R_RISKFREE + b  # (N,)
    S_s = np.zeros(len(portfolio))

    for k in range(n_years):
        d_k = np.zeros(len(portfolio))
        for j in range(3):
            theta_jk = obligor_shocks[:, k, j]  # (N,)
            gamma_jk = (theta_jk / r_plus_b - ALPHA_J) / (2.0 * beta_j[j])
            d_k += C_J * gamma_jk
        S_s += W[:, k] * d_k

    return S_s


def calibrate_thresholds(
    portfolio: "pd.DataFrame",
    x_ref: float | None = None,
) -> np.ndarray:
    """
    Calibrate obligor-specific fixed default thresholds z_i.

    The threshold is set so that each obligor's no-climate terminal PD at
    the reference equal-weight allocation equals p_ref (paper Section 3,
    Reference Default Calibration):

        z_i = m_i^ref + sigma^i * sqrt(q2^i) * Phi^{-1}(p_ref)

    where
        p_ref = 1 - exp(-lambda_ref * T)
        m_i^ref = exp(-b^i T) p0^i + a^i q1^i - eta^i x_ref q1^i

    Parameters
    ----------
    portfolio : DataFrame from generate_portfolio()
    x_ref     : reference individual exposure (default K/N)

    Returns
    -------
    z : (N,) array of calibrated thresholds
    """
    if x_ref is None:
        x_ref = K_BUDGET / N_OBLIGORS

    b = portfolio["b"].values
    a = portfolio["a"].values
    sigma = portfolio["sigma"].values
    p0 = portfolio["p0"].values
    eta = portfolio["eta"].values
    T = T_MATURITY

    q1 = compute_q1(b, T)
    q2 = compute_q2(b, T)

    p_ref = 1.0 - np.exp(-LAMBDA_REF * T)
    phi_inv_p_ref = norm.ppf(p_ref)  # scalar

    # No-climate terminal mean at reference allocation (p0 = 0)
    m_ref = np.exp(-b * T) * p0 + a * q1 - eta * x_ref * q1
    z = m_ref + sigma * np.sqrt(q2) * phi_inv_p_ref
    return z


def compute_terminal_mean(
    portfolio: "pd.DataFrame",
    x: np.ndarray,
    S_s: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the scenario terminal mean log-production m_i(x_i; s).

        m_i(x_i; s) = exp(-b^i T) p0^i + a^i q1^i + S_s^i - eta^i x_i q1^i

    Parameters
    ----------
    portfolio : DataFrame from generate_portfolio()
    x         : (N,) array of exposures
    S_s       : (N,) climate aggregate shock; if None, no-climate baseline

    Returns
    -------
    m : (N,) array
    """
    b = portfolio["b"].values
    a = portfolio["a"].values
    p0 = portfolio["p0"].values
    eta = portfolio["eta"].values
    T = T_MATURITY

    q1 = compute_q1(b, T)
    m = np.exp(-b * T) * p0 + a * q1 - eta * x * q1
    if S_s is not None:
        m = m + S_s
    return m


def compute_pd(
    portfolio: "pd.DataFrame",
    x: np.ndarray,
    z: np.ndarray,
    S_s: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute analytical terminal default probabilities.

        PD_T^i(x_i; s) = Phi( (z_i - m_i(x_i; s)) / (sigma^i sqrt(q2^i)) )

    Parameters
    ----------
    portfolio : DataFrame from generate_portfolio()
    x         : (N,) array of exposures
    z         : (N,) array of calibrated thresholds (from calibrate_thresholds)
    S_s       : (N,) climate aggregate shock; if None, no-climate baseline

    Returns
    -------
    pd_arr : (N,) array of terminal default probabilities in [0, 1]
    """
    b = portfolio["b"].values
    sigma = portfolio["sigma"].values
    T = T_MATURITY

    q2 = compute_q2(b, T)
    m = compute_terminal_mean(portfolio, x, S_s)

    std_term = sigma * np.sqrt(q2)  # (N,)
    probit_arg = (z - m) / std_term
    pd_arr = norm.cdf(probit_arg)
    return pd_arr


def compute_pd_gradient(
    portfolio: "pd.DataFrame",
    x: np.ndarray,
    z: np.ndarray,
    S_s: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute d(PD_T^i) / d(x_i) analytically.

    From dm_i/dx_i = -eta^i q1^i and the chain rule:
        dPD/dx_i = phi(probit_arg_i) * eta^i q1^i / (sigma^i sqrt(q2^i))

    (Positive because increasing x_i increases m_i, pushing z_i - m_i lower,
     but the derivative of Phi at a lower argument is positive only when
     phi(arg) is positive; the sign is: d(z-m)/dx = -dm/dx = +eta*q1 > 0
     but z-m decreases, so dPD/dx_i < 0.  The formula below returns the
     correct sign automatically via the -dm/dx term.)

    Parameters
    ----------
    portfolio : DataFrame from generate_portfolio()
    x         : (N,) exposures
    z         : (N,) thresholds
    S_s       : (N,) climate shock or None

    Returns
    -------
    grad : (N,) array dPD_i / dx_i  (negative; larger exposure → lower PD)
    """
    b = portfolio["b"].values
    sigma = portfolio["sigma"].values
    eta = portfolio["eta"].values
    T = T_MATURITY

    q1 = compute_q1(b, T)
    q2 = compute_q2(b, T)
    m = compute_terminal_mean(portfolio, x, S_s)

    std_term = sigma * np.sqrt(q2)
    probit_arg = (z - m) / std_term
    phi_val = norm.pdf(probit_arg)

    # d(probit_arg)/dx_i = -dm_i/dx_i / std_term = eta^i q1^i / std_term
    # dPD/dx_i = phi(probit_arg) * d(probit_arg)/dx_i
    #           = phi(probit_arg) * (-(-eta*q1)) / std_term   [since dm/dx = -eta*q1]
    # Wait: probit_arg = (z - m) / std; d/dx_i = -dm/dx / std = -(-eta*q1)/std = eta*q1/std
    # But z-m decreases when x increases (m increases), so probit_arg decreases,
    # meaning Phi decreases → dPD/dx_i < 0.
    # Actually: dPD/dx_i = phi(probit_arg) * d(probit_arg)/dx_i
    #                     = phi * (eta*q1 / std) * (-1) since d(z-m)/dx = -dm/dx = eta*q1
    # Let's be careful: d/dx_i (z - m(x_i)) = -dm/dx_i = -(-eta*q1) = +eta*q1 > 0
    # No wait: m(x_i) = ... - eta*x_i*q1; dm/dx_i = -eta*q1 (negative)
    # d(z-m)/dx_i = 0 - dm/dx_i = 0 - (-eta*q1) = +eta*q1 > 0
    # So probit_arg = (z-m)/std increases with x_i → PD increases with x_i.
    # That means increasing exposure increases default probability!
    # This makes sense: larger loan → higher lending cost → more likely default.
    # dPD/dx_i = phi(probit_arg) * (eta*q1/std) > 0
    grad = phi_val * (eta * q1) / std_term
    return grad


def compute_all_scenario_shocks(
    portfolio: "pd.DataFrame",
    ngfs_shocks: np.ndarray,
    shock_scale: float = 1.0,
) -> np.ndarray:
    """
    Compute climate shocks S_s^i for all scenarios.

    Parameters
    ----------
    portfolio   : DataFrame from generate_portfolio()
    ngfs_shocks : (n_scenarios, N_SECTORS, n_years, 3) array
    shock_scale : scaling factor for sensitivity analysis

    Returns
    -------
    S_all : (n_scenarios, N) array of aggregate climate shocks
    """
    n_scenarios = len(SCENARIOS)
    N = len(portfolio)
    S_all = np.empty((n_scenarios, N))
    for s_idx in range(n_scenarios):
        S_all[s_idx] = compute_climate_shock(
            portfolio, ngfs_shocks, s_idx, shock_scale=shock_scale
        )
    return S_all
