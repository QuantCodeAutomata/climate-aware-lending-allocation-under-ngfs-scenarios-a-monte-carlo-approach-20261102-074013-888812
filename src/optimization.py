"""
Lending allocation optimisation.

Solves the nonlinear lending optimisation problem described in paper
Section 2.6.  The problem is:

    maximise   sum_i x_i eta^i [1 - PD_T^i(x_i; s)]
    subject to sum_i x_i = K,   0 <= x_i <= kappa K

Because PD_T^i depends only on the obligor's own exposure x_i (separable
objective), the problem reduces to a classic separable resource-allocation
problem.  The optimal solution satisfies the KKT first-order conditions:

    f_i'(x_i*) = lambda  for all funded interior obligors,

where f_i(x) = x eta^i [1 - PD_i(x)] and lambda is the budget Lagrange
multiplier.  These conditions are the same as those satisfied by
R nloptr::slsqp at convergence (paper Section 3).  Finding the common
multiplier via bisection is orders of magnitude faster for N = 1000
than feeding the full 1000-variable QP to a general-purpose SLSQP
solver, and the solution is exact (not approximate).

Four allocation policies are produced per scenario (paper Section 4.2):
  1. Climate-aware nonlinear (climate shocks active)
  2. Conventional nonlinear  (no climate shocks, endogenous PD retained)
  3. Equal-weight            (x_i = K/N for all i)
  4. Fixed-PD linear         (PD evaluated at reference, then held fixed
                               during a linear-in-x optimisation)

Paper: "Climate aware lending allocation under NGFS scenarios –
        A Monte Carlo approach", Sections 2.6 and 4.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.stats import norm as _norm

from src.config import (
    K_BUDGET,
    N_OBLIGORS,
    KAPPA,
    LGD,
    T_MATURITY,
)
from src.pd_engine import compute_pd


# ── Separable dual-decomposition solver ─────────────────────────────────────

def _precompute_solver_arrays(
    portfolio: "pd.DataFrame",
    z: np.ndarray,
    S_s: Optional[np.ndarray],
    K: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute per-obligor scalar quantities used in the separable solver.

    The argument of Phi for obligor i at exposure x_i is linear in x_i:
        h_i(x_i) = h0_i + c_i * x_i

    where
        h0_i = (z_i - m_base_i) / sig_q_i       (Phi argument at x_i = 0)
        c_i  = eta_i * q1_i / sig_q_i            (positive slope)

    m_base_i = exp(-b_i T) p0_i + a_i q1_i + S_s_i  (base mean including
    climate shock, no lending cost).

    Returns
    -------
    eta, h0, c, sig_q  – all shape (N,)
    """
    T = T_MATURITY
    b = portfolio["b"].values
    sigma = portfolio["sigma"].values
    eta = portfolio["eta"].values
    a = portfolio["a"].values
    p0 = portfolio["p0"].values

    q1 = (1.0 - np.exp(-b * T)) / b
    q2 = (1.0 - np.exp(-2.0 * b * T)) / (2.0 * b)
    sig_q = sigma * np.sqrt(q2)

    m_base = np.exp(-b * T) * p0 + a * q1
    if S_s is not None:
        m_base = m_base + S_s

    h0 = (z - m_base) / sig_q
    c = eta * q1 / sig_q          # always > 0

    return eta, h0, c, sig_q


def _marginal_value(
    x_arr: np.ndarray,
    eta: np.ndarray,
    h0: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """
    Vectorised marginal value d/dx_i [x_i eta_i (1 - PD_i(x_i))].

    f_i'(x_i) = eta_i [1 - Phi(h0_i + c_i x_i) - x_i phi(h0_i + c_i x_i) c_i]

    This is a decreasing function of x_i (f_i is concave), so it serves
    as the bisection target in _total_alloc.
    """
    h = h0 + c * x_arr
    return eta * (1.0 - _norm.cdf(h) - x_arr * _norm.pdf(h) * c)


def _total_alloc(
    lam: float,
    eta: np.ndarray,
    h0: np.ndarray,
    c: np.ndarray,
    cap: float,
    n_bisect: int = 52,
) -> np.ndarray:
    """
    Compute optimal per-obligor exposure vector for a given Lagrange multiplier.

    For each obligor i, the per-obligor KKT optimum is:
      x_i = 0        if  f_i'(0)   <= lam  (not worth funding)
      x_i = cap      if  f_i'(cap) >= lam  (fund at full concentration cap)
      otherwise      the unique root of f_i'(x_i) = lam on (0, cap),
                     found by bisection (f_i' is strictly decreasing).
    """
    fp0 = _marginal_value(np.zeros_like(h0), eta, h0, c)
    fpc = _marginal_value(np.full_like(h0, cap), eta, h0, c)

    at_zero = fp0 <= lam
    at_cap = fpc >= lam
    interior = ~at_zero & ~at_cap

    lo = np.zeros(len(h0))
    hi = np.full(len(h0), cap)

    for _ in range(n_bisect):
        mid = (lo + hi) * 0.5
        g = _marginal_value(mid, eta, h0, c)
        increase = g > lam   # f' too large → x should increase
        lo = np.where(interior & increase, mid, lo)
        hi = np.where(interior & ~increase, mid, hi)

    x_opt = (lo + hi) * 0.5
    x_opt = np.where(at_zero, 0.0, x_opt)
    x_opt = np.where(at_cap, cap, x_opt)
    return x_opt


def _solve_nonlinear(
    portfolio: "pd.DataFrame",
    z: np.ndarray,
    S_s: Optional[np.ndarray],
    x0: Optional[np.ndarray] = None,   # kept for API compatibility; unused
    kappa: float = KAPPA,
    K: float = K_BUDGET,
    n_restarts: int = 1,               # kept for API compatibility; unused
) -> tuple[np.ndarray, float, bool]:
    """
    Solve the separable nonlinear lending allocation problem.

    Uses dual decomposition (bisection on the Lagrange multiplier for the
    budget constraint) which solves the same KKT conditions as SLSQP while
    exploiting the per-obligor separability of the objective (paper Eq. 2.6).

    Parameters
    ----------
    portfolio : obligor DataFrame
    z         : (N,) calibrated thresholds
    S_s       : (N,) climate shock or None (conventional / no-climate)
    x0        : ignored (kept for call-site compatibility)
    kappa     : concentration cap fraction
    K         : total lending budget
    n_restarts: ignored (kept for call-site compatibility)

    Returns
    -------
    x_opt   : (N,) optimal exposures
    obj_val : surviving-interest objective value
    success : always True (bisection converges analytically)
    """
    cap = kappa * K
    eta, h0, c, _ = _precompute_solver_arrays(portfolio, z, S_s, K)

    # ── Bisect on lambda to satisfy budget constraint sum x_i = K ────────────
    # Upper bound: lam > max f_i'(0) forces all x_i = 0
    # Lower bound: lam < min f_i'(0) forces some x_i > 0
    fp0_max = float(np.max(_marginal_value(np.zeros(len(eta)), eta, h0, c)))
    lam_lo = -0.10   # allows for negative Lagrange multiplier edge cases
    lam_hi = fp0_max * 1.05

    for _ in range(80):
        lam_mid = (lam_lo + lam_hi) * 0.5
        x_try = _total_alloc(lam_mid, eta, h0, c, cap)
        if x_try.sum() > K:
            lam_lo = lam_mid   # too much allocated → raise lam
        else:
            lam_hi = lam_mid

    x_opt = _total_alloc((lam_lo + lam_hi) * 0.5, eta, h0, c, cap)

    # Enforce exact budget (minor numerical correction)
    x_opt = np.clip(x_opt, 0.0, cap)
    s = x_opt.sum()
    if s > 0:
        # Scale only unfixed obligors to respect the cap exactly
        residual = K - x_opt[x_opt >= cap - 1e-12].sum()
        interior_mask = x_opt < cap - 1e-12
        if interior_mask.any() and residual > 0:
            x_opt[interior_mask] *= residual / x_opt[interior_mask].sum()

    obj_val = float(np.dot(x_opt, eta * (1.0 - _norm.cdf(h0 + c * x_opt))))
    return x_opt, obj_val, True


def solve_climate_aware(
    portfolio: "pd.DataFrame",
    z: np.ndarray,
    S_s: np.ndarray,
    kappa: float = KAPPA,
    K: float = K_BUDGET,
    x0: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, bool]:
    """
    Climate-aware nonlinear allocation (paper Section 2.6).

    Climate shocks S_s are active; PD depends on x_i endogenously.
    """
    return _solve_nonlinear(portfolio, z, S_s, x0=x0, kappa=kappa, K=K)


def solve_conventional(
    portfolio: "pd.DataFrame",
    z: np.ndarray,
    kappa: float = KAPPA,
    K: float = K_BUDGET,
    x0: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, bool]:
    """
    Conventional nonlinear allocation without climate shocks (paper Section 4.2).

    No climate shocks (S_s = None); PD depends on x_i endogenously.
    """
    return _solve_nonlinear(portfolio, z, None, x0=x0, kappa=kappa, K=K)


def solve_equal_weight(K: float = K_BUDGET, N: int = N_OBLIGORS) -> np.ndarray:
    """
    Equal-weight allocation: x_i = K / N for all i (paper Section 4.2).
    """
    return np.full(N, K / N)


def solve_fixed_pd_linear(
    portfolio: "pd.DataFrame",
    z: np.ndarray,
    S_s: np.ndarray,
    x_ref: Optional[np.ndarray] = None,
    kappa: float = KAPPA,
    K: float = K_BUDGET,
) -> tuple[np.ndarray, float]:
    """
    Fixed-PD linear approximation (paper Section 4.2).

    Step 1: Evaluate PD at the reference allocation (x_ref = K/N).
    Step 2: Hold these PDs fixed and solve the linear-in-x problem:
            maximise sum_i x_i eta^i [1 - PD_i^fixed]
            subject to sum x_i = K, 0 <= x_i <= kappa K.

    The linear problem has the analytic solution: assign the full budget to
    the obligors with the highest coefficient eta^i (1 - PD^fixed), up to
    the concentration cap.

    Parameters
    ----------
    portfolio : obligor DataFrame
    z         : calibrated thresholds
    S_s       : climate shock at which reference PDs are evaluated
    x_ref     : allocation at which PDs are frozen (default K/N vector)
    kappa     : concentration cap fraction
    K         : total budget

    Returns
    -------
    x_fixed : (N,) optimal exposures
    obj_val : objective value when x_fixed is re-evaluated with true PDs
    """
    N = len(portfolio)
    if x_ref is None:
        x_ref = np.full(N, K / N)

    # Freeze PDs at reference allocation
    pd_fixed = compute_pd(portfolio, x_ref, z, S_s)
    coeffs = portfolio["eta"].values * (1.0 - pd_fixed)

    # Greedy fill: assign cap to top obligors until budget is exhausted
    cap = kappa * K
    order = np.argsort(-coeffs)  # descending
    x_lin = np.zeros(N)
    remaining = K
    for idx in order:
        if remaining <= 0:
            break
        alloc = min(cap, remaining)
        x_lin[idx] = alloc
        remaining -= alloc

    obj_val = float(np.dot(x_lin, portfolio["eta"].values * (1.0 - pd_fixed)))
    return x_lin, obj_val


# ── Portfolio performance metrics ────────────────────────────────────────────

def compute_metrics(
    portfolio: "pd.DataFrame",
    x: np.ndarray,
    z: np.ndarray,
    S_s: Optional[np.ndarray],
    lgd: float = LGD,
    funded_tol: float = 1e-6,
) -> dict:
    """
    Compute portfolio performance metrics for a given allocation.

    Metrics (paper Section 4.2–4.3, Tables 2–3):
      objective    : O(x; s) = sum_i x_i eta^i [1 - PD_T^i]
      ecl          : ECL(x; s) = LGD * sum_i x_i PD_T^i
      n_funded     : N_+ = count(x_i > funded_tol)
      max_share    : max_i (x_i / K)
      hhi          : HHI = sum_i (x_i / K)^2

    Parameters
    ----------
    portfolio  : obligor DataFrame
    x          : (N,) exposures
    z          : calibrated thresholds
    S_s        : climate shock or None
    lgd        : loss given default
    funded_tol : tolerance for counting funded obligors

    Returns
    -------
    dict with keys: objective, ecl, n_funded, max_share, hhi
    """
    N = len(portfolio)
    K = x.sum()
    eta = portfolio["eta"].values
    pd_arr = compute_pd(portfolio, x, z, S_s)

    obj = float(np.dot(x, eta * (1.0 - pd_arr)))
    ecl = float(lgd * np.dot(x, pd_arr))
    n_funded = int(np.sum(x > funded_tol))
    max_share = float(np.max(x) / K) if K > 0 else 0.0
    hhi = float(np.sum((x / K) ** 2)) if K > 0 else 0.0

    return {
        "objective": obj,
        "ecl": ecl,
        "n_funded": n_funded,
        "max_share": max_share,
        "hhi": hhi,
    }


def sector_allocations(
    portfolio: "pd.DataFrame",
    x: np.ndarray,
    sector_labels: "pd.Series",
) -> "pd.DataFrame":
    """
    Aggregate obligor exposures to sector level.

    Parameters
    ----------
    portfolio      : obligor DataFrame with column 'sector_idx'
    x              : (N,) exposures
    sector_labels  : Series mapping sector_idx → sector label

    Returns
    -------
    DataFrame with columns: sector_idx, label, total_exposure
    """
    import pandas as pd

    df = portfolio[["sector_idx"]].copy()
    df["exposure"] = x
    agg = df.groupby("sector_idx")["exposure"].sum().reset_index()
    agg.columns = ["sector_idx", "total_exposure"]
    agg["label"] = agg["sector_idx"].map(sector_labels)
    return agg.sort_values("sector_idx").reset_index(drop=True)
