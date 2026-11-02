"""
Synthetic obligor portfolio generation.

Generates a fixed population of N = 1,000 synthetic obligors with sector
assignments and parameter draws as specified in paper Section 3.

Obligor parameters (all drawn independently and fixed across scenarios):
  p0^i  = 0                            (initial log-production; paper Section 3)
  a^i   ~ Uniform(0,   0.5)            (productivity baseline)
  b^i   ~ Uniform(1,   4)              (mean-reversion speed)
  sigma^i ~ Uniform(0.1, 0.5)          (production volatility)
  rho^i ~ Uniform(-1,  1)              (systemic loading)
  Lambda_i ~ Uniform(0.025, 0.06)      (lending spread)
  eta^i = r + Lambda_i                 (lending rate; r = 0.02)

Sector assignments: obligors are assigned to sectors with probabilities
proportional to sqrt(baseline_prod_j), preserving fixed assignments
across all scenario evaluations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    N_OBLIGORS,
    N_SECTORS,
    P0_INITIAL,
    A_LOW,
    A_HIGH,
    B_LOW,
    B_HIGH,
    SIGMA_LOW,
    SIGMA_HIGH,
    RHO_LOW,
    RHO_HIGH,
    LAMBDA_SPREAD_LOW,
    LAMBDA_SPREAD_HIGH,
    R_RISKFREE,
    PORTFOLIO_SEED,
)
from data.generate_synthetic import get_sector_table


def generate_portfolio(seed: int = PORTFOLIO_SEED) -> pd.DataFrame:
    """
    Draw the fixed synthetic obligor population.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility (paper does not specify the seed).

    Returns
    -------
    DataFrame with one row per obligor and columns:
      obligor_id : int    – 0-based index
      sector_idx : int    – sector assignment (0..49)
      p0         : float  – initial log-production (always 0)
      a          : float  – baseline productivity
      b          : float  – mean-reversion speed (year^-1)
      sigma      : float  – production volatility
      rho        : float  – systemic correlation loading
      Lambda     : float  – lending spread
      eta        : float  – lending rate = r + Lambda
    """
    rng = np.random.default_rng(seed)

    # Sector assignment weights: proportional to sqrt(baseline_prod) (paper Section 3)
    sectors = get_sector_table()
    weights = np.sqrt(sectors["baseline_prod"].values)
    weights = weights / weights.sum()

    sector_indices = rng.choice(N_SECTORS, size=N_OBLIGORS, replace=True, p=weights)

    # Draw obligor parameters
    a_vals = rng.uniform(A_LOW, A_HIGH, N_OBLIGORS)
    b_vals = rng.uniform(B_LOW, B_HIGH, N_OBLIGORS)
    sigma_vals = rng.uniform(SIGMA_LOW, SIGMA_HIGH, N_OBLIGORS)
    rho_vals = rng.uniform(RHO_LOW, RHO_HIGH, N_OBLIGORS)
    spread_vals = rng.uniform(LAMBDA_SPREAD_LOW, LAMBDA_SPREAD_HIGH, N_OBLIGORS)
    eta_vals = R_RISKFREE + spread_vals

    df = pd.DataFrame(
        {
            "obligor_id": np.arange(N_OBLIGORS),
            "sector_idx": sector_indices,
            "p0": P0_INITIAL,
            "a": a_vals,
            "b": b_vals,
            "sigma": sigma_vals,
            "rho": rho_vals,
            "Lambda": spread_vals,
            "eta": eta_vals,
        }
    )
    return df
