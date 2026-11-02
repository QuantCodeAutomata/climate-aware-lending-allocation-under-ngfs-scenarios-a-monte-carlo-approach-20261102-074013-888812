"""
Pytest configuration and shared fixtures for the climate-aware lending
allocation replication tests.

All fixtures use synthetic data generated deterministically from fixed seeds
so that tests are fully self-contained and require no external API access.
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import generate_portfolio
from src.pd_engine import calibrate_thresholds, compute_all_scenario_shocks
from src.simulation import build_systemic_covariance
from data.generate_synthetic import get_sector_table, get_ngfs_shocks
from src.config import (
    A_LOW, A_HIGH, B_LOW, B_HIGH, SIGMA_LOW, SIGMA_HIGH,
    RHO_LOW, RHO_HIGH, LAMBDA_SPREAD_LOW, LAMBDA_SPREAD_HIGH,
    R_RISKFREE, P0_INITIAL, K_BUDGET, N_SECTORS,
)


# Register custom markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "market_data: marks tests requiring real market data API keys "
        "(skip when keys absent)",
    )


# ---------------------------------------------------------------------------
# Shared data fixtures (no API calls)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def full_portfolio():
    """Full N=1,000 obligor portfolio; generated once per session."""
    return generate_portfolio(seed=42)


@pytest.fixture(scope="session")
def sector_table():
    return get_sector_table()


@pytest.fixture(scope="session")
def ngfs_shocks():
    return get_ngfs_shocks()


@pytest.fixture(scope="session")
def thresholds(full_portfolio):
    return calibrate_thresholds(full_portfolio)


@pytest.fixture(scope="session")
def climate_shocks(full_portfolio, ngfs_shocks):
    return compute_all_scenario_shocks(full_portfolio, ngfs_shocks)


@pytest.fixture(scope="session")
def systemic_cov(full_portfolio):
    return build_systemic_covariance(full_portfolio)


# ---------------------------------------------------------------------------
# Small portfolio fixture for optimisation tests (N=50, fast SLSQP)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def opt_portfolio():
    """
    N=50 obligor portfolio for fast optimisation unit tests.

    Built directly without modifying global config so it is safe as a
    session-scoped fixture alongside other fixtures.
    """
    n = 50
    rng = np.random.default_rng(99)
    a = rng.uniform(A_LOW, A_HIGH, n)
    b = rng.uniform(B_LOW, B_HIGH, n)
    sigma = rng.uniform(SIGMA_LOW, SIGMA_HIGH, n)
    rho = rng.uniform(RHO_LOW, RHO_HIGH, n)
    spread = rng.uniform(LAMBDA_SPREAD_LOW, LAMBDA_SPREAD_HIGH, n)
    eta = R_RISKFREE + spread
    sector_idx = rng.integers(0, N_SECTORS, n)
    return pd.DataFrame({
        "obligor_id": np.arange(n),
        "sector_idx": sector_idx,
        "p0": P0_INITIAL,
        "a": a,
        "b": b,
        "sigma": sigma,
        "rho": rho,
        "Lambda": spread,
        "eta": eta,
    })


@pytest.fixture(scope="session")
def opt_thresholds(opt_portfolio):
    """Thresholds calibrated to x_ref = K/N for the small opt_portfolio."""
    n = len(opt_portfolio)
    return calibrate_thresholds(opt_portfolio, x_ref=K_BUDGET / n)


@pytest.fixture(scope="session")
def opt_shocks(opt_portfolio, ngfs_shocks):
    """Climate shocks for the small opt_portfolio."""
    return compute_all_scenario_shocks(opt_portfolio, ngfs_shocks)
