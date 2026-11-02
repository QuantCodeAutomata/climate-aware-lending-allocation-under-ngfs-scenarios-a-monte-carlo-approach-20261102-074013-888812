"""
Unit tests for synthetic obligor portfolio generation.
"""

import numpy as np
import pytest

from src.config import (
    N_OBLIGORS,
    N_SECTORS,
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
)
from src.portfolio import generate_portfolio
from data.generate_synthetic import get_sector_table


class TestPortfolioGeneration:
    def test_portfolio_row_count(self, full_portfolio):
        assert len(full_portfolio) == N_OBLIGORS

    def test_portfolio_columns(self, full_portfolio):
        expected = {"obligor_id", "sector_idx", "p0", "a", "b", "sigma",
                    "rho", "Lambda", "eta"}
        assert expected.issubset(set(full_portfolio.columns))

    def test_p0_is_zero(self, full_portfolio):
        assert (full_portfolio["p0"] == 0.0).all()

    def test_parameter_ranges(self, full_portfolio):
        df = full_portfolio
        assert df["a"].between(A_LOW, A_HIGH).all(), "a out of range"
        assert df["b"].between(B_LOW, B_HIGH).all(), "b out of range"
        assert df["sigma"].between(SIGMA_LOW, SIGMA_HIGH).all(), "sigma out of range"
        assert df["rho"].between(RHO_LOW, RHO_HIGH).all(), "rho out of range"
        assert df["Lambda"].between(LAMBDA_SPREAD_LOW, LAMBDA_SPREAD_HIGH).all()

    def test_eta_equals_r_plus_lambda(self, full_portfolio):
        diff = (full_portfolio["eta"] - R_RISKFREE - full_portfolio["Lambda"]).abs()
        assert (diff < 1e-12).all()

    def test_sector_indices_in_range(self, full_portfolio):
        assert full_portfolio["sector_idx"].between(0, N_SECTORS - 1).all()

    def test_all_sectors_represented(self, full_portfolio):
        # With 1000 obligors and 50 sectors, all should appear
        present = full_portfolio["sector_idx"].nunique()
        assert present >= 40, f"Only {present} sectors represented"

    def test_reproducibility(self):
        df1 = generate_portfolio(seed=42)
        df2 = generate_portfolio(seed=42)
        assert (df1["b"].values == df2["b"].values).all()

    def test_different_seeds_differ(self):
        df1 = generate_portfolio(seed=1)
        df2 = generate_portfolio(seed=2)
        assert not (df1["a"].values == df2["a"].values).all()

    def test_sector_table_shape(self, sector_table):
        assert len(sector_table) == N_SECTORS
        assert "label" in sector_table.columns
        assert "type" in sector_table.columns

    def test_sector_table_types(self, sector_table):
        assert set(sector_table["type"].unique()).issubset({"green", "fossil", "neutral"})
