"""
Synthetic data generation for the climate-aware lending allocation experiments.

Produces:
  1. Sector metadata table (50 sectors) with baseline GEME3 production
     and sector-type labels (green / fossil / neutral).
  2. NGFS 2025-2029 sector-level shock tensors shaped
     [scenario, sector, year, channel] for the four NGFS 2024 short-term
     scenarios: DAPS, HWTP, SWUC, DiRe.

The exact NGFS sector tables (GEME3 production, CLIMACRED financial
adjustments, physical-impact estimates) are not published in the paper.
The synthetic shocks below are constructed to satisfy the qualitative
properties stated in paper Section 4.1:
  - DAPS: small, near-uniform shocks → slight portfolio PD increase,
    low cross-sectional dispersion.
  - HWTP / SWUC: large heterogeneous shocks → green sectors gain,
    fossil sectors lose; net portfolio PD slightly below reference,
    high cross-sectional dispersion.
  - DiRe: transition shocks plus physical impacts on all sectors
    → heterogeneous effects with overall slight PD reduction.

Channel index convention (paper Section 3):
  j = 0  production shock (GEME3 deviation from baseline)
  j = 1  financial shock  (negative CLIMACRED corporate-spread adjustment)
  j = 2  physical shock   (negative sum of physical losses;
                            included only for DAPS and DiRe per the paper)

Paper: "Climate aware lending allocation under NGFS scenarios –
        A Monte Carlo approach", Section 3, Implementation Details.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    N_SECTORS,
    SCENARIOS,
    SCENARIO_YEARS,
    PROD_RATIO_LOW,
    PROD_RATIO_HIGH,
)

# ── Sector catalogue ────────────────────────────────────────────────────────

# 50 harmonised sectors; names and type classifications are representative of
# GEME3/CLIMACRED sector groupings described in the paper (Section 3).
# Exact sector names not enumerated in the paper; labelling here is
# illustrative and consistent with the paper's selected-sector discussion
# (Section 4.1, Table 1).

_SECTOR_RECORDS = [
    # idx  label                          type       baseline_prod
    (0,   "EV transport equipment",       "green",   4.5),
    (1,   "Wind power",                   "green",   3.8),
    (2,   "Solar power",                  "green",   3.2),
    (3,   "Power supply",                 "green",   5.0),
    (4,   "Electric machinery",           "green",   2.9),
    (5,   "Battery storage",              "green",   2.0),
    (6,   "Rail transport",               "green",   2.4),
    (7,   "Heating and cooling",          "neutral", 3.5),
    (8,   "Agriculture",                  "neutral", 6.0),
    (9,   "Food processing",              "neutral", 5.5),
    (10,  "Chemicals",                    "neutral", 4.8),
    (11,  "Textiles",                     "neutral", 2.2),
    (12,  "Wood and paper",               "neutral", 2.6),
    (13,  "Construction materials",       "neutral", 3.9),
    (14,  "Metal products",               "neutral", 4.1),
    (15,  "Machinery and equipment",      "neutral", 4.3),
    (16,  "Automotive (ICE)",             "fossil",  3.7),
    (17,  "Aviation",                     "fossil",  2.8),
    (18,  "Shipping",                     "fossil",  2.3),
    (19,  "Road freight",                 "fossil",  3.1),
    (20,  "Coal mining",                  "fossil",  1.8),
    (21,  "Gas",                          "fossil",  5.2),
    (22,  "Crude oil",                    "fossil",  6.5),
    (23,  "Oil",                          "fossil",  5.8),
    (24,  "Natural gas distribution",     "fossil",  3.6),
    (25,  "Petroleum refining",           "fossil",  4.0),
    (26,  "Plastics",                     "fossil",  3.0),
    (27,  "Cement",                       "fossil",  3.3),
    (28,  "Steel",                        "fossil",  4.2),
    (29,  "Aluminium",                    "fossil",  2.7),
    (30,  "Glass",                        "neutral", 1.9),
    (31,  "Rubber",                       "neutral", 2.1),
    (32,  "Electronics",                  "neutral", 3.4),
    (33,  "Telecommunications",           "neutral", 4.6),
    (34,  "Financial services",           "neutral", 5.1),
    (35,  "Real estate",                  "neutral", 4.7),
    (36,  "Retail trade",                 "neutral", 3.8),
    (37,  "Wholesale trade",              "neutral", 3.6),
    (38,  "Healthcare",                   "neutral", 3.0),
    (39,  "Education",                    "neutral", 2.5),
    (40,  "Public administration",        "neutral", 2.8),
    (41,  "Water supply",                 "neutral", 2.0),
    (42,  "Waste management",             "neutral", 1.8),
    (43,  "Forestry",                     "neutral", 2.2),
    (44,  "Fishing",                      "neutral", 1.6),
    (45,  "Pharmaceuticals",              "neutral", 3.5),
    (46,  "Aerospace",                    "neutral", 2.9),
    (47,  "Defence",                      "neutral", 2.4),
    (48,  "Logistics",                    "neutral", 3.2),
    (49,  "Other services",               "neutral", 2.6),
]

assert len(_SECTOR_RECORDS) == N_SECTORS


def get_sector_table() -> pd.DataFrame:
    """
    Return a DataFrame with sector metadata.

    Columns
    -------
    sector_idx   : int   – zero-based sector index (0..49)
    label        : str   – human-readable sector name
    type         : str   – 'green' | 'fossil' | 'neutral'
    baseline_prod: float – synthetic baseline GEME3 production index
    """
    df = pd.DataFrame(
        _SECTOR_RECORDS,
        columns=["sector_idx", "label", "type", "baseline_prod"],
    )
    return df


# ── NGFS sector-level shocks ────────────────────────────────────────────────

def _clamp_prod_ratio(ratio: np.ndarray) -> np.ndarray:
    """Clip production ratio to [PROD_RATIO_LOW, PROD_RATIO_HIGH] per paper."""
    return np.clip(ratio, PROD_RATIO_LOW, PROD_RATIO_HIGH)


def build_ngfs_shocks(rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Build the sector-level annual NGFS shock tensor.

    Shape: (n_scenarios, N_SECTORS, n_years, 3)
    Axis semantics:
      axis 0 – scenario index (order matches SCENARIOS list)
      axis 1 – sector index 0..49
      axis 2 – year index 0..4 for 2025..2029
      axis 3 – channel index: 0=production, 1=financial, 2=physical

    The shocks theta_{j,s,k}^sector are:
      j=0  production: ln(scenario_production / baseline_production),
           clipped so that exp(theta) lies in [0.25, 4] per paper Section 3.
      j=1  financial: negative CLIMACRED corporate-bond spread adjustment.
      j=2  physical:  negative physical-impact sum; included for DAPS (idx 0)
                      and DiRe (idx 3); zero for HWTP and SWUC (paper Section 3).

    Synthetic scenario-shock calibration
    -------------------------------------
    The paper does not publish the raw NGFS tables.  The values below are
    constructed to reproduce the qualitative properties of Table 1 and the
    discussion in Sections 4.1–4.3:
      - DAPS  (idx 0): physical-risk dominant, small transition;
                       negative physical losses dominate → portfolio mean
                       PD slightly above reference.
      - HWTP  (idx 1): strong ordered transition; green sectors gain large
                       production and spread benefits, fossil sectors lose;
                       portfolio mean PD slightly below reference with wide
                       cross-sectional spread.
      - SWUC  (idx 2): moderate transition, no physical; similar pattern
                       to HWTP at smaller magnitude.
      - DiRe  (idx 3): disorderly transition + physical risk; transition
                       channel favours green sectors, physical channel hurts
                       all; net portfolio PD slightly below reference.
    """
    n_scenarios = len(SCENARIOS)
    n_years = len(SCENARIO_YEARS)
    n_channels = 3

    shocks = np.zeros((n_scenarios, N_SECTORS, n_years, n_channels))

    sectors = get_sector_table()
    green_mask = (sectors["type"] == "green").values
    fossil_mask = (sectors["type"] == "fossil").values
    neutral_mask = (sectors["type"] == "neutral").values

    # Ramp over years: shocks start small and build (2025→2029)
    year_ramp = np.array([0.5, 0.7, 0.9, 1.0, 1.1])  # multiplicative ramp

    # Sign convention (paper Section 3):
    #   theta_production > 0  means scenario production exceeds baseline → beneficial
    #   theta_financial  > 0  means theta = −(spread adj); negative spread adj
    #                          (lower spread for green) → theta > 0 → beneficial
    #   theta_physical   < 0  always adverse (losses)

    # Fossil sectors are 14/50 and carry higher baseline production, so they
    # receive ~30 % of obligors.  Green sectors receive ~14 %.  To achieve a
    # slight net portfolio PD reduction under transition scenarios the green
    # production shock must be large enough to outweigh the fossil decline on
    # an obligor-weighted basis.  The calibration below targets weighted-average
    # production theta ≈ +0.03 for HWTP/SWUC/DiRe.

    # -- DAPS (idx 0): Delayed Additional Policy Scenario --
    # Physical risk present; very small or absent transition shocks.
    # Net effect: small uniform production drag + physical losses → mean PD
    # slightly above reference with low cross-sectional dispersion.
    s = 0
    prod_daps = np.zeros(N_SECTORS)
    prod_daps[green_mask] = -0.008
    prod_daps[fossil_mask] = -0.012
    prod_daps[neutral_mask] = -0.010
    fin_daps = np.zeros(N_SECTORS)  # negligible financial adjustment under DAPS
    fin_daps[:] = -0.001            # slight uniform negative
    phys_daps = np.zeros(N_SECTORS)
    phys_daps[green_mask] = -0.006   # physical losses active for DAPS
    phys_daps[fossil_mask] = -0.014
    phys_daps[neutral_mask] = -0.009
    for k in range(n_years):
        shocks[s, :, k, 0] = prod_daps * year_ramp[k]
        shocks[s, :, k, 1] = fin_daps * year_ramp[k]
        shocks[s, :, k, 2] = phys_daps * year_ramp[k]

    # -- HWTP (idx 1): ordered fast transition --
    # Large green production boom, steep fossil decline; financial spreads fall
    # for green (theta_fin > 0) and rise for fossil (theta_fin < 0).
    # No physical channel (paper Section 3).
    # Weighted-average production target ≈ +0.03 (see comment above).
    s = 1
    prod_hwtp = np.zeros(N_SECTORS)
    prod_hwtp[green_mask] = 0.40    # large green production boom
    prod_hwtp[fossil_mask] = -0.10  # fossil decline; smaller magnitude avoids
    prod_hwtp[neutral_mask] = 0.05  # neutrals benefit modestly
    # theta_financial sign: negative spread adjustment → positive theta
    fin_hwtp = np.zeros(N_SECTORS)
    fin_hwtp[green_mask] = +0.030   # lower spreads for green (beneficial)
    fin_hwtp[fossil_mask] = -0.025  # higher spreads for fossil (adverse)
    fin_hwtp[neutral_mask] = +0.003
    for k in range(n_years):
        shocks[s, :, k, 0] = prod_hwtp * year_ramp[k]
        shocks[s, :, k, 1] = fin_hwtp * year_ramp[k]
        shocks[s, :, k, 2] = 0.0  # no physical channel for HWTP

    # -- SWUC (idx 2): moderate transition (paper does not specify the name fully)
    # Similar to HWTP at 60% amplitude; no physical channel.
    s = 2
    prod_swuc = prod_hwtp * 0.60
    fin_swuc = fin_hwtp * 0.60
    for k in range(n_years):
        shocks[s, :, k, 0] = prod_swuc * year_ramp[k]
        shocks[s, :, k, 1] = fin_swuc * year_ramp[k]
        shocks[s, :, k, 2] = 0.0  # no physical channel for SWUC

    # -- DiRe (idx 3): Disorderly transition + physical risk --
    # Transition heterogeneity at 75% of HWTP plus physical impacts on all
    # sectors (paper Section 3: physical channel active for DiRe).
    s = 3
    prod_dire = prod_hwtp * 0.75
    fin_dire = fin_hwtp * 0.70
    phys_dire = np.zeros(N_SECTORS)
    phys_dire[green_mask] = -0.010
    phys_dire[fossil_mask] = -0.016
    phys_dire[neutral_mask] = -0.012
    for k in range(n_years):
        shocks[s, :, k, 0] = prod_dire * year_ramp[k]
        shocks[s, :, k, 1] = fin_dire * year_ramp[k]
        shocks[s, :, k, 2] = phys_dire * year_ramp[k]

    # Apply production-ratio bounding: convert log-deviation to ratio, clip, reconvert
    # theta_prod is a log-deviation: ratio = exp(theta_prod)
    prod_shocks = shocks[:, :, :, 0]
    ratios = np.exp(prod_shocks)
    ratios = _clamp_prod_ratio(ratios)
    shocks[:, :, :, 0] = np.log(ratios)  # back to log-space; bounded

    return shocks


def get_ngfs_shocks() -> np.ndarray:
    """
    Return the prebuilt NGFS shock tensor.

    Shape: (n_scenarios=4, N_SECTORS=50, n_years=5, n_channels=3)
    """
    return build_ngfs_shocks()
