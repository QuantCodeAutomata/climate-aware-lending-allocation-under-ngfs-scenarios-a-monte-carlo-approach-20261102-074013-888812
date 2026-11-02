"""
Experiment 1: Scenario-dependent analytical terminal default probabilities
under NGFS short-term scenarios.

Reproduces Table 1 and the sector analysis of Section 4.1 of the paper.

Steps:
  1. Generate synthetic obligor portfolio (N = 1,000, 50 sectors).
  2. Compute NGFS-scenario climate shocks for each obligor.
  3. Calibrate fixed default thresholds z_i from the no-climate reference.
  4. Compute analytical five-year terminal PDs under each NGFS scenario at
     the equal-weight reference allocation.
  5. Summarise cross-sectional PD distributions and sector-level PD changes.
  6. Save results to results/.

Paper: "Climate aware lending allocation under NGFS scenarios –
        A Monte Carlo approach", Sections 3–4.1, Table 1.

Usage:
    python -m exp.exp1
"""

import json
import numpy as np
import pandas as pd

from src.config import (
    K_BUDGET,
    N_OBLIGORS,
    T_MATURITY,
    LAMBDA_REF,
    SCENARIOS,
    ILLUSTRATIVE_SECTOR_LABELS,
)
from src.portfolio import generate_portfolio
from src.pd_engine import (
    calibrate_thresholds,
    compute_pd,
    compute_all_scenario_shocks,
)
from src.reporting import (
    pd_cross_section_summary,
    sector_pd_change_summary,
    plot_pd_distributions,
    plot_sector_pd_changes,
    save_table,
    format_pct,
)
from data.generate_synthetic import get_sector_table, get_ngfs_shocks


def run() -> dict:
    """
    Execute experiment 1 and return key metrics.

    Returns
    -------
    dict with:
      no_climate_mean_pd: float  – reference mean PD (should be ~13.929%)
      scenario_mean_pds : dict   – mean PD per scenario
      cross_section_df  : DataFrame – distribution statistics
      sector_changes_df : DataFrame – sector-level PD changes
    """
    print("=== Experiment 1: Analytical terminal PDs under NGFS scenarios ===\n")

    # ── 1. Portfolio and scenario data ────────────────────────────────────────
    portfolio = generate_portfolio()
    sector_table = get_sector_table()
    ngfs_shocks = get_ngfs_shocks()

    print(f"Portfolio: {len(portfolio)} obligors, "
          f"{portfolio['sector_idx'].nunique()} sectors")

    # ── 2. Reference calibration ──────────────────────────────────────────────
    x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
    z = calibrate_thresholds(portfolio, x_ref=K_BUDGET / N_OBLIGORS)

    # No-climate PD at reference allocation (should equal p_ref for each obligor)
    no_climate_pd = compute_pd(portfolio, x_ref, z, S_s=None)
    p_ref = 1.0 - np.exp(-LAMBDA_REF * T_MATURITY)
    no_climate_mean = float(no_climate_pd.mean())
    print(f"No-climate reference PD: p_ref = {p_ref * 100:.4f}%  "
          f"(computed mean = {no_climate_mean * 100:.4f}%)")
    assert abs(no_climate_mean - p_ref) < 1e-8, \
        f"Calibration error: mean PD {no_climate_mean:.6f} != p_ref {p_ref:.6f}"

    # ── 3. Scenario climate shocks ────────────────────────────────────────────
    S_all = compute_all_scenario_shocks(portfolio, ngfs_shocks)

    # ── 4. Analytical scenario PDs at reference allocation ────────────────────
    n_scenarios = len(SCENARIOS)
    pd_matrix = np.empty((n_scenarios, N_OBLIGORS))
    for s, name in enumerate(SCENARIOS):
        pd_matrix[s] = compute_pd(portfolio, x_ref, z, S_s=S_all[s])
        print(f"  {name:6s} mean PD = {pd_matrix[s].mean() * 100:.4f}%  "
              f"std = {pd_matrix[s].std() * 100:.4f}%")

    # ── 5. Cross-sectional distribution summary ───────────────────────────────
    cross_section_df = pd_cross_section_summary(
        pd_matrix, SCENARIOS, no_climate_pd=no_climate_pd
    )
    print("\nCross-sectional PD distribution (Table 1 analogue):")
    print(cross_section_df.to_string(index=False, float_format=lambda x: f"{x*100:.4f}%"))

    # ── 6. Sector-level PD changes ────────────────────────────────────────────
    sector_all_df = sector_pd_change_summary(
        portfolio, pd_matrix, no_climate_pd, sector_table,
        SCENARIOS, illustrative_labels=None
    )
    illustrative_df = sector_pd_change_summary(
        portfolio, pd_matrix, no_climate_pd, sector_table,
        SCENARIOS, illustrative_labels=ILLUSTRATIVE_SECTOR_LABELS
    )
    print("\nIllustrative sector PD changes (Section 4.1):")
    change_cols = [f"{s}_pd_change" for s in SCENARIOS]
    disp = illustrative_df[["label", "type"] + change_cols].copy()
    for col in change_cols:
        disp[col] = disp[col].map(lambda v: f"{v * 100:.4f}pp")
    print(disp.to_string(index=False))

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    save_table(cross_section_df, "exp1_pd_cross_section.csv")
    save_table(sector_all_df, "exp1_sector_pd_changes.csv")
    save_table(illustrative_df, "exp1_sector_pd_changes_illustrative.csv")

    plot_pd_distributions(pd_matrix, no_climate_pd, SCENARIOS,
                          filename="exp1_pd_distributions.png")
    plot_sector_pd_changes(illustrative_df, SCENARIOS,
                           filename="exp1_sector_pd_changes.png")

    # Scenario mean PDs summary
    scenario_means = {name: float(pd_matrix[s].mean()) for s, name in enumerate(SCENARIOS)}
    summary = {
        "no_climate_mean_pd": no_climate_mean,
        "p_ref": float(p_ref),
        **{f"{k}_mean_pd": v for k, v in scenario_means.items()},
    }
    with open("results/exp1_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nResults saved to results/")
    return {
        "no_climate_mean_pd": no_climate_mean,
        "scenario_mean_pds": scenario_means,
        "cross_section_df": cross_section_df,
        "sector_changes_df": illustrative_df,
        "pd_matrix": pd_matrix,
        "portfolio": portfolio,
        "z": z,
        "S_all": S_all,
        "no_climate_pd": no_climate_pd,
    }


if __name__ == "__main__":
    run()
