"""
Experiment 2: Scenario-specific climate-aware nonlinear lending optimisation
versus conventional, equal-weight, and fixed-PD benchmark policies.

Reproduces Tables 2–3 and the sector reallocation analysis of Section 4.2–4.3.

Steps:
  1. Reuse portfolio, thresholds, and climate shocks from Experiment 1.
  2. For each NGFS scenario, solve four allocation problems:
       (a) Climate-aware nonlinear (SLSQP with climate shocks).
       (b) Conventional nonlinear (SLSQP, no climate shocks).
       (c) Equal-weight (K/N for all i).
       (d) Fixed-PD linear (frozen PDs, greedy fill up to cap).
  3. Re-evaluate all allocations with the true exposure-dependent PD function.
  4. Compute objective O, ECL, N_funded, max_share, HHI, and incremental
     metrics relative to conventional.
  5. Save tables and plots.

Paper: "Climate aware lending allocation under NGFS scenarios –
        A Monte Carlo approach", Sections 2.6, 4.2–4.3, Tables 2–3.

Usage:
    python -m exp.exp2
"""

import json
import numpy as np
import pandas as pd

from src.config import (
    K_BUDGET,
    N_OBLIGORS,
    KAPPA,
    LGD,
    SCENARIOS,
)
from src.portfolio import generate_portfolio
from src.pd_engine import compute_all_scenario_shocks
from src.optimization import (
    solve_climate_aware,
    solve_conventional,
    solve_equal_weight,
    solve_fixed_pd_linear,
    compute_metrics,
    sector_allocations,
)
from src.pd_engine import calibrate_thresholds
from src.reporting import (
    save_table,
    plot_loss_distributions,
    RESULTS_DIR,
)
from data.generate_synthetic import get_sector_table, get_ngfs_shocks


POLICIES = ["climate_aware", "conventional", "equal_weight", "fixed_pd"]


def run(
    portfolio: pd.DataFrame | None = None,
    z: np.ndarray | None = None,
    S_all: np.ndarray | None = None,
) -> dict:
    """
    Execute experiment 2 and return allocations and metrics.

    Parameters
    ----------
    portfolio : if provided, reuse from exp1 (for consistency)
    z         : calibrated thresholds (if None, recomputed)
    S_all     : (n_scenarios, N) climate shocks (if None, recomputed)

    Returns
    -------
    dict with allocations and metric DataFrames per scenario
    """
    print("=== Experiment 2: Climate-aware lending optimisation ===\n")

    # ── 1. Shared data ────────────────────────────────────────────────────────
    if portfolio is None:
        portfolio = generate_portfolio()
    sector_table = get_sector_table()
    ngfs_shocks = get_ngfs_shocks()

    if z is None:
        z = calibrate_thresholds(portfolio)
    if S_all is None:
        S_all = compute_all_scenario_shocks(portfolio, ngfs_shocks)

    sector_labels = sector_table.set_index("sector_idx")["label"]

    all_metrics: list[dict] = []
    all_allocations: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in SCENARIOS
    }
    sector_realloc_records: list[dict] = []

    # Conventional solution used as warm start for climate-aware; computed once
    # per scenario (no climate shocks in conventional → same for all scenarios)
    print("Solving conventional allocation (warm start)...")
    x_conv, _, _ = solve_conventional(portfolio, z)
    print(f"  Conventional: N_funded={int(np.sum(x_conv > 1e-6))}, "
          f"obj={compute_metrics(portfolio, x_conv, z, None)['objective']:.4f}")

    for s_idx, scenario in enumerate(SCENARIOS):
        print(f"\n── Scenario: {scenario} ──────────────────────────")
        S_s = S_all[s_idx]

        # (a) Climate-aware nonlinear
        x_ca, _, _ = solve_climate_aware(portfolio, z, S_s, x0=x_conv.copy())

        # (b) Conventional nonlinear (same for all scenarios; cache)
        if s_idx == 0:
            x_cv = x_conv.copy()
        else:
            x_cv = x_conv.copy()

        # (c) Equal-weight
        x_ew = solve_equal_weight()

        # (d) Fixed-PD linear
        x_fp, _ = solve_fixed_pd_linear(portfolio, z, S_s)

        allocations = {
            "climate_aware": x_ca,
            "conventional": x_cv,
            "equal_weight": x_ew,
            "fixed_pd": x_fp,
        }
        all_allocations[scenario] = allocations

        # Re-evaluate all policies with true PD function
        for policy, x_pol in allocations.items():
            m = compute_metrics(portfolio, x_pol, z, S_s)
            m["scenario"] = scenario
            m["policy"] = policy
            all_metrics.append(m)
            print(f"  {policy:15s}: obj={m['objective']:.4f}  "
                  f"ecl={m['ecl']:.4f}  "
                  f"N+={m['n_funded']}  "
                  f"max_share={m['max_share']:.4f}  "
                  f"hhi={m['hhi']:.6f}")

        # Sector reallocations: climate-aware minus conventional
        sec_ca = sector_allocations(portfolio, x_ca, sector_labels)
        sec_cv = sector_allocations(portfolio, x_cv, sector_labels)
        sec_diff = sec_ca.copy()
        sec_diff["reallocation"] = (
            sec_ca["total_exposure"] - sec_cv["total_exposure"]
        )
        sec_diff["scenario"] = scenario
        sector_realloc_records.append(sec_diff)

    # ── Build summary DataFrames ───────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)

    # Table 2 analogue: pivot by scenario × policy
    table2_cols = ["objective", "ecl", "n_funded", "max_share", "hhi"]
    table2 = metrics_df.pivot_table(
        values=table2_cols, index=["scenario"], columns=["policy"], aggfunc="first"
    )
    table2.columns = [f"{col}_{pol}" for col, pol in table2.columns]
    table2 = table2.reset_index()

    # Table 3 analogue: ECL reduction of climate-aware vs conventional
    table3_rows = []
    for scenario in SCENARIOS:
        sub = metrics_df[metrics_df["scenario"] == scenario]
        ecl_ca = sub[sub["policy"] == "climate_aware"]["ecl"].values[0]
        ecl_cv = sub[sub["policy"] == "conventional"]["ecl"].values[0]
        obj_ca = sub[sub["policy"] == "climate_aware"]["objective"].values[0]
        obj_cv = sub[sub["policy"] == "conventional"]["objective"].values[0]
        ecl_red = ecl_cv - ecl_ca
        pct_red = ecl_red / ecl_cv * 100 if ecl_cv != 0 else 0.0
        table3_rows.append(
            {
                "scenario": scenario,
                "ecl_climate_aware": ecl_ca,
                "ecl_conventional": ecl_cv,
                "ecl_reduction": ecl_red,
                "ecl_reduction_pct": pct_red,
                "delta_objective": obj_ca - obj_cv,
            }
        )
    table3 = pd.DataFrame(table3_rows)

    sector_realloc_df = pd.concat(sector_realloc_records, ignore_index=True)

    # ── Print Table 3 ─────────────────────────────────────────────────────────
    print("\nTable 3 – ECL reduction (climate-aware vs conventional):")
    print(table3.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_table(metrics_df, "exp2_metrics.csv")
    save_table(table2, "exp2_table2.csv")
    save_table(table3, "exp2_table3.csv")
    save_table(sector_realloc_df, "exp2_sector_reallocations.csv")

    # Summary JSON
    summary = {}
    for scenario in SCENARIOS:
        sub = metrics_df[metrics_df["scenario"] == scenario]
        summary[scenario] = {
            pol: {k: float(sub[sub["policy"] == pol][k].values[0])
                  for k in ["objective", "ecl", "n_funded", "max_share", "hhi"]}
            for pol in POLICIES
        }
    with open("results/exp2_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nResults saved to results/")
    return {
        "metrics_df": metrics_df,
        "table2": table2,
        "table3": table3,
        "allocations": all_allocations,
        "sector_realloc_df": sector_realloc_df,
        "portfolio": portfolio,
        "z": z,
        "S_all": S_all,
    }


if __name__ == "__main__":
    run()
