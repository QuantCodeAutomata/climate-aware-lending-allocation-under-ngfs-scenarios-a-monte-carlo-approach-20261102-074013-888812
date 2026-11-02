"""
Experiment 4: Robustness and sensitivity analysis for dependence assumptions,
structural calibration, shock mapping, diversification limits, and PCA dimension.

Reproduces Table 4 and Section 5.2 of the paper.

Sensitivity dimensions (paper Section 5.2):
  A. Dependence: Gaussian vs Student-t copula (nu = 4) – shared with exp3.
  B. Structural calibration: sigma ±20%, productivity a ±20%,
     lending-rate eta ±25%.
  C. Climate transmission: shock intensity × {0.5, 1, 2}.
  D. Diversification cap: kappa in {0.005, 0.01, 0.015, 0.03, 0.05}.
  E. PCA dimension: q in {2, 5, 10} – shared with exp3.

For structural calibration sensitivity (B), the reference thresholds z_i are
held fixed as stated in the paper's discussion of calibration robustness.

Paper: "Climate aware lending allocation under NGFS scenarios –
        A Monte Carlo approach", Section 5.2, Table 4.

Usage:
    python -m exp.exp4
"""

from __future__ import annotations

import json
import warnings
import numpy as np
import pandas as pd

from src.config import (
    K_BUDGET,
    N_OBLIGORS,
    KAPPA,
    LGD,
    M_DRAWS,
    Q_PCA,
    NU_T_COPULA,
    ES_LEVEL,
    MC_SEED,
    SCENARIOS,
    KAPPA_GRID,
    SIGMA_SCALE_GRID,
    A_SCALE_GRID,
    ETA_SCALE_GRID,
    SHOCK_SCALE_GRID,
)
from src.portfolio import generate_portfolio
from src.pd_engine import (
    calibrate_thresholds,
    compute_pd,
    compute_all_scenario_shocks,
)
from src.optimization import (
    solve_climate_aware,
    solve_conventional,
    solve_equal_weight,
    compute_metrics,
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
)
from src.reporting import (
    save_table,
    plot_sensitivity_kappa,
    RESULTS_DIR,
)
from data.generate_synthetic import get_ngfs_shocks


def _run_mc_es99(
    portfolio: pd.DataFrame,
    pd_arr: np.ndarray,
    x: np.ndarray,
    sigma: np.ndarray,
    q: int = Q_PCA,
    M: int = M_DRAWS,
    seed: int = MC_SEED,
    use_t: bool = False,
    nu: int = NU_T_COPULA,
) -> float:
    """Run a Monte Carlo simulation and return ES_99."""
    eigenvalues, eigenvectors, _ = pca_decompose(sigma, q)
    delta = compute_idiosyncratic_variance(sigma, eigenvalues, eigenvectors)
    Y = simulate_gaussian_latents(portfolio, eigenvalues, eigenvectors,
                                  delta, M=M, seed=seed)
    if use_t:
        Y = apply_t_copula(Y, nu=nu, seed=seed + 1000)
    D = default_indicators(Y, pd_arr)
    L = portfolio_losses(D, x, lgd=LGD)
    return float(expected_shortfall(L, ES_LEVEL))


def run(
    portfolio: pd.DataFrame | None = None,
    z: np.ndarray | None = None,
    S_all: np.ndarray | None = None,
    sigma_cov: np.ndarray | None = None,
) -> dict:
    """
    Execute experiment 4 robustness and sensitivity analysis.

    Returns
    -------
    dict with DataFrames for each sensitivity dimension
    """
    print("=== Experiment 4: Robustness and sensitivity analysis ===\n")

    # ── Shared baseline data ──────────────────────────────────────────────────
    if portfolio is None:
        portfolio = generate_portfolio()
    ngfs_shocks = get_ngfs_shocks()
    if z is None:
        z = calibrate_thresholds(portfolio)
    if S_all is None:
        S_all = compute_all_scenario_shocks(portfolio, ngfs_shocks)
    if sigma_cov is None:
        sigma_cov = build_systemic_covariance(portfolio)

    x_ref = np.full(N_OBLIGORS, K_BUDGET / N_OBLIGORS)
    x_conv_base, _, _ = solve_conventional(portfolio, z)

    # Pre-compute baseline climate-aware allocations
    x_ca_baseline = {}
    for s_idx, scenario in enumerate(SCENARIOS):
        S_s = S_all[s_idx]
        x_ca, _, _ = solve_climate_aware(portfolio, z, S_s, x0=x_conv_base.copy())
        x_ca_baseline[scenario] = x_ca

    # ── A. Dependence sensitivity (Gaussian vs t-copula) ─────────────────────
    print("A. Dependence sensitivity (Gaussian vs Student-t, nu=4)")
    dep_records: list[dict] = []
    eigenvalues_b, eigenvectors_b, _ = pca_decompose(sigma_cov, Q_PCA)
    delta_b = compute_idiosyncratic_variance(sigma_cov, eigenvalues_b, eigenvectors_b)
    Y_base = simulate_gaussian_latents(
        portfolio, eigenvalues_b, eigenvectors_b, delta_b, M=M_DRAWS, seed=MC_SEED
    )

    for s_idx, scenario in enumerate(SCENARIOS):
        S_s = S_all[s_idx]
        x_ca = x_ca_baseline[scenario]
        pd_arr = compute_pd(portfolio, x_ca, z, S_s)

        D_g = default_indicators(Y_base, pd_arr)
        L_g = portfolio_losses(D_g, x_ca, lgd=LGD)
        es99_g = expected_shortfall(L_g, ES_LEVEL)

        Y_t = apply_t_copula(Y_base, nu=NU_T_COPULA, seed=MC_SEED + 1000 + s_idx)
        D_t = default_indicators(Y_t, pd_arr)
        L_t = portfolio_losses(D_t, x_ca, lgd=LGD)
        es99_t = expected_shortfall(L_t, ES_LEVEL)

        dep_records.append(
            {
                "scenario": scenario,
                "es99_gauss": es99_g,
                "es99_t": es99_t,
                "pct_increase": (es99_t / es99_g - 1) * 100 if es99_g > 0 else float("nan"),
            }
        )
        print(f"  {scenario}: ES99(G)={es99_g:.4f}  ES99(t)={es99_t:.4f}  "
              f"+{(es99_t/es99_g-1)*100:.2f}%")

    dep_df = pd.DataFrame(dep_records)
    save_table(dep_df, "exp4_dependence_sensitivity.csv")

    # ── B. Structural calibration sensitivity ─────────────────────────────────
    print("\nB. Structural calibration sensitivity")
    # For each perturbation, keep thresholds z fixed (paper Section 5.2)
    calib_records: list[dict] = []
    # Focus on first scenario and climate-aware allocation for MC metrics
    # Use mean PD across obligors as summary for all scenarios
    for scale_type, scale_grid, attr in [
        ("sigma_scale", SIGMA_SCALE_GRID, "sigma"),
        ("a_scale",     A_SCALE_GRID,     "a"),
        ("eta_scale",   ETA_SCALE_GRID,   "eta"),
    ]:
        for scale in scale_grid:
            # Create perturbed portfolio (clone, scale attribute)
            port_p = portfolio.copy()
            port_p[attr] = port_p[attr] * scale

            for s_idx, scenario in enumerate(SCENARIOS):
                S_s = S_all[s_idx]
                # Use baseline z (fixed thresholds, paper Section 5.2)
                pd_arr_p = compute_pd(port_p, x_ref, z, S_s)
                mean_pd_p = float(pd_arr_p.mean())
                calib_records.append(
                    {
                        "sensitivity": scale_type,
                        "scale": scale,
                        "scenario": scenario,
                        "mean_pd": mean_pd_p,
                    }
                )
            # Print representative (DAPS)
            sub = [r for r in calib_records
                   if r["sensitivity"] == scale_type
                   and r["scale"] == scale
                   and r["scenario"] == "DAPS"]
            if sub:
                print(f"  {scale_type}={scale:.2f} DAPS mean PD = "
                      f"{sub[0]['mean_pd'] * 100:.4f}%")

    calib_df = pd.DataFrame(calib_records)
    save_table(calib_df, "exp4_calibration_sensitivity.csv")

    # ── C. Climate shock intensity sensitivity ────────────────────────────────
    print("\nC. Climate shock intensity sensitivity")
    shock_records: list[dict] = []
    for scale in SHOCK_SCALE_GRID:
        S_scaled = compute_all_scenario_shocks(portfolio, ngfs_shocks,
                                               shock_scale=scale)
        for s_idx, scenario in enumerate(SCENARIOS):
            S_s = S_scaled[s_idx]
            pd_arr_s = compute_pd(portfolio, x_ref, z, S_s)
            shock_records.append(
                {
                    "shock_scale": scale,
                    "scenario": scenario,
                    "mean_pd": float(pd_arr_s.mean()),
                    "std_pd": float(pd_arr_s.std()),
                }
            )
        # Print
        for row in shock_records[-len(SCENARIOS):]:
            print(f"  scale={scale} {row['scenario']:6s}: "
                  f"mean PD={row['mean_pd']*100:.4f}%  "
                  f"std={row['std_pd']*100:.4f}%")

    shock_df = pd.DataFrame(shock_records)
    save_table(shock_df, "exp4_shock_sensitivity.csv")

    # ── D. Diversification cap sensitivity ────────────────────────────────────
    print("\nD. Diversification cap sensitivity")
    kappa_records: list[dict] = []
    for kappa_val in KAPPA_GRID:
        for s_idx, scenario in enumerate(SCENARIOS):
            S_s = S_all[s_idx]
            # Solve optimisation with this cap
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_k, _, _ = solve_climate_aware(
                    portfolio, z, S_s,
                    kappa=kappa_val,
                    x0=None,
                )
            m = compute_metrics(portfolio, x_k, z, S_s)
            kappa_records.append(
                {
                    "kappa": kappa_val,
                    "scenario": scenario,
                    "n_funded": m["n_funded"],
                    "max_share": m["max_share"],
                    "hhi": m["hhi"],
                    "objective": m["objective"],
                    "ecl": m["ecl"],
                }
            )
        print(f"  kappa={kappa_val:.3f}: "
              f"N+={kappa_records[-1]['n_funded']}  "
              f"max_share={kappa_records[-1]['max_share']:.4f}  "
              f"HHI={kappa_records[-1]['hhi']:.6f}")

    kappa_df = pd.DataFrame(kappa_records)
    save_table(kappa_df, "exp4_kappa_sensitivity.csv")
    plot_sensitivity_kappa(kappa_df, filename="exp4_kappa_sensitivity.png")

    # ── E. PCA dimension robustness ───────────────────────────────────────────
    # (Detailed in exp3; Table 4 summary recomputed here)
    print("\nE. PCA dimension robustness (variance explained)")
    pca_var_records: list[dict] = []
    from src.config import Q_PCA_ROBUSTNESS
    for q in Q_PCA_ROBUSTNESS:
        _, _, var_q = pca_decompose(sigma_cov, q)
        pca_var_records.append({"q": q, "var_explained": var_q})
        print(f"  q={q}: var_explained = {var_q * 100:.8f}%")

    pca_var_df = pd.DataFrame(pca_var_records)
    save_table(pca_var_df, "exp4_pca_variance.csv")

    # ── Consolidated Table 4 ──────────────────────────────────────────────────
    # Dependence + calibration summaries for DAPS baseline PD
    no_climate_pd = compute_pd(portfolio, x_ref, z, S_s=None)
    ref_mean_pd = float(no_climate_pd.mean())

    table4_rows: list[dict] = []
    for rec in dep_records:
        table4_rows.append(
            {
                "dimension": "dependence",
                "variant": f"Gaussian",
                "scenario": rec["scenario"],
                "metric": "ES99",
                "value": rec["es99_gauss"],
            }
        )
        table4_rows.append(
            {
                "dimension": "dependence",
                "variant": f"Student-t(nu={NU_T_COPULA})",
                "scenario": rec["scenario"],
                "metric": "ES99",
                "value": rec["es99_t"],
            }
        )

    for rec in calib_records:
        table4_rows.append(
            {
                "dimension": rec["sensitivity"],
                "variant": f"scale={rec['scale']:.2f}",
                "scenario": rec["scenario"],
                "metric": "mean_pd",
                "value": rec["mean_pd"],
            }
        )

    for rec in shock_records:
        table4_rows.append(
            {
                "dimension": "shock_scale",
                "variant": f"scale={rec['shock_scale']}",
                "scenario": rec["scenario"],
                "metric": "mean_pd",
                "value": rec["mean_pd"],
            }
        )

    table4_df = pd.DataFrame(table4_rows)
    save_table(table4_df, "exp4_table4.csv")

    # Summary JSON
    summary = {
        "reference_mean_pd": ref_mean_pd,
        "dependence": dep_df.to_dict(orient="records"),
        "pca_variance": pca_var_df.to_dict(orient="records"),
    }
    with open("results/exp4_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nResults saved to results/")
    return {
        "dep_df": dep_df,
        "calib_df": calib_df,
        "shock_df": shock_df,
        "kappa_df": kappa_df,
        "pca_var_df": pca_var_df,
        "table4_df": table4_df,
    }


if __name__ == "__main__":
    run()
