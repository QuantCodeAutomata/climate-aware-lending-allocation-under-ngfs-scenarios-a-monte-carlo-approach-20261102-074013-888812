"""
Experiment 3: Monte Carlo joint default and portfolio loss simulation with
PCA-based Gaussian dependence and Student-t copula robustness.

Reproduces Table 3 (MC loss columns) and Section 5.2 robustness analysis
of the paper.

Steps:
  1. Reuse portfolio, calibrated thresholds, climate shocks, and optimised
     allocations from Experiments 1 and 2.
  2. Build the systemic covariance matrix and PCA decomposition (q = 5).
  3. Simulate M = 6,000 joint default scenarios using Gaussian latents.
  4. Validate simulated marginal PDs against analytical values.
  5. Estimate portfolio loss distributions and ES_99 for climate-aware and
     equal-weight policies across all four NGFS scenarios.
  6. Replicate paired MC loss comparisons (common latent draws).
  7. Apply Student-t copula (nu = 4) and compare ES_99.
  8. Test PCA robustness: q in {2, 5, 10}.
  9. Save results to results/.

Paper: "Climate aware lending allocation under NGFS scenarios –
        A Monte Carlo approach", Sections 2.4–2.5, 4.3, 5.2, Table 3.

Usage:
    python -m exp.exp3
"""

import json
import numpy as np
import pandas as pd

from src.config import (
    K_BUDGET,
    N_OBLIGORS,
    LGD,
    M_DRAWS,
    Q_PCA,
    Q_PCA_ROBUSTNESS,
    NU_T_COPULA,
    ES_LEVEL,
    MC_SEED,
    SCENARIOS,
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
    paired_loss_comparison,
    run_simulation,
)
from src.reporting import (
    save_table,
    plot_loss_distributions,
    RESULTS_DIR,
)
from data.generate_synthetic import get_ngfs_shocks


def run(
    portfolio: pd.DataFrame | None = None,
    z: np.ndarray | None = None,
    S_all: np.ndarray | None = None,
    allocations: dict | None = None,
) -> dict:
    """
    Execute experiment 3.

    Parameters
    ----------
    portfolio  : if provided, reuse from earlier experiments
    z          : calibrated thresholds (recomputed if None)
    S_all      : (n_scenarios, N) climate shocks (recomputed if None)
    allocations: dict[scenario][policy] → (N,) exposures

    Returns
    -------
    dict with loss arrays, ES99 values, and validation statistics
    """
    print("=== Experiment 3: Monte Carlo simulation (Gaussian + t-copula) ===\n")

    # ── 1. Shared data ────────────────────────────────────────────────────────
    if portfolio is None:
        portfolio = generate_portfolio()
    ngfs_shocks = get_ngfs_shocks()
    if z is None:
        z = calibrate_thresholds(portfolio)
    if S_all is None:
        S_all = compute_all_scenario_shocks(portfolio, ngfs_shocks)

    # Conventional allocation (no climate; shared across scenarios)
    if allocations is None:
        print("Solving allocations for MC simulation...")
        x_conv, _, _ = solve_conventional(portfolio, z)
        allocations = {}
        for s_idx, scenario in enumerate(SCENARIOS):
            S_s = S_all[s_idx]
            x_ca, _, _ = solve_climate_aware(portfolio, z, S_s, x0=x_conv.copy())
            x_ew = solve_equal_weight()
            allocations[scenario] = {
                "climate_aware": x_ca,
                "conventional": x_conv.copy(),
                "equal_weight": x_ew,
            }

    # ── 2. Build PCA (once; covariance is policy-independent) ─────────────────
    print("Building systemic covariance matrix and PCA decomposition...")
    sigma = build_systemic_covariance(portfolio)
    eigenvalues, eigenvectors, var_explained = pca_decompose(sigma, Q_PCA)
    delta = compute_idiosyncratic_variance(sigma, eigenvalues, eigenvectors)
    print(f"  q = {Q_PCA} components retained; "
          f"variance explained = {var_explained * 100:.8f}%")
    assert var_explained > 0.9999, \
        f"Expected >99.99% variance explained, got {var_explained * 100:.4f}%"

    # ── 3. Pre-simulate common Gaussian latents (for paired comparison) ────────
    print(f"Simulating {M_DRAWS} Gaussian latent paths...")
    Y_common = simulate_gaussian_latents(
        portfolio, eigenvalues, eigenvectors, delta, M=M_DRAWS, seed=MC_SEED
    )

    # ── 4. Main simulation: Gaussian + t-copula for each scenario ─────────────
    records: list[dict] = []
    all_losses: dict[str, dict[str, np.ndarray]] = {}

    for s_idx, scenario in enumerate(SCENARIOS):
        print(f"\n── Scenario: {scenario} ──────────────────────────")
        S_s = S_all[s_idx]
        all_losses[scenario] = {}

        for policy in ["climate_aware", "equal_weight"]:
            if policy not in allocations.get(scenario, {}):
                continue
            x_pol = allocations[scenario][policy]
            pd_arr = compute_pd(portfolio, x_pol, z, S_s)

            # Gaussian defaults and losses
            D_gauss = default_indicators(Y_common, pd_arr)
            sim_pd = D_gauss.mean(axis=0)
            losses_g = portfolio_losses(D_gauss, x_pol, lgd=LGD)
            es99_g = expected_shortfall(losses_g, ES_LEVEL)

            # Student-t copula
            Y_t = apply_t_copula(Y_common, nu=NU_T_COPULA, seed=MC_SEED + 1000 + s_idx)
            D_t = default_indicators(Y_t, pd_arr)
            losses_t = portfolio_losses(D_t, x_pol, lgd=LGD)
            es99_t = expected_shortfall(losses_t, ES_LEVEL)

            # Marginal PD validation
            analytical_pd = pd_arr
            mc_error = np.sqrt(analytical_pd * (1 - analytical_pd) / M_DRAWS)
            n_within_4se = int(np.sum(np.abs(sim_pd - analytical_pd) <= 4 * mc_error))

            all_losses[scenario][policy] = losses_g

            rec = {
                "scenario": scenario,
                "policy": policy,
                "mean_loss_gauss": float(losses_g.mean()),
                "es99_gauss": float(es99_g),
                "es99_t": float(es99_t),
                "es99_t_increase_pct": (es99_t / es99_g - 1.0) * 100.0
                    if es99_g > 0 else float("nan"),
                "sim_pd_mean": float(sim_pd.mean()),
                "analytical_pd_mean": float(analytical_pd.mean()),
                "n_within_4se": n_within_4se,
            }
            records.append(rec)
            print(f"  {policy:15s}: mean_loss={rec['mean_loss_gauss']:.4f}  "
                  f"ES_99(G)={es99_g:.4f}  ES_99(t)={es99_t:.4f}  "
                  f"t_increase={rec['es99_t_increase_pct']:.2f}%  "
                  f"PD_valid={n_within_4se}/{N_OBLIGORS}")

    # ── 5. Paired MC comparison: climate-aware vs conventional ────────────────
    print("\nPaired MC comparisons (climate-aware vs conventional):")
    paired_records: list[dict] = []
    for s_idx, scenario in enumerate(SCENARIOS):
        S_s = S_all[s_idx]
        if "climate_aware" not in allocations.get(scenario, {}):
            continue
        x_ca = allocations[scenario]["climate_aware"]
        x_cv = allocations[scenario].get("conventional", solve_equal_weight())

        pd_ca = compute_pd(portfolio, x_ca, z, S_s)
        pd_cv = compute_pd(portfolio, x_cv, z, S_s)

        D_ca = default_indicators(Y_common, pd_ca)
        D_cv = default_indicators(Y_common, pd_cv)

        losses_ca = portfolio_losses(D_ca, x_ca, lgd=LGD)
        losses_cv = portfolio_losses(D_cv, x_cv, lgd=LGD)

        comp = paired_loss_comparison(losses_ca, losses_cv, ci_level=0.95)
        comp["scenario"] = scenario
        paired_records.append(comp)
        print(f"  {scenario:6s}: mean_diff={comp['mean_diff']:.4f}  "
              f"95% CI=[{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}]")

    # ── 6. PCA robustness: q in {2, 5, 10} ────────────────────────────────────
    print("\nPCA dimension robustness:")
    pca_records: list[dict] = []
    # Use first scenario (DAPS) climate-aware allocation for brevity
    s_demo, scenario_demo = 0, SCENARIOS[0]
    S_demo = S_all[s_demo]
    x_demo = allocations[scenario_demo].get("climate_aware", solve_equal_weight())
    pd_demo = compute_pd(portfolio, x_demo, z, S_demo)

    for q in Q_PCA_ROBUSTNESS:
        ev_q, uvec_q, var_q = pca_decompose(sigma, q)
        delta_q = compute_idiosyncratic_variance(sigma, ev_q, uvec_q)
        Y_q = simulate_gaussian_latents(
            portfolio, ev_q, uvec_q, delta_q, M=M_DRAWS, seed=MC_SEED
        )
        D_q = default_indicators(Y_q, pd_demo)
        losses_q = portfolio_losses(D_q, x_demo, lgd=LGD)
        es99_q = expected_shortfall(losses_q, ES_LEVEL)
        pca_records.append(
            {
                "q": q,
                "var_explained": var_q,
                "es99_gauss": es99_q,
                "scenario": scenario_demo,
            }
        )
        print(f"  q={q:2d}: var_explained={var_q * 100:.8f}%  "
              f"ES_99={es99_q:.4f}")

    # ── 7. Build result DataFrames and save ───────────────────────────────────
    mc_df = pd.DataFrame(records)
    paired_df = pd.DataFrame(paired_records)
    pca_rob_df = pd.DataFrame(pca_records)

    save_table(mc_df, "exp3_mc_results.csv")
    save_table(paired_df, "exp3_paired_comparison.csv")
    save_table(pca_rob_df, "exp3_pca_robustness.csv")

    # Plot loss distributions for the first scenario
    s0 = SCENARIOS[0]
    if all_losses.get(s0):
        plot_loss_distributions(
            all_losses[s0], filename=f"exp3_loss_distributions_{s0}.png"
        )

    # JSON summary
    summary = {
        "var_explained_q5": var_explained,
        "scenarios": {},
    }
    for rec in records:
        key = f"{rec['scenario']}_{rec['policy']}"
        summary["scenarios"][key] = {
            "es99_gauss": rec["es99_gauss"],
            "es99_t": rec["es99_t"],
            "es99_t_increase_pct": rec["es99_t_increase_pct"],
        }
    with open("results/exp3_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nResults saved to results/")
    return {
        "mc_df": mc_df,
        "paired_df": paired_df,
        "pca_rob_df": pca_rob_df,
        "Y_common": Y_common,
        "var_explained": var_explained,
    }


if __name__ == "__main__":
    run()
