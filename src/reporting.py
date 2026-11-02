"""
Reporting utilities: summary statistics, table formatting, and plots.

Centralises all output-generation logic so that experiment scripts remain thin.

Paper: "Climate aware lending allocation under NGFS scenarios –
        A Monte Carlo approach".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import SCENARIOS

RESULTS_DIR = Path("results")


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)


# ── Cross-sectional PD summaries ─────────────────────────────────────────────

def pd_cross_section_summary(
    pd_matrix: np.ndarray,
    scenario_names: list[str] = SCENARIOS,
    no_climate_pd: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute cross-sectional distribution statistics of terminal PDs across
    obligors for each scenario.

    Columns: scenario, mean, median, std, p5, p95, min, max
    (paper Section 4.1, Table 1 equivalents).

    Parameters
    ----------
    pd_matrix      : (n_scenarios, N) array of obligor PDs
    scenario_names : list of scenario labels (length n_scenarios)
    no_climate_pd  : (N,) no-climate reference PDs (optional extra row)

    Returns
    -------
    DataFrame with one row per scenario (plus optional reference row)
    """
    rows = []
    if no_climate_pd is not None:
        arr = no_climate_pd
        rows.append(
            {
                "scenario": "No-climate reference",
                "mean": arr.mean(),
                "median": np.median(arr),
                "std": arr.std(),
                "p5": np.percentile(arr, 5),
                "p95": np.percentile(arr, 95),
                "min": arr.min(),
                "max": arr.max(),
            }
        )
    for s, name in enumerate(scenario_names):
        arr = pd_matrix[s]
        rows.append(
            {
                "scenario": name,
                "mean": arr.mean(),
                "median": np.median(arr),
                "std": arr.std(),
                "p5": np.percentile(arr, 5),
                "p95": np.percentile(arr, 95),
                "min": arr.min(),
                "max": arr.max(),
            }
        )
    return pd.DataFrame(rows)


def sector_pd_change_summary(
    portfolio: pd.DataFrame,
    pd_matrix: np.ndarray,
    no_climate_pd: np.ndarray,
    sector_table: pd.DataFrame,
    scenario_names: list[str] = SCENARIOS,
    illustrative_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute sector-average PD changes relative to no-climate reference.

    Parameters
    ----------
    portfolio          : obligor DataFrame with column 'sector_idx'
    pd_matrix          : (n_scenarios, N) scenario PDs
    no_climate_pd      : (N,) reference PDs
    sector_table       : DataFrame with 'sector_idx' and 'label' columns
    scenario_names     : scenario labels
    illustrative_labels: if not None, filter to these sector labels

    Returns
    -------
    DataFrame with columns: sector_idx, label, type, {scenario}_pd_change, ...
    """
    sector_idx = portfolio["sector_idx"].values
    n_scenarios = len(scenario_names)
    n_sectors = sector_table.shape[0]

    records = []
    for sec_id in range(n_sectors):
        mask = sector_idx == sec_id
        if mask.sum() == 0:
            continue
        ref_mean = no_climate_pd[mask].mean()
        row = {"sector_idx": sec_id}
        for s, name in enumerate(scenario_names):
            scen_mean = pd_matrix[s][mask].mean()
            row[f"{name}_pd_change"] = scen_mean - ref_mean
        records.append(row)

    df = pd.DataFrame(records)
    label_map = sector_table.set_index("sector_idx")["label"]
    type_map = sector_table.set_index("sector_idx")["type"]
    df["label"] = df["sector_idx"].map(label_map)
    df["type"] = df["sector_idx"].map(type_map)

    if illustrative_labels is not None:
        df = df[df["label"].isin(illustrative_labels)].copy()

    return df.sort_values("sector_idx").reset_index(drop=True)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_pd_distributions(
    pd_matrix: np.ndarray,
    no_climate_pd: np.ndarray,
    scenario_names: list[str] = SCENARIOS,
    filename: str = "pd_distributions.png",
) -> None:
    """
    Plot cross-sectional PD distributions across scenarios (paper Section 4.1).
    """
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = 50
    ax.hist(
        no_climate_pd * 100,
        bins=bins,
        alpha=0.4,
        label="No-climate reference",
        density=True,
        color="grey",
    )
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for s, (name, col) in enumerate(zip(scenario_names, colors)):
        ax.hist(
            pd_matrix[s] * 100,
            bins=bins,
            alpha=0.4,
            label=name,
            density=True,
            color=col,
        )
    ax.set_xlabel("Five-year terminal PD (%)")
    ax.set_ylabel("Density")
    ax.set_title("Cross-sectional distribution of terminal default probabilities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=120)
    plt.close(fig)


def plot_sector_pd_changes(
    sector_df: pd.DataFrame,
    scenario_names: list[str] = SCENARIOS,
    filename: str = "sector_pd_changes.png",
) -> None:
    """
    Bar chart of sector-level PD changes for illustrative sectors.
    """
    _ensure_results_dir()
    cols = [f"{s}_pd_change" for s in scenario_names]
    plot_df = sector_df.set_index("label")[cols] * 100  # convert to %
    plot_df.columns = scenario_names

    ax = plot_df.plot(kind="bar", figsize=(12, 5))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sector")
    ax.set_ylabel("PD change vs reference (pp)")
    ax.set_title("Sector-level terminal PD changes under NGFS scenarios")
    ax.legend(title="Scenario")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=120)
    plt.close()


def plot_loss_distributions(
    losses_dict: dict[str, np.ndarray],
    filename: str = "loss_distributions.png",
) -> None:
    """
    Plot portfolio loss distributions for multiple policies/scenarios.

    Parameters
    ----------
    losses_dict : mapping label → (M,) loss array
    """
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, losses in losses_dict.items():
        ax.hist(losses, bins=80, alpha=0.5, density=True, label=label)
    ax.set_xlabel("Portfolio credit loss")
    ax.set_ylabel("Density")
    ax.set_title("Portfolio loss distributions (Gaussian copula)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=120)
    plt.close(fig)


def plot_sensitivity_kappa(
    kappa_results: pd.DataFrame,
    filename: str = "sensitivity_kappa.png",
) -> None:
    """
    Line plot of N_funded and HHI as function of concentration cap kappa.
    """
    _ensure_results_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for scenario in kappa_results["scenario"].unique():
        sub = kappa_results[kappa_results["scenario"] == scenario]
        axes[0].plot(sub["kappa"] * 100, sub["n_funded"], marker="o", label=scenario)
        axes[1].plot(sub["kappa"] * 100, sub["hhi"], marker="o", label=scenario)
    axes[0].set_xlabel("Concentration cap kappa (%)")
    axes[0].set_ylabel("Funded obligors N+")
    axes[0].set_title("Number of funded obligors vs cap")
    axes[0].legend()
    axes[1].set_xlabel("Concentration cap kappa (%)")
    axes[1].set_ylabel("HHI")
    axes[1].set_title("Portfolio concentration (HHI) vs cap")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=120)
    plt.close(fig)


# ── CSV / table savers ────────────────────────────────────────────────────────

def save_table(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame as CSV in the results directory."""
    _ensure_results_dir()
    df.to_csv(RESULTS_DIR / filename, index=False, float_format="%.6f")


def format_pct(x: float, decimals: int = 3) -> str:
    return f"{x * 100:.{decimals}f}%"
