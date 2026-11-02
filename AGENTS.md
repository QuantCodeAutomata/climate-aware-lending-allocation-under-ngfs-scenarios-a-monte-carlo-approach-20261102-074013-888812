# AGENTS.md – Repository guide

**Paper:** Climate aware lending allocation under NGFS scenarios – A Monte Carlo approach (2024)

This repository reproduces four experiments from the paper: analytical
terminal default probabilities under NGFS 2024 short-term scenarios (exp1),
climate-aware nonlinear lending optimisation against three benchmark policies
(exp2), Monte Carlo portfolio-loss simulation with PCA-based Gaussian and
Student-t copula dependence (exp3), and a structured robustness and
sensitivity analysis (exp4).  All data are generated synthetically; no
external API keys are required.

---

## Layout

```
src/                    Flat library modules (no subdirectories)
  config.py             All experiment parameters as UPPER_CASE constants
  portfolio.py          Synthetic obligor generation (N=1000, 50 sectors)
  pd_engine.py          Analytical PD computation, threshold calibration,
                        climate shock aggregation
  optimization.py       Separable nonlinear lending allocation solver
                        (dual decomposition + bisection on Lagrange multiplier)
  simulation.py         PCA decomposition, Gaussian / Student-t copula MC
  reporting.py          Summary statistics and CSV/JSON/PNG output

data/
  __init__.py
  generate_synthetic.py Synthetic NGFS sector shock generation for 4 scenarios,
                        50 sectors, 5 years, 3 channels

exp/
  __init__.py
  exp1.py               Analytical PDs under NGFS scenarios (Table 1)
  exp2.py               Lending optimisation vs benchmarks (Tables 2–3)
  exp3.py               Monte Carlo simulation (Table 3, Section 5.2)
  exp4.py               Robustness and sensitivity (Table 4, Section 5.2)

tests/
  conftest.py           Session fixtures; registers market_data marker
  test_pd_engine.py     Unit tests for PD computation and calibration
  test_portfolio.py     Unit tests for portfolio generation
  test_optimization.py  Unit tests for the separable solver
  test_simulation.py    Unit tests for PCA, latent simulation, ES_99

results/
  RESULTS.md            Metric comparisons with paper verdicts
  exp1_*.csv/png/json
  exp2_*.csv/json
  exp3_*.csv/png/json
  exp4_*.csv/png/json
```

No other top-level files or directories are present (`.git` excepted).

---

## Setup

```bash
pip install -r requirements.txt
```

No `.env` file or API keys are needed.

---

## Running tests

```bash
python -m pytest                        # full suite (71 tests)
python -m pytest -m "not market_data"   # skip any market-data tests
```

Tests are in `tests/`; run from the repository root without any `sys.path`
modifications.

---

## Running experiments

Run from the repository root:

```bash
python -m exp.exp1   # analytical PDs (~5 s)
python -m exp.exp2   # lending optimisation (~5 s)
python -m exp.exp3   # Monte Carlo simulation (~30 s)
python -m exp.exp4   # robustness analysis (~60 s)
```

Full paper configuration (same as above; no reduced parameters):

```bash
bash reproduce.sh
```

Each run **overwrites** the files in `results/`.  After a rerun, inspect the
new result files and update the numbers and verdicts in `results/RESULTS.md`
by hand.

---

## Conventions

- **All experiment parameters** live in `src/config.py` as module-level
  `UPPER_CASE` constants with a comment stating their meaning, unit, and paper
  source.  Do not add parameters elsewhere.

- **Data generation** is performed only in `data/generate_synthetic.py`.
  Experiment scripts and `src/` modules call this module; they do not contain
  their own data generators.

- **Tests** use small synthetic fixtures defined in `tests/conftest.py` or
  inline.  The `@pytest.mark.market_data` marker is reserved for tests that
  would call an external data API; none exist in this repository.

- **No secrets** go in any tracked file.  The repository requires no API keys.

- **Reproducibility.** All random draws use explicit seeds passed from
  `src/config.py` (`PORTFOLIO_SEED`, `MC_SEED`).  Changing a seed changes all
  downstream results.

---

## Key algorithms

### Separable nonlinear optimisation (exp2)

The per-obligor objective `f_i(x_i) = x_i η_i [1 − PD_i(x_i)]` is
concave and depends only on `x_i`, so the budget-constrained problem is
separable.  KKT conditions reduce to finding a common Lagrange multiplier
`λ` such that `Σ_i x_i*(λ) = K`.  `_solve_nonlinear` in
`src/optimization.py` bisects on `λ` (80 outer iterations) and solves for
each obligor's optimal `x_i` via a 52-iteration bisection on `f_i'(x_i) = λ`.
This is equivalent to R `nloptr::slsqp` at convergence and runs in ~0.5 s for
N = 1,000.

### PCA systemic covariance (exp3)

`src/simulation.py` builds the N×N covariance `K_{X_T}^{i,j}` analytically,
computes its eigendecomposition with `numpy.linalg.eigh`, retains `q` leading
components, and constructs standardised Gaussian latent variables preserving
unit marginal variance.  Student-t robustness re-scales the Gaussian latents
by `sqrt(W/ν)` where `W ~ χ²_ν` and then maps through the t-CDF and normal
quantile function to preserve marginal PDs.
