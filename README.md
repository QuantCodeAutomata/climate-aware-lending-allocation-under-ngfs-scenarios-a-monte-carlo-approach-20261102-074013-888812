# Climate aware lending allocation under NGFS scenarios – A Monte Carlo approach

**Paper title:** Climate aware lending allocation under NGFS scenarios – A Monte Carlo approach  
**Authors:** Not stated in the paper  
**Year:** 2024 (NGFS 2024 data vintage)

This repository reproduces four experiments from the paper using a synthetic
1,000-obligor portfolio mapped to 50 harmonised sectors and annual NGFS
2025–2029 climate scenario shocks.

## Experiments

| ID | Title |
|----|-------|
| exp1 | Scenario-dependent analytical terminal default probabilities under NGFS short-term scenarios |
| exp2 | Climate-aware nonlinear lending optimisation vs conventional, equal-weight, and fixed-PD benchmarks |
| exp3 | Monte Carlo joint default and portfolio loss simulation with PCA-based Gaussian dependence and Student-t copula robustness |
| exp4 | Robustness and sensitivity analysis (dependence, structural calibration, shock mapping, diversification limits, PCA dimension) |

## Layout

```
src/           Core library modules (config, portfolio, pd_engine, optimization, simulation, reporting)
exp/           Experiment scripts (exp1–exp4)
data/          Synthetic data generators (generate_synthetic.py)
tests/         Pytest test suite
results/       Generated tables, plots, and RESULTS.md
```

## Setup

```bash
pip install -r requirements.txt
```

No external API keys are required; all data are generated synthetically.

## Running experiments

Each experiment can be run independently:

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
```

To run all experiments in sequence:

```bash
bash reproduce.sh
```

## Running tests

```bash
python -m pytest
```

Skip slow tests if needed:

```bash
python -m pytest -m "not market_data"
```

## Results

All output tables (CSV), plots (PNG), and summary metrics (JSON) are written
to `results/`.  See `results/RESULTS.md` for a comparison of computed metrics
against the paper's reported values.

## Configuration

All experiment parameters are defined as module-level constants in
`src/config.py`.  Each constant is annotated with its source (paper section
or "not specified in the paper").
