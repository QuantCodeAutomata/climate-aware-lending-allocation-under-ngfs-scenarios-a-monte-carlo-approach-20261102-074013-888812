#!/usr/bin/env bash
# Reproduce all experiments and run the full test suite.
# Usage:  bash reproduce.sh

set -u
cd "$(dirname "$0")"

echo "=== Running tests ==="
python -m pytest -m "not market_data" -v
echo ""

echo "=== Experiment 1: Analytical terminal default probabilities ==="
python -m exp.exp1
echo ""

echo "=== Experiment 2: Lending optimisation vs benchmarks ==="
python -m exp.exp2
echo ""

echo "=== Experiment 3: Monte Carlo joint default simulation ==="
python -m exp.exp3
echo ""

echo "=== Experiment 4: Robustness and sensitivity analysis ==="
python -m exp.exp4
echo ""

echo "=== All experiments complete. See results/ for output ==="
