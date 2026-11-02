"""
Configuration for climate-aware lending allocation experiments.

All parameters are module-level UPPER_CASE variables with comments stating
meaning, unit, and source (paper section or equation).

Paper: "Climate aware lending allocation under NGFS scenarios - A Monte Carlo approach"
"""

# ---------------------------------------------------------------------------
# Portfolio construction parameters  (Section 3, Implementation Details)
# ---------------------------------------------------------------------------

N_OBLIGORS = 1000            # Number of synthetic obligors; paper Section 3
N_SECTORS = 50               # Number of harmonised sectors; paper Section 3
K_BUDGET = 100.0             # Total lending budget (monetary units); paper Section 3
T_MATURITY = 5               # Loan maturity in years; paper Section 3
R_RISKFREE = 0.02            # Risk-free rate r; paper Section 3
LAMBDA_REF = 0.03            # Reference hazard rate (year^-1); paper Section 3

# Obligor parameter distributions (paper Section 3, Implementation Details)
P0_INITIAL = 0.0             # Initial log-production p0^i; paper Section 3
A_LOW = 0.0                  # Lower bound of productivity a^i ~ Uniform(A_LOW, A_HIGH)
A_HIGH = 0.5                 # Upper bound; paper Section 3
B_LOW = 1.0                  # Lower bound of mean-reversion speed b^i ~ Uniform(B_LOW, B_HIGH)
B_HIGH = 4.0                 # Upper bound; paper Section 3
SIGMA_LOW = 0.1              # Lower bound of volatility sigma^i ~ Uniform(SIGMA_LOW, SIGMA_HIGH)
SIGMA_HIGH = 0.5             # Upper bound; paper Section 3
RHO_LOW = -1.0               # Lower bound of systemic correlation rho^i ~ Uniform(RHO_LOW, RHO_HIGH)
RHO_HIGH = 1.0               # Upper bound; paper Section 3
LAMBDA_SPREAD_LOW = 0.025    # Lower bound of lending spread Lambda_i ~ Uniform(low, high)
LAMBDA_SPREAD_HIGH = 0.06    # Upper bound; paper Section 3

# Reference allocation
X_REF = K_BUDGET / N_OBLIGORS   # Equal-weight reference allocation x_i^ref = K/N = 0.1; paper Section 3

# ---------------------------------------------------------------------------
# Climate mapping parameters  (Section 2, Climate Scenario Integration)
# ---------------------------------------------------------------------------

C_J = 0.01          # Climate sensitivity coefficient c_j (same for all channels j); paper Section 3
ALPHA_J = 0.0       # Offset alpha_j (same for all channels j); paper Section 3

# beta_j values assigned to channels j = 1 (production), 2 (financial), 3 (physical)
# Assignment not explicitly stated in the paper; natural ordering from least to most damped.
BETA_J = [0.1, 0.5, 0.8]   # Response scale parameters; paper Section 3

# Production ratio bounds for the GEME3 production deviation (paper Section 3)
PROD_RATIO_LOW = 0.25       # Lower bound on scenario/baseline production ratio
PROD_RATIO_HIGH = 4.0       # Upper bound on scenario/baseline production ratio

# ---------------------------------------------------------------------------
# Optimisation parameters  (Section 2.6, Climate-Aware Lending Optimisation)
# ---------------------------------------------------------------------------

KAPPA = 0.05        # Diversification cap: x_i <= kappa * K; paper Section 2.6
LGD = 0.45          # Loss given default; paper Section 2.6

# SLSQP numerical tolerance for classifying obligors as "funded" (paper does not specify)
FUNDED_TOL = 1e-6   # Threshold for x_i > 0 (numerical tolerance); not specified in the paper

# ---------------------------------------------------------------------------
# Monte Carlo simulation parameters  (Section 2.5, Monte Carlo Simulation)
# ---------------------------------------------------------------------------

M_DRAWS = 6000      # Number of Monte Carlo paths M; paper Section 2.5
Q_PCA = 5           # Number of PCA components retained in baseline; paper Section 2.4
Q_PCA_ROBUSTNESS = [2, 5, 10]  # PCA dimensions for robustness checks; paper Section 5.2
NU_T_COPULA = 4     # Degrees of freedom for Student-t copula robustness; paper Section 5.2
ES_LEVEL = 0.99     # Expected shortfall confidence level; paper Section 2.5

# ---------------------------------------------------------------------------
# Scenarios  (Section 1.4, Short-Term Climate Scenarios, NGFS 2024)
# ---------------------------------------------------------------------------

SCENARIOS = ["DAPS", "HWTP", "SWUC", "DiRe"]   # Four NGFS 2024 short-term scenarios
SCENARIO_YEARS = list(range(2025, 2030))        # Annual observations 2025–2029

# ---------------------------------------------------------------------------
# Synthetic data generation seeds
# ---------------------------------------------------------------------------

PORTFOLIO_SEED = 42     # Random seed for obligor generation; not specified in the paper
MC_SEED = 123           # Random seed for Monte Carlo simulation; not specified in the paper

# ---------------------------------------------------------------------------
# Robustness / sensitivity analysis parameters  (Section 5.2)
# ---------------------------------------------------------------------------

# Diversification cap sweep (paper Section 5.2)
KAPPA_GRID = [0.005, 0.01, 0.015, 0.03, 0.05]

# Volatility perturbation factors (paper Section 5.2): ±20%
SIGMA_SCALE_GRID = [0.8, 1.0, 1.2]

# Productivity perturbation factors (paper Section 5.2): ±20%
A_SCALE_GRID = [0.8, 1.0, 1.2]

# Lending-rate perturbation factors (paper Section 5.2): ±25%
ETA_SCALE_GRID = [0.75, 1.0, 1.25]

# Climate-shock intensity scaling factors (paper Section 5.2)
SHOCK_SCALE_GRID = [0.5, 1.0, 2.0]

# Selected illustrative sectors for sector analysis (paper Section 4.1)
# Indices into the 50-sector list; labels defined in data/generate_synthetic.py
ILLUSTRATIVE_SECTOR_LABELS = [
    "EV transport equipment",
    "Wind power",
    "Power supply",
    "Gas",
    "Crude oil",
    "Oil",
]
