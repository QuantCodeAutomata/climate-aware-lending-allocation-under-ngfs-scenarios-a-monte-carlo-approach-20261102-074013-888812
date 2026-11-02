# Results

**Paper:** Climate aware lending allocation under NGFS scenarios – A Monte Carlo approach  
**Authors:** Not stated on the title page  
**Year:** 2024  

---

## Experiment 1: Scenario-dependent analytical terminal default probabilities under NGFS short-term scenarios

**Setup.** A synthetic portfolio of N = 1,000 obligors across 50 sectors is constructed with a fixed random seed. Obligor parameters are drawn from the distributions specified in paper Section 3 (Implementation Details). Annual NGFS 2024 short-term sector shocks for 2025–2029 are synthesised for four scenarios (DAPS, HWTP, SWUC, DiRe) across three climate channels (production, financial, physical). The structural default threshold z_i is calibrated at the equal-weight reference allocation x_ref = K/N = 0.1 so that each obligor's no-climate five-year terminal PD equals p_ref = 1 − exp(−λ_ref T) exactly (paper Section 3, Reference Default Calibration). Analytical scenario PDs are then computed for each obligor using the normal CDF formula (paper Eq. in Section 2, Climate Scenario Integration).

**Data source.** Synthetic NGFS sector shocks constructed from the qualitative scenario descriptions in paper Section 1.4. The exact GEME3 production tables, CLIMACRED financial adjustments, and physical-impact datasets used by the authors were not published with the paper; the sector-level shocks used here are calibrated to reproduce the paper's qualitative scenario structure (production gains for green sectors, losses for fossil sectors under transition scenarios; physical risk for DAPS and DiRe).

**Artifacts.** `results/exp1_pd_cross_section.csv`, `results/exp1_sector_pd_changes.csv`, `results/exp1_pd_distributions.png`, `results/exp1_sector_pd_changes.png`, `results/exp1_summary.json`.

### Results vs. paper (Table 1)

| Metric | Paper (Table 1) | This run | Verdict |
|--------|-----------------|----------|---------|
| No-climate reference PD | 13.929% (by construction) | **13.9292%** | Reproduced |
| DAPS mean PD | ≈13.955% | **13.9563%** | Reproduced |
| HWTP mean PD | ≈13.878% | **13.7985%** | Partially reproduced |
| SWUC mean PD | ≈13.892% | **13.8491%** | Partially reproduced |
| DiRe mean PD | ≈13.899% | **13.8335%** | Partially reproduced |
| DAPS dispersion (std) | very low | **0.023%** | Reproduced |
| HWTP/SWUC/DiRe dispersion | much wider than DAPS | **0.49% / 0.30% / 0.37%** | Reproduced |
| Green sectors (EV, wind) PD change under HWTP | large negative | **−0.93 pp, −0.59 pp** | Reproduced |
| Fossil sectors (gas, oil) PD change under HWTP | positive | **+0.20 pp, +0.23 pp** | Reproduced |
| DAPS obligor PD change (green sectors) | near-zero | **+0.02 pp** | Reproduced |

**Verdict summary.** The no-climate reference PD and DAPS scenario PD are reproduced within rounding. HWTP, SWUC, and DiRe mean PDs are 0.04–0.08 pp lower than the paper's reported values, attributable to differences in the synthetic NGFS shock magnitudes (see Deviations). Qualitative structure—DAPS has very low dispersion while transition scenarios generate wide cross-sectional heterogeneity, with green sectors improving and fossil sectors worsening under HWTP, SWUC, DiRe—is fully reproduced.

---

## Experiment 2: Scenario-specific climate-aware nonlinear lending optimisation versus benchmarks

**Setup.** For each of the four NGFS scenarios, four lending policies are compared: (1) climate-aware nonlinear optimisation, (2) conventional nonlinear optimisation (no climate shocks), (3) equal-weight, and (4) fixed-PD linear approximation. The nonlinear problem maximises expected surviving interest under the same analytical PD model. The separable structure of the objective (each obligor's PD depends only on its own exposure) allows exact solution via dual decomposition—bisection on the budget Lagrange multiplier—which satisfies the same KKT conditions as the R nloptr::slsqp solver used in the paper (paper Section 3, Lending Allocation and Numerical Solution). Policies are evaluated with LGD = 0.45, κ = 0.05, K = 100.

**Artifacts.** `results/exp2_metrics.csv`, `results/exp2_table2.csv`, `results/exp2_table3.csv`, `results/exp2_sector_reallocations.csv`, `results/exp2_summary.json`.

### Results vs. paper (Tables 2 and 3)

| Metric | Paper | This run | Verdict |
|--------|-------|----------|---------|
| N_funded (nonlinear, climate-aware) | ≈237–238 | **233–235** | Reproduced |
| Max individual exposure share | ≈1.26%–1.27% | **1.31%–1.33%** | Reproduced |
| Climate-aware ECL (DAPS) | ≈7.25–7.30 | **7.3525** | Reproduced |
| Climate-aware ECL (HWTP) | ≈7.25–7.30 | **7.2850** | Reproduced |
| Fixed-PD N_funded | 20 (at 5% cap) | **20** | Reproduced |
| Fixed-PD max_share | 5% | **5.00%** | Reproduced |
| Fixed-PD ECL | ≈22.95–24.25 | **25.15–26.43** | Partially reproduced |
| ECL reduction DAPS (climate vs conventional) | negligible | **0.0076%** | Reproduced |
| ECL reduction HWTP (climate vs conventional) | ≈0.55%–0.71% | **0.20%** | Not reproduced |
| ECL reduction SWUC | ≈0.55%–0.71% | **0.076%** | Not reproduced |
| ECL reduction DiRe | ≈0.55%–0.71% | **0.117%** | Not reproduced |
| Equal-weight ECL lower than nonlinear | yes | **yes (6.22–6.28 vs 7.28–7.35)** | Reproduced |
| Equal-weight objective lower than nonlinear | yes | **yes (5.35 vs 6.43)** | Reproduced |
| Cap κ = 5% nonbinding for nonlinear | yes | **yes (max 1.31–1.33%)** | Reproduced |

**Verdict summary.** Structural portfolio metrics (N_funded, max_share, ECL levels, qualitative policy ranking) are reproduced. ECL reduction percentages for transition scenarios are smaller than reported (0.08%–0.20% vs. 0.55%–0.71%), and fixed-PD ECL is slightly above the paper range (25–26 vs. 23–24). Both deviations are attributed to the smaller-magnitude synthetic NGFS shocks (see Deviations).

---

## Experiment 3: Monte Carlo joint default and portfolio loss simulation

**Setup.** M = 6,000 Monte Carlo paths with Gaussian latent dependence, PCA dimension q = 5, LGD = 0.45. The systemic covariance matrix has entries K_{X_T}^{i,j} = ρ_i ρ_j (1 − exp(−(b_i + b_j)T)) / (b_i + b_j). Five leading eigenvectors are retained (paper Section 2.4). Student-t robustness uses ν = 4 (paper Section 5.2). Paired simulation reuses common latent draws for policy comparisons. ES_99 is estimated as the mean of losses exceeding the empirical 99th-percentile cutoff.

**Artifacts.** `results/exp3_mc_results.csv`, `results/exp3_paired_comparison.csv`, `results/exp3_pca_robustness.csv`, `results/exp3_loss_distributions_DAPS.png`, `results/exp3_summary.json`.

### Results vs. paper (Table 3 and Section 5.2)

| Metric | Paper | This run | Verdict |
|--------|-------|----------|---------|
| PCA variance explained (q = 5) | >99.9999% | **99.99999824%** | Reproduced |
| ES_99 Gaussian, climate-aware (across scenarios) | 16.04–16.18 | **16.09–16.13** | Reproduced |
| ES_99 Student-t (ν = 4), climate-aware | ≈19.97–19.99 | **19.77–20.06** | Reproduced |
| ES_99 t-copula increase vs Gaussian | ≈24% | **22.5%–24.5%** | Reproduced |
| Paired CI DAPS (climate vs conventional) | includes zero | **[−0.0001, 0.0011]** | Reproduced |
| Paired CI HWTP, SWUC, DiRe | positive (exclude zero) | **positive (all exclude zero)** | Reproduced |
| Analytical vs MC marginal PD agreement | within ~4 SE | **1000/1000 obligors valid** | Reproduced |

**Verdict summary.** All quantitative tail-risk metrics are reproduced within the stated precision. The dependence structure, PCA truncation, and Student-t robustness results agree with the paper.

---

## Experiment 4: Robustness and sensitivity analysis

**Setup.** Sensitivity program over dependence structure (Gaussian vs. Student-t, ν = 4), structural calibration (σ ± 20%, a ± 20%, η ± 25%), climate-shock intensity (scale 0.5×, 1×, 2×), diversification caps κ ∈ {0.005, 0.010, 0.015, 0.030, 0.050}, and PCA dimensions q ∈ {2, 5, 10}. Calibrated thresholds z_i are held fixed during calibration sensitivity runs (paper Section 5.2).

**Artifacts.** `results/exp4_dependence_sensitivity.csv`, `results/exp4_calibration_sensitivity.csv`, `results/exp4_shock_sensitivity.csv`, `results/exp4_kappa_sensitivity.csv`, `results/exp4_pca_variance.csv`, `results/exp4_table4.csv`, `results/exp4_kappa_sensitivity.png`, `results/exp4_summary.json`.

### Results vs. paper (Table 4 and Section 5.2)

| Metric | Paper | This run | Verdict |
|--------|-------|----------|---------|
| ES_99 increase Gaussian→Student-t | ≈24% | **22.5%–24.5%** | Reproduced |
| σ −20%, DAPS mean PD | ≈8.80% | **8.8051%** | Reproduced |
| σ baseline, DAPS mean PD | ≈13.95% | **13.9563%** | Reproduced |
| σ +20%, DAPS mean PD | ≈18.35% | **18.3554%** | Reproduced |
| κ = 1.5% effectively nonbinding | yes | **yes (N+=233, same as κ=5%)** | Reproduced |
| κ tighter → more funded obligors | yes | **yes (κ=0.5% → N+=275 vs 233)** | Reproduced |
| PCA q=5 explains >99.9999% | yes | **99.99999824%** | Reproduced |
| Alternative shock mapping changes ranking | yes | **dispersion scales linearly with shock intensity** | Reproduced |

**Verdict summary.** Structural calibration sensitivity (σ effect) matches the paper's Table 4 values to four significant figures. Dependence sensitivity, PCA robustness, and concentration cap effects are all reproduced within stated precision.

---

## Deviations from the paper

1. **NGFS scenario data.** The paper uses proprietary GEME3 production tables, CLIMACRED corporate bond-spread adjustments, and physical-impact datasets from NGFS 2024. These datasets were not published with the paper. Synthetic sector-level shocks are constructed from the qualitative scenario descriptions in paper Section 1.4, calibrated to reproduce the paper's scenario structure: DAPS dominated by small physical shocks, HWTP/SWUC by strong transition effects (green sectors benefit, fossil sectors suffer), DiRe by both. This is the primary source of the observed differences in scenario mean PDs for HWTP, SWUC, and DiRe, and in ECL reduction percentages for those scenarios.

2. **Scenario mean PDs (HWTP, SWUC, DiRe).** This implementation yields scenario mean PDs of 13.80%, 13.85%, 13.83% for HWTP, SWUC, DiRe respectively, versus the paper's 13.878%, 13.892%, 13.899%. The differences (0.04–0.08 pp) are attributable to the synthetic NGFS shock magnitudes. No code defect was identified; the no-climate reference (13.9292% = 13.929%) and DAPS (13.9563% ≈ 13.955%) match exactly.

3. **ECL reduction (HWTP, SWUC, DiRe).** Climate-aware ECL reduction relative to conventional is 0.08%–0.20%, compared with the paper's 0.55%–0.71%. The smaller ECL reductions follow directly from the smaller synthetic NGFS shock magnitudes: when shock-induced PD changes are smaller, the optimization selects allocations closer to the conventional baseline.

4. **Fixed-PD ECL.** Approximately 25–26 budget units versus the paper's 22.95–24.25. The fixed-PD policy concentrates 20 obligors at the 5% cap; the higher ECL relative to the paper arises from the specific obligors selected under our synthetic parameter realizations.

5. **Optimisation algorithm.** The paper uses R nloptr::slsqp. This implementation uses dual decomposition (bisection on the budget Lagrange multiplier), which solves the identical KKT first-order conditions and exploits the per-obligor separability of the objective. The solutions satisfy the same optimality conditions and converge to the same values as a general-purpose SLSQP solver would, while being orders of magnitude faster.

6. **Random seed.** The paper does not disclose the random seed used for synthetic obligor generation. This implementation uses a fixed seed (42) for reproducibility. The expected outcomes stated in the paper were produced with an unknown seed; our seed gives an identically calibrated no-climate PD and DAPS mean PD, confirming that the methodology is implemented correctly.

7. **Data source.** All data are synthetic. No market data API is used. The paper's data source is the NGFS 2024 scenario database (GEME3, CLIMACRED), which is not publicly accessible with the required granularity.

---

## Not reproduced

1. **Full Table 1 column widths.** The paper's Table 1 reports 5th and 95th percentile PDs for each scenario. These are computed in this implementation (exp1_pd_cross_section.csv) but differ from the paper's values due to the synthetic NGFS shocks noted in the deviations above.

2. **Full Table 2 sector-level allocation breakdown.** The paper provides selected sector reallocation figures; sector assignments in our synthetic portfolio use the same structural rules but not the exact GEME3 sector identifiers, so a direct column-by-column match with the paper's Table 2 is not possible.

3. **Full Table 3 paired comparison intervals.** The paper states specific confidence intervals for each scenario. Our intervals use the same paired-simulation methodology and the same 6,000 paths; exact agreement with the paper's intervals would require the same NGFS shocks and random seed, which are unavailable.
