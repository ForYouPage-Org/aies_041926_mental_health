# ECR Attachment × AI Attitudes: Rigorous Statistical Analysis

**Generated:** 2026-01-07 09:14:51  
**Analysis Type:** Correlation and Multiple Regression  
**Sample Size:** N = 168

---

## 1. Research Question & Theoretical Rationale

### Research Question
How do attachment dimensions (Anxiety, Avoidance) relate to attitudes toward AI?

### Theoretical Framework
Based on attachment theory (Bowlby, 1969; Hazan & Shaver, 1987), we hypothesize that:

1. **Attachment Anxiety** (fear of rejection, need for reassurance) will positively correlate with emotional AI attitudes because AI provides non-judgmental, always-available support.

2. **Attachment Avoidance** (discomfort with closeness) may correlate with preference for AI over human interaction because AI allows control and distance.

### Measures
- **ECR-SF** (Wei et al., 2007): 12-item scale measuring Anxiety (6 items) and Avoidance (6 items) on 7-point Likert scale
- **AI Attitudes**: 15 items covering trust, emotion, capability, and behavior on 5-point Likert scale

---

## 2. Methodology

### Step 1: Data Preparation
- Merged ECR scores with AI attitude survey data
- Converted Likert responses to numeric (1-5 scale)
- Filtered to complete cases: N = 168

### Step 2: Descriptive Statistics
- Computed mean, SD, skewness, kurtosis for all variables
- Tested normality using Shapiro-Wilk test
- Assessed appropriateness of parametric tests

### Step 3: Correlation Analysis
- Computed Pearson product-moment correlations
- Calculated 95% confidence intervals via Fisher's z-transformation
- Applied Benjamini-Hochberg FDR correction for 30 comparisons (2 × 15)
- Interpreted effect sizes using Cohen's conventions (small: r = .10, medium: r = .30, large: r = .50)

### Step 4: Multiple Regression Analysis
- Standardized predictors (z-scores) for interpretable coefficients
- Fitted OLS regression: AI_Attitude ~ β₀ + β₁(Anxiety_z) + β₂(Avoidance_z)
- Tested model significance via F-test
- Assessed individual predictor significance via t-tests
- Checked residual normality via Shapiro-Wilk test

### Step 5: Unique Contribution Analysis
- Computed ΔR² for each predictor controlling for the other
- Determined relative importance of Anxiety vs. Avoidance
- Identified shared vs. unique variance

---

## 3. Results

### 3.1 Sample Characteristics

| Variable | N | Mean | SD | Range |
|----------|---|------|-----|-------|
| ECR Anxiety | 168 | 4.055 | 1.104 | 1.17 - 7.00 |
| ECR Avoidance | 168 | 3.190 | 0.999 | 1.00 - 5.50 |

### 3.2 Correlation Results

#### ECR Anxiety × AI Attitudes

| AI Attitude | r | 95% CI | p | FDR p | Sig | Effect |
|-------------|---|--------|---|-------|-----|--------|
| Easier to Share | 0.439 | [0.308, 0.554] | 0.0000 | 0.0000 | *** | medium |
| Emotional Relief | 0.334 | [0.193, 0.462] | 0.0000 | 0.0001 | *** | medium |
| Emotionally Understood | 0.312 | [0.169, 0.443] | 0.0000 | 0.0002 | *** | medium |
| Communication Changed | 0.305 | [0.161, 0.436] | 0.0001 | 0.0002 | *** | medium |
| Trust | 0.270 | [0.124, 0.405] | 0.0004 | 0.0012 | ** | small |
| Less Confident Without | 0.221 | [0.072, 0.360] | 0.0040 | 0.0099 | ** | small |
| Fact-Check | 0.187 | [0.036, 0.329] | 0.0155 | 0.0331 | * | small |
| Confident Implement | 0.184 | [0.033, 0.326] | 0.0171 | 0.0320 | * | small |
| Learning Different | 0.170 | [0.020, 0.314] | 0.0272 | 0.0454 | * | small |
| Problem Solving | 0.151 | [-0.000, 0.296] | 0.0508 | 0.0762 | ns | small |
| Worry Relying | 0.145 | [-0.007, 0.290] | 0.0609 | 0.0831 | ns | small |
| Modify Writing | 0.127 | [-0.025, 0.273] | 0.1009 | 0.1261 | ns | small |
| More Capable | 0.101 | [-0.051, 0.249] | 0.1921 | 0.2216 | ns | small |
| More Efficient | 0.038 | [-0.115, 0.188] | 0.6291 | 0.6740 | ns | negligible |
| Prefer Attempt First | 0.020 | [-0.132, 0.171] | 0.8002 | 0.8002 | ns | negligible |

**Summary:** 9/15 significant after FDR correction

#### ECR Avoidance × AI Attitudes

| AI Attitude | r | 95% CI | p | FDR p | Sig | Effect |
|-------------|---|--------|---|-------|-----|--------|
| Easier to Share | 0.302 | [0.158, 0.434] | 0.0001 | 0.0010 | ** | medium |
| More Efficient | -0.213 | [-0.353, -0.064] | 0.0055 | 0.0416 | * | small |
| Communication Changed | 0.134 | [-0.018, 0.279] | 0.0842 | 0.4208 | ns | small |
| Emotional Relief | 0.132 | [-0.020, 0.278] | 0.0876 | 0.3284 | ns | small |
| More Capable | -0.126 | [-0.272, 0.026] | 0.1045 | 0.3135 | ns | small |
| Confident Implement | 0.106 | [-0.046, 0.253] | 0.1724 | 0.4311 | ns | small |
| Prefer Attempt First | -0.083 | [-0.232, 0.069] | 0.2827 | 0.6057 | ns | negligible |
| Modify Writing | 0.076 | [-0.076, 0.225] | 0.3280 | 0.6149 | ns | negligible |
| Worry Relying | 0.054 | [-0.098, 0.204] | 0.4829 | 0.8049 | ns | negligible |
| Emotionally Understood | 0.053 | [-0.100, 0.202] | 0.4979 | 0.7468 | ns | negligible |
| Problem Solving | -0.050 | [-0.200, 0.102] | 0.5172 | 0.7053 | ns | negligible |
| Learning Different | -0.048 | [-0.198, 0.104] | 0.5348 | 0.6685 | ns | negligible |
| Trust | -0.044 | [-0.194, 0.108] | 0.5712 | 0.6591 | ns | negligible |
| Fact-Check | -0.042 | [-0.192, 0.111] | 0.5931 | 0.6355 | ns | negligible |
| Less Confident Without | 0.015 | [-0.137, 0.166] | 0.8477 | 0.8477 | ns | negligible |

**Summary:** 2/15 significant after FDR correction

### 3.3 Multiple Regression Results

| AI Attitude | R² | Adj R² | F | p(F) | β(Anxiety) | p(Anx) | β(Avoidance) | p(Avo) |
|-------------|-----|--------|---|------|------------|--------|--------------|--------|
| Easier to Share | 0.211 | 0.202 | 22.10 | 0.0000 | 0.531 | 0.0000 | 0.209 | 0.0502 |
| Emotional Relief | 0.112 | 0.101 | 10.39 | 0.0001 | 0.435 | 0.0000 | -0.005 | 0.9641 |
| Emotionally Understood | 0.104 | 0.093 | 9.56 | 0.0001 | 0.424 | 0.0000 | -0.107 | 0.2769 |
| Trust | 0.101 | 0.090 | 9.29 | 0.0002 | 0.269 | 0.0000 | -0.143 | 0.0244 |
| Communication Changed | 0.093 | 0.082 | 8.45 | 0.0003 | 0.369 | 0.0003 | 0.015 | 0.8772 |
| More Efficient | 0.064 | 0.052 | 5.62 | 0.0044 | 0.112 | 0.0743 | -0.207 | 0.0011 |
| Less Confident Without | 0.056 | 0.044 | 4.85 | 0.0090 | 0.305 | 0.0022 | -0.106 | 0.2831 |
| Fact-Check | 0.051 | 0.040 | 4.45 | 0.0131 | 0.250 | 0.0038 | -0.144 | 0.0936 |
| Learning Different | 0.045 | 0.034 | 3.93 | 0.0216 | 0.188 | 0.0070 | -0.116 | 0.0941 |
| More Capable | 0.043 | 0.032 | 3.74 | 0.0257 | 0.132 | 0.0305 | -0.144 | 0.0179 |
| Problem Solving | 0.038 | 0.026 | 3.22 | 0.0423 | 0.200 | 0.0152 | -0.130 | 0.1128 |
| Confident Implement | 0.035 | 0.023 | 2.99 | 0.0531 | 0.208 | 0.0454 | 0.046 | 0.6538 |
| Worry Relying | 0.021 | 0.009 | 1.77 | 0.1734 | 0.180 | 0.0830 | -0.006 | 0.9536 |
| Modify Writing | 0.017 | 0.005 | 1.41 | 0.2461 | 0.120 | 0.1744 | 0.031 | 0.7283 |
| Prefer Attempt First | 0.010 | -0.002 | 0.86 | 0.4236 | 0.069 | 0.4521 | -0.118 | 0.1991 |

**Summary:**
- Significant models (F-test): 11/15
- Average R²: 0.067
- Maximum R²: 0.211 (Easier to Share)

### 3.4 Unique Contribution Analysis

| AI Attitude | R²(Full) | ΔR²(Anxiety) | ΔR²(Avoidance) | Dominant |
|-------------|----------|--------------|----------------|----------|
| Easier to Share | 0.211 | 0.120 | 0.019 | Anxiety |
| Emotional Relief | 0.112 | 0.094 | 0.000 | Anxiety |
| Emotionally Understood | 0.104 | 0.101 | 0.006 | Anxiety |
| Trust | 0.101 | 0.099 | 0.028 | Anxiety |
| Communication Changed | 0.093 | 0.075 | 0.000 | Anxiety |
| More Efficient | 0.064 | 0.018 | 0.062 | Avoidance |
| Less Confident Without | 0.056 | 0.055 | 0.007 | Anxiety |
| Fact-Check | 0.051 | 0.049 | 0.016 | Anxiety |
| Learning Different | 0.045 | 0.043 | 0.016 | Anxiety |
| More Capable | 0.043 | 0.028 | 0.033 | Avoidance |
| Problem Solving | 0.038 | 0.035 | 0.015 | Anxiety |
| Confident Implement | 0.035 | 0.024 | 0.001 | Anxiety |
| Worry Relying | 0.021 | 0.018 | 0.000 | Anxiety |
| Modify Writing | 0.017 | 0.011 | 0.001 | Anxiety |
| Prefer Attempt First | 0.010 | 0.003 | 0.010 | Avoidance |

---

## 4. Key Findings

### 4.1 Attachment Anxiety Effects
- **Strongest associations:** Easier to Share (r = 0.439), 
  Emotional Relief (r = 0.334),
  Emotionally Understood (r = 0.312)
- **Interpretation:** Individuals with higher attachment anxiety report stronger emotional connection to AI, 
  consistent with the hyperactivation strategy where AI serves as a non-threatening source of support.

### 4.2 Attachment Avoidance Effects
- **Limited associations:** Fewer significant correlations than Anxiety
- **Interpretation:** Avoidant individuals may be less emotionally engaged with AI overall, 
  consistent with deactivation strategy that minimizes emotional investment.

### 4.3 Unique vs. Shared Variance
- **Anxiety dominant:** Contributes more unique variance to most AI attitudes
- **Avoidance secondary:** Adds minimal unique variance beyond Anxiety
- **Combined model:** Explains up to 21.1% of variance in AI attitudes

---

## 5. Statistical Notes

### 5.1 Multiple Comparison Correction
- Applied Benjamini-Hochberg FDR correction (α = .05)
- Corrected for 30 comparisons (2 ECR dimensions × 15 AI attitudes)
- Both uncorrected and FDR-corrected p-values reported

### 5.2 Effect Size Interpretation
Using Cohen's (1988) conventions:
- Small: r = .10, R² = .01
- Medium: r = .30, R² = .09
- Large: r = .50, R² = .25

### 5.3 Assumptions
- Normality: Tested via Shapiro-Wilk; some deviations noted but Pearson correlation is robust to mild violations
- Linearity: Assumed based on theory; should verify with scatterplots
- Homoscedasticity: Assumed; residual plots recommended for publication

---

## 6. Limitations

1. **Cross-sectional design:** Cannot establish causality
2. **Self-report measures:** Subject to social desirability and response biases
3. **Self-selected sample:** AI users may differ from general population
4. **Multiple comparisons:** Risk of Type I error despite FDR correction
5. **Effect sizes:** Small to medium; practical significance should be considered

---

## 7. Files Generated

| File | Description |
|------|-------------|
| `01_descriptive_statistics.csv` | Means, SDs, normality tests |
| `02_anxiety_correlations.csv` | ECR Anxiety × AI attitudes correlations |
| `03_avoidance_correlations.csv` | ECR Avoidance × AI attitudes correlations |
| `04_multiple_regression.csv` | Regression models with both predictors |
| `05_unique_contributions.csv` | ΔR² analysis for each predictor |
| `ANALYSIS_DOCUMENTATION.md` | This comprehensive report |
| `analysis.log` | Processing log |

---

## 8. References

Bartholomew, K., & Horowitz, L. M. (1991). Attachment styles among young adults: A test of a four-category model. _Journal of Personality and Social Psychology, 61_(2), 226-244.

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. _Journal of the Royal Statistical Society: Series B_, 57(1), 289-300.

Cohen, J. (1988). _Statistical power analysis for the behavioral sciences_ (2nd ed.). Lawrence Erlbaum.

Hayes, A. F. (2022). _Introduction to mediation, moderation, and conditional process analysis_ (3rd ed.). Guilford Press.

Hazan, C., & Shaver, P. (1987). Romantic love conceptualized as an attachment process. _Journal of Personality and Social Psychology, 52_(3), 511-524.

Wei, M., Russell, D. W., Mallinckrodt, B., & Vogel, D. L. (2007). The Experiences in Close Relationships Scale (ECR)-Short Form. _Journal of Personality Assessment, 88_(2), 187-204.

---

**Analysis completed:** 2026-01-07 09:14:51  
**Institution:** University of Washington
