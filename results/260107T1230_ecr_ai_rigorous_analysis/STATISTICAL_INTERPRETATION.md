# ECR × AI Attitudes: Statistical Results Interpretation

**Date:** 2026-01-07  
**Sample:** N = 168 complete cases

---

## Understanding the Statistics

### Key Statistical Terms

| Statistic | What It Measures                    | How to Interpret                                            |
| --------- | ----------------------------------- | ----------------------------------------------------------- |
| **r**     | Pearson correlation coefficient     | Strength/direction of linear relationship (-1 to +1)        |
| **R²**    | Coefficient of determination        | Proportion of variance explained (0 to 1)                   |
| **F**     | F-statistic                         | Whether the overall regression model is significant         |
| **t**     | t-statistic                         | Whether an individual predictor is significant              |
| **p**     | p-value                             | Probability of observing results if null hypothesis is true |
| **β**     | Standardized regression coefficient | Effect size of predictor in standard deviation units        |

---

## F-Statistic Explained

### What is F?

The **F-statistic** tests whether the regression model as a whole explains a significant amount of variance in the outcome variable. It compares:

- **Model variance** (how much the predictors explain)
- **Error variance** (unexplained variation)

**Formula:**

```
F = (SS_regression / df_regression) / (SS_residual / df_residual)
```

### Interpreting F

| F value | df₁, df₂ | p-value | Interpretation                                   |
| ------- | -------- | ------- | ------------------------------------------------ |
| High    | —        | < .05   | Model significantly predicts the outcome         |
| Low     | —        | > .05   | Model does not significantly predict the outcome |

### Our Results (Significant F-tests)

| AI Attitude                | F(2,165) | p      | Interpretation                 |
| -------------------------- | -------- | ------ | ------------------------------ |
| **Easier to Share**        | 22.10    | < .001 | Very strong model significance |
| **Emotional Relief**       | 10.39    | < .001 | Strong model significance      |
| **Emotionally Understood** | 9.56     | < .001 | Strong model significance      |
| **Trust**                  | 9.29     | < .001 | Strong model significance      |
| **Communication Changed**  | 8.45     | < .001 | Strong model significance      |
| More Efficient             | 5.62     | .004   | Moderate model significance    |
| Less Confident Without     | 4.85     | .009   | Moderate model significance    |
| Fact-Check                 | 4.45     | .013   | Moderate model significance    |
| Learning Different         | 3.93     | .022   | Marginally significant         |
| More Capable               | 3.74     | .026   | Marginally significant         |
| Problem Solving            | 3.22     | .042   | Marginally significant         |

**Non-significant models:** Confident Implement, Worry Relying, Modify Writing, Prefer Attempt First (F < 3.0, p > .05)

---

## t-Statistic Explained

### What is t?

The **t-statistic** tests whether an individual predictor significantly contributes to the model, controlling for other predictors. It measures how many standard errors the coefficient is away from zero.

**Formula:**

```
t = β / SE(β)
```

### Interpreting t

|           | t                    | value | Interpretation |
| --------- | -------------------- | ----- | -------------- |
| > 3.0     | Very strong evidence |
| 2.0 - 3.0 | Moderate evidence    |
| < 2.0     | Weak or no evidence  |

**Critical value:** For df ≈ 165, t > 1.97 is significant at p < .05

### Our Results: Anxiety Predictor

| AI Attitude                | β     | t    | p      | Significant? |
| -------------------------- | ----- | ---- | ------ | ------------ |
| **Easier to Share**        | 0.531 | 5.01 | < .001 | ✓✓✓          |
| **Emotional Relief**       | 0.435 | 4.19 | < .001 | ✓✓✓          |
| **Emotionally Understood** | 0.424 | 4.31 | < .001 | ✓✓✓          |
| **Trust**                  | 0.269 | 4.27 | < .001 | ✓✓✓          |
| **Communication Changed**  | 0.369 | 3.69 | < .001 | ✓✓✓          |
| Less Confident Without     | 0.305 | 3.11 | .002   | ✓✓           |
| Fact-Check                 | 0.250 | 2.93 | .004   | ✓✓           |
| Learning Different         | 0.188 | 2.73 | .007   | ✓✓           |
| Problem Solving            | 0.200 | 2.45 | .015   | ✓            |
| More Capable               | 0.132 | 2.18 | .031   | ✓            |
| Confident Implement        | 0.208 | 2.02 | .045   | ✓            |

### Our Results: Avoidance Predictor

| AI Attitude        | β      | t     | p    | Significant? |
| ------------------ | ------ | ----- | ---- | ------------ |
| **More Efficient** | -0.207 | -3.31 | .001 | ✓✓✓          |
| More Capable       | -0.144 | -2.39 | .018 | ✓            |
| Trust              | -0.143 | -2.27 | .024 | ✓            |
| Easier to Share    | 0.209  | 1.97  | .050 | marginal     |

**Pattern:** Anxiety is a significant predictor for 11/15 attitudes; Avoidance for only 3/15

---

## p-Value Explained

### What is p?

The **p-value** is the probability of observing a test statistic as extreme as (or more extreme than) the one calculated, assuming the null hypothesis is true.

### Interpreting p

| p-value    | Notation | Interpretation                    |
| ---------- | -------- | --------------------------------- |
| < .001     | \*\*\*   | Very strong evidence against null |
| .001 - .01 | \*\*     | Strong evidence                   |
| .01 - .05  | \*       | Moderate evidence                 |
| > .05      | ns       | Insufficient evidence             |

### Multiple Comparison Correction

We applied **Benjamini-Hochberg FDR correction** because we ran 30 tests (2 ECR dimensions × 15 AI attitudes).

**Why?** Running multiple tests inflates Type I error (false positives). FDR correction controls the expected proportion of false discoveries.

**Result:** After FDR correction:

- Anxiety: 9/15 remain significant
- Avoidance: 2/15 remain significant

---

## R² (Variance Explained)

### Interpreting R²

| R²        | Effect Size | Meaning                  |
| --------- | ----------- | ------------------------ |
| < .01     | Negligible  | < 1% variance explained  |
| .01 - .09 | Small       | 1-9% variance explained  |
| .09 - .25 | Medium      | 9-25% variance explained |
| > .25     | Large       | > 25% variance explained |

### Our Results

| AI Attitude            | R²   | Effect | Meaning                                   |
| ---------------------- | ---- | ------ | ----------------------------------------- |
| **Easier to Share**    | .211 | Medium | 21.1% of variance explained by attachment |
| Emotional Relief       | .112 | Medium | 11.2%                                     |
| Emotionally Understood | .104 | Medium | 10.4%                                     |
| Trust                  | .101 | Medium | 10.1%                                     |
| Communication Changed  | .093 | Medium | 9.3%                                      |
| More Efficient         | .064 | Small  | 6.4%                                      |
| Less Confident Without | .056 | Small  | 5.6%                                      |
| Fact-Check             | .051 | Small  | 5.1%                                      |

---

## Summary of Key Findings

### Correlation Analysis

**ECR Anxiety correlates significantly with 9/15 AI attitudes:**

| Rank | AI Attitude            | r         | 95% CI     | Effect |
| ---- | ---------------------- | --------- | ---------- | ------ |
| 1    | Easier to Share        | .44\*\*\* | [.31, .55] | Medium |
| 2    | Emotional Relief       | .33\*\*\* | [.19, .46] | Medium |
| 3    | Emotionally Understood | .31\*\*\* | [.17, .44] | Medium |
| 4    | Communication Changed  | .30\*\*\* | [.16, .44] | Medium |
| 5    | Trust                  | .27\*\*   | [.12, .41] | Small  |
| 6    | Less Confident Without | .22\*\*   | [.07, .36] | Small  |
| 7    | Fact-Check             | .19\*     | [.04, .33] | Small  |
| 8    | Confident Implement    | .18\*     | [.03, .33] | Small  |
| 9    | Learning Different     | .17\*     | [.02, .31] | Small  |

**ECR Avoidance correlates significantly with 2/15 AI attitudes:**

| Rank | AI Attitude     | r       | 95% CI       | Effect           |
| ---- | --------------- | ------- | ------------ | ---------------- |
| 1    | Easier to Share | .30\*\* | [.16, .43]   | Medium           |
| 2    | More Efficient  | -.21\*  | [-.35, -.06] | Small (negative) |

### Multiple Regression

**Best model: "Easier to Share"**

```
AI_Q6 = β₀ + 0.531(Anxiety_z) + 0.209(Avoidance_z)

F(2,165) = 22.10, p < .001
R² = .211 (21.1% variance explained)

Anxiety: β = 0.53, t = 5.01, p < .001 ✓✓✓
Avoidance: β = 0.21, t = 1.97, p = .050 (marginal)
```

**Interpretation:** A 1 SD increase in Attachment Anxiety is associated with a 0.53 SD increase in "Easier to share personal struggles with AI" preference, controlling for Avoidance.

### Unique Contributions (ΔR²)

| AI Attitude            | Total R² | ΔR² Anxiety | ΔR² Avoidance | Dominant      |
| ---------------------- | -------- | ----------- | ------------- | ------------- |
| Easier to Share        | .211     | .120        | .019          | **Anxiety**   |
| Emotional Relief       | .112     | .094        | .000          | **Anxiety**   |
| Emotionally Understood | .104     | .101        | .006          | **Anxiety**   |
| Trust                  | .101     | .099        | .028          | **Anxiety**   |
| More Efficient         | .064     | .018        | .062          | **Avoidance** |

**Conclusion:** Attachment Anxiety is the dominant predictor for most AI attitudes. Avoidance only dominates for efficiency-related attitudes.

---

## Practical Interpretation

### What These Results Mean

1. **Anxiously attached individuals** (high fear of rejection, need for reassurance):

   - Find it **easier to share personal struggles with AI** than with humans
   - Experience more **emotional relief** from AI interactions
   - Feel more **emotionally understood** by AI
   - These are medium-sized effects (r ≈ .30-.44)

2. **Avoidantly attached individuals** (discomfort with closeness):

   - Also find AI disclosure easier (but to lesser degree)
   - Rate AI as **less efficient** (negative relationship)
   - Otherwise show few significant associations

3. **Combined model power:**
   - Attachment dimensions explain up to **21% of variance** in AI attitudes
   - Anxiety contributes **3-6× more unique variance** than Avoidance

### Limitations

- Cross-sectional design (cannot infer causality)
- Self-selected AI user sample
- Small-to-medium effect sizes
- Multiple comparisons (Type I error risk despite correction)

---

## Files Reference

| File                            | Contents                                    |
| ------------------------------- | ------------------------------------------- |
| `01_descriptive_statistics.csv` | M, SD, skewness, kurtosis, normality tests  |
| `02_anxiety_correlations.csv`   | r, p, 95% CI, FDR-corrected p for Anxiety   |
| `03_avoidance_correlations.csv` | r, p, 95% CI, FDR-corrected p for Avoidance |
| `04_multiple_regression.csv`    | R², F, t, β, p for all regression models    |
| `05_unique_contributions.csv`   | ΔR² for each predictor                      |

---

## References

Cohen, J. (1988). _Statistical power analysis for the behavioral sciences_ (2nd ed.). Lawrence Erlbaum.

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. _Journal of the Royal Statistical Society: Series B_, 57(1), 289-300.

Wei, M., Russell, D. W., Mallinckrodt, B., & Vogel, D. L. (2007). The ECR-Short Form. _Journal of Personality Assessment_, 88(2), 187-204.
