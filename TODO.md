# TODO

## Primary: mental-health analogue of [FaccT26_Attachment_Fyorin.pdf](FaccT26_Attachment_Fyorin.pdf)

Replace ECR anxiety / avoidance with PHQ-4 anxiety + PHQ-4 depression (PROMIS anxiety as robustness). Outcomes and covariates unchanged from paper §3.3.

- [ ] **Wire data paths to `/Users/marxw/Research/projects/sirius/data/`.** [analysis/phq4_ai_correlation_regression_analysis.py:49](analysis/phq4_ai_correlation_regression_analysis.py#L49) currently uses `DATA_DIR = BASE_DIR.parent / "data"` — confirm or repoint.
- [ ] **Run PHQ-4 regression.** `python analysis/phq4_ai_correlation_regression_analysis.py` → writes `results/<ts>_phq4_ai_controlled_regression/`.
- [ ] **Run PROMIS robustness.** `--anxiety-source promis`.
- [ ] **Validate PHQ-4 reliability.** Cronbach α for the 2-item anxiety and 2-item depression subscales; report in a diagnostics note under [paper/](paper/).
- [ ] **Check multicollinearity.** Anxiety × depression correlation is typically r > .6; inspect [07_vif_multicollinearity.csv](results/260401T1145_phq4_ai_controlled_regression/07_vif_multicollinearity.csv) — if VIF > 5, refit with a combined distress composite or report residualized predictors.

## Figures (parallel to paper Fig. 1)

- [ ] **Forest plot of standardized β (anxiety vs. depression × 5 outcomes).** Adapt [analysis/fig_regression_forest_plot.py](analysis/fig_regression_forest_plot.py) and [analysis/vis_hr_2.py](analysis/vis_hr_2.py) — swap `Beta_Avoidance` → `Beta_Depression`, update colors/labels. Save to [figures/](figures/).
- [ ] **Distribution panel.** PHQ-4 anxiety, PHQ-4 depression, PROMIS anxiety histograms + descriptives.

## Scoping / writing (goes in [paper/](paper/))

- [ ] **`paper/contribution_scoping.md`** — framing: what does switching the predictor from trait-attachment to state-mental-health tell us? Is "anxiously attached" in the original paper picking up depression/anxiety, or something distinct?
- [ ] **`paper/phq4_diagnostics.md`** — PHQ-4 score distributions, clinical cutoff breakdown (0–2 none, 3–5 mild, 6–8 moderate, 9–12 severe), sample comparison against general-pop norms.
- [ ] **Head-to-head vs. ECR.** Partial correlations: does PHQ-4 anxiety predict AI outcomes after controlling for ECR anxiety? (Or vice versa.)

## Data hygiene

- [ ] **Confirm `survey_scored.csv` mapping** in [`/Users/marxw/Research/projects/sirius/data/`](../../projects/sirius/data/) — which version is canonical (`survey_scored.csv`, `survey_scored_full.csv`, `survey_scored_complete.csv`)?
- [ ] **Demographics join.** Scripts glob `demographics/prolific_export_*.csv`; verify those files exist at the final `DATA_DIR`.

## Note on terminology

Original prompt said "hierarchical clustering" — the paper uses hierarchical *regression* (Step 1: age, gender, usage frequency; Step 2: add psychological predictors). Proceeding with hierarchical regression. If clustering (e.g., k-means on anxiety × depression to derive subgroups, then test group × AI-attitude) is actually wanted, add as separate task.

## Out of scope (this iteration)

- LIWC / NRC-VAD linguistic analysis (RQ2 in paper). Scripts exist in [analysis/](analysis/) — run later if the regression yields signal worth contextualizing.
- Qualitative content coding of top-quartile conversations (paper §3.3 content analysis).
