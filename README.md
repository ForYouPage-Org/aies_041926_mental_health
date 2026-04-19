# Mental Health × AI Attitudes

Replicates the hierarchical regression analysis from [`FaccT26_Attachment_Fyorin.pdf`](FaccT26_Attachment_Fyorin.pdf) — but swaps the attachment predictors (ECR-SF anxiety / avoidance) for mental health predictors (PHQ-4 anxiety, PHQ-4 depression; PROMIS anxiety as robustness check).

**RQ:** Do anxiety and depression scores predict users' perceived experiences with ChatGPT (emotional engagement, trust, self-efficacy, behavioral change, dependency concern), beyond demographics and usage frequency?

## Repo layout

| Path | Contents |
|---|---|
| [analysis/](analysis/) | Python analysis scripts (regression, LIWC, VAD, topic modeling, figures) |
| [data/](data/) | Local copies: [survey.csv](data/survey.csv), [ecr_scores_validated_clean.csv](data/ecr_scores_validated_clean.csv), Prolific demographics |
| [figures/](figures/) | Publication figures (forest plots, distributions) |
| [paper/](paper/) | Scoping / diagnostics markdown (currently empty — to be added) |
| [results/](results/) | Timestamped analysis runs (CSV + diagnostics) |
| [Archive/](Archive/) | Prior ECR-era analysis reports and notes |
| [FaccT26_Attachment_Fyorin.pdf](FaccT26_Attachment_Fyorin.pdf) | Reference paper |

## Data source

Canonical data lives at `/Users/marxw/Research/projects/sirius/data/` (survey, conversations, demographics, scored composites). The [data/](data/) folder here holds a local snapshot for reproducibility. Scripts currently resolve `DATA_DIR = BASE_DIR.parent / "data"` — update this to point at the sirius location or to local `data/`.

## Primary analysis script

[analysis/phq4_ai_correlation_regression_analysis.py](analysis/phq4_ai_correlation_regression_analysis.py) — mirrors the attachment hierarchical regression using PHQ-4 anxiety + depression (or PROMIS anxiety via `--anxiety-source promis`). Outputs to `results/<timestamp>_phq4_ai_controlled_regression/`.

```bash
python analysis/phq4_ai_correlation_regression_analysis.py                    # PHQ-4 anxiety + depression
python analysis/phq4_ai_correlation_regression_analysis.py --anxiety-source promis
```

## Outcome composites (5)

| Construct | Items | Source (paper §3.2.2) |
|---|---|---|
| Emotional Engagement | Q4, Q5, Q6 | composite (α ≈ .88) |
| Self-Efficacy | Q7, Q8, Q9 | composite (α ≈ .71) |
| Behavioral Change | Q10, Q11, Q12 | composite (α ≈ .71) |
| Trust | Q1 | single item |
| Dependency Concern | Q13 | single item |

## Reference structure

Layout mirrors [`/Users/marxw/Research/papers/aies_041726/`](../aies_041726) (`analysis/`, `data/`, `figures/`, `paper/`, `results/`).
