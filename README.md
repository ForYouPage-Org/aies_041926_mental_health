# Mental Health × AI Attitudes

Replicates the hierarchical regression + LIWC psycholinguistic analysis from [`FaccT26_Attachment_Fyorin.pdf`](FaccT26_Attachment_Fyorin.pdf) — but swaps attachment predictors (ECR-SF anxiety / avoidance) for mental-health predictors (PHQ-4 anxiety, PHQ-4 depression; PROMIS anxiety as robustness).

**RQ1.** Do anxiety and depression predict perceived ChatGPT experiences (emotional engagement, trust, self-efficacy, behavioral change, dependency concern) beyond demographics and usage frequency?
**RQ2.** Do they leave psycholinguistic traces (LIWC-22) in users' ChatGPT prompts?

## Repo layout

| Path | Contents |
|---|---|
| [analysis/](analysis/) | Active analysis scripts (see below) |
| [analysis/_archive/](analysis/_archive/) | Older / unused scripts (ECR-specific topic modeling, sampling, alternate vis) |
| [data/](data/) | Self-contained data snapshot (see *Data* below) |
| [figures/](figures/) | Publication figures |
| [paper/](paper/) | Scoping / diagnostics markdown (to be written) |
| [results/](results/) | Timestamped analysis runs |
| [anonymous-submission-latex-2026.tex](anonymous-submission-latex-2026.tex) | Draft manuscript (AAAI-2026 template; intro written, later sections stubbed) |
| [aaai2026.{sty,bib,bst}](aaai2026.sty) | LaTeX class + bibliography |
| [FaccT26_Attachment_Fyorin.pdf](FaccT26_Attachment_Fyorin.pdf) | Reference paper |

## Active analysis scripts

| Script | Purpose | Output |
|---|---|---|
| [analysis/phq4_ai_correlation_regression_analysis.py](analysis/phq4_ai_correlation_regression_analysis.py) | RQ1 — hierarchical regression of PHQ-4 anxiety + depression onto 5 AI-attitude composites. `--anxiety-source promis` for robustness. | `results/<ts>_phq4_ai_controlled_regression/` |
| [analysis/phq4_liwc_analysis.py](analysis/phq4_liwc_analysis.py) | RQ2 — PHQ-4 anxiety + depression × 46 LIWC-22 markers, per-participant averaged. | `results/<ts>_phq4_liwc_analysis/` |
| [analysis/liwc22_analysis.py](analysis/liwc22_analysis.py) | Original ECR × LIWC (kept for RQ3 side-by-side). | `results/<ts>_LIWC22_analysis/` |
| [analysis/ecr_ai_correlation_regression_analysis.py](analysis/ecr_ai_correlation_regression_analysis.py) | Original ECR regression (kept for RQ3 side-by-side). | `results/<ts>_ecr_ai_controlled_regression/` |
| [analysis/nrc_vad_analysis.py](analysis/nrc_vad_analysis.py) | Valence/arousal/dominance robustness. | `results/<ts>_VAD_analysis/` |
| [analysis/fig_regression_forest_plot.py](analysis/fig_regression_forest_plot.py), [analysis/vis_hr_2.py](analysis/vis_hr_2.py), [analysis/grouped_correlation_forest.py](analysis/grouped_correlation_forest.py) | Figure generation. | `figures/` |

## Data (self-contained in [data/](data/))

All files copied from `/Users/marxw/Research/projects/sirius/` so this repo runs standalone.

| File | Source | Used by |
|---|---|---|
| [data/survey.csv](data/survey.csv) | local (pre-existing) | regression + LIWC (raw PHQ-4/PROMIS items + ChatGPT frequency) |
| [data/survey_scored.csv](data/survey_scored.csv) | `sirius/data/` | regression + LIWC (AI attitudes AI_Q1–Q15, Age) |
| [data/ecr_scores_validated_clean.csv](data/ecr_scores_validated_clean.csv) | local (pre-existing) | ECR comparison analysis |
| [data/chat_participants_list.csv](data/chat_participants_list.csv) | `sirius/Archive/analysis_102025/` | LIWC participant matching |
| [data/liwc22_user_prompts.csv](data/liwc22_user_prompts.csv) | `results/251229T1212_NRC_VAD_analysis/` (already in this repo) | LIWC analyses |
| [data/filtered_user_prompts_lt20words.csv](data/filtered_user_prompts_lt20words.csv) | same as above | raw prompt source |
| [data/demographics/](data/demographics/) | `sirius/data/demographics/` | age/gender join (7 Prolific exports) |

## Quick run

```bash
python analysis/phq4_ai_correlation_regression_analysis.py    # RQ1
python analysis/phq4_liwc_analysis.py                          # RQ2
python analysis/phq4_ai_correlation_regression_analysis.py --anxiety-source promis   # robustness
```

Initial results (N=158 for RQ1, N=104 for RQ2): PHQ-4 depression is a stronger predictor than PHQ-4 anxiety on both the experience composites (emotional engagement r=.27, p<.01) and LIWC markers (7/46 features FDR-sig: future focus, we-words, negative emotion, affiliation, social references, prosocial, they-words).

## Outcome composites (from paper §3.2.2)

| Construct | Items | α (original) |
|---|---|---|
| Emotional Engagement | Q4, Q5, Q6 | .88 |
| Self-Efficacy | Q7, Q8, Q9 | .71 |
| Behavioral Change | Q10, Q11, Q12 | .71 |
| Trust | Q1 | single item |
| Dependency Concern | Q13 | single item |

## Reference structure

Layout mirrors [`/Users/marxw/Research/papers/aies_041726/`](../aies_041726).
