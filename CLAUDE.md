# CLAUDE.md — Onboarding for Claude Code on a fresh clone

This file is auto-loaded by Claude Code when the repo is opened. It tells you (the assistant) what this project is, how to bootstrap the environment, and what work is wanted from you. Read it end-to-end before answering the user.

## Project at a glance

AAAI-2026 submission (depression-primary framing). Replicates the hierarchical-regression + LIWC psycholinguistic design of [FaccT26_Attachment_Fyorin.pdf](FaccT26_Attachment_Fyorin.pdf), but swaps trait attachment (ECR-SF) for state mental health (PHQ-4 anxiety, PHQ-4 depression; PROMIS anxiety as robustness).

- **RQ1.** Do PHQ-4 anxiety/depression predict 5 ChatGPT-attitude composites beyond demographics + usage frequency? — [analysis/phq4_categorical_rq1.py](analysis/phq4_categorical_rq1.py)
- **RQ2.** Do they leave traces in users' actual ChatGPT prompts (LIWC-22)? — [analysis/phq4_categorical_rq2_liwc.py](analysis/phq4_categorical_rq2_liwc.py)
- **RQ3.** Can a local LLM (Ollama / qwen3:8b) detect PHQ-2 / GAD-2 symptoms from a single conversation? — [analysis/phq4_detection_probe_rq3_v5_per_conv.py](analysis/phq4_detection_probe_rq3_v5_per_conv.py) is the current active version (v5; per-conversation scoring + 5 aggregation strategies).

Manuscript is [anonymous-submission-latex-2026.tex](anonymous-submission-latex-2026.tex). Don't edit the .tex unless asked — it round-trips through Overleaf via the `overleaf` git remote.

## First-time setup (run this once)

The teammate received a `aies_bootstrap.tar.gz` out-of-band (sensitive data — not in git). After they extract it at the repo root, the tree should contain `data/`, `results/`, and the two `Clinical Expert Case Review [#2]-STUDY00021493.*` files. If those are missing, stop and ask them where the tarball is.

Step-by-step:

```bash
# 1. Extract the bootstrap bundle (sensitive data — must be transferred out-of-band)
tar -xzf aies_bootstrap.tar.gz   # creates data/, results/, Clinical Expert Case Review *

# 2. Python deps (Python 3.11+; 3.14 works)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. RQ3 only: install Ollama + pull the model used in the paper
#    https://ollama.com/download
ollama serve &           # if not already running
ollama pull qwen3:8b

# 4. RQ3 only: the v5 script reads /tmp/clinical_review_mappings.csv
cp data/clinical_review_mappings.csv /tmp/clinical_review_mappings.csv
```

Sanity check after setup — these files must all exist:

- `data/survey.csv` (501 rows; raw PHQ-4/PROMIS items)
- `data/survey_scored.csv` (501 rows; AI_Q1..Q15 + Age)
- `data/demographics/prolific_export_*.csv` (7 files, globbed by the regression script)
- `data/liwc22_user_prompts.csv`, `data/filtered_user_prompts_lt20words.csv`
- `data/chat_participants_list.csv`, `data/ecr_scores_validated_clean.csv`
- `results/260419T1358_phq4_ai_controlled_regression/00_merged_analysis_data.csv` (RQ3 v5 reads this)
- `Clinical Expert Case Review [#2]-STUDY00021493.xlsx` + `... - Unamed.csv` at repo root (RQ3 v5 reads these)

## What the user wants from you

Two things, in priority order:

### 1. Verify the published analyses reproduce

Re-run the four scripts that back the paper's reported numbers and confirm they match the values cited in [anonymous-submission-latex-2026.tex](anonymous-submission-latex-2026.tex). Each writes a new timestamped folder under `results/` — diff against the reference run noted below.

| Script | Reference run | Key numbers to check |
|---|---|---|
| [analysis/phq4_ai_correlation_regression_analysis.py](analysis/phq4_ai_correlation_regression_analysis.py) | [results/260419T1358_phq4_ai_controlled_regression/](results/260419T1358_phq4_ai_controlled_regression/) | N=158; emotional engagement r ≈ .27, p < .01 (depression); see `04_hierarchical_regression.csv` |
| [analysis/phq4_liwc_analysis.py](analysis/phq4_liwc_analysis.py) | [results/260419T1358_phq4_liwc_analysis/](results/260419T1358_phq4_liwc_analysis/) | N=104; 7/46 LIWC features FDR-sig (future focus, we-words, neg emotion, affiliation, social refs, prosocial, they-words) |
| [analysis/phq4_categorical_rq1.py](analysis/phq4_categorical_rq1.py) | [results/260502T1637_phq4_rq1_categorical/](results/260502T1637_phq4_rq1_categorical/) | RQ1 categorical (clinical-cutoff group contrasts) |
| [analysis/phq4_categorical_rq2_liwc.py](analysis/phq4_categorical_rq2_liwc.py) | [results/260502T1640_phq4_rq2_liwc_categorical/](results/260502T1640_phq4_rq2_liwc_categorical/) | RQ2 categorical |
| [analysis/phq4_detection_probe_rq3_v5_per_conv.py](analysis/phq4_detection_probe_rq3_v5_per_conv.py) | [results/260502T1953_phq4_detection_probe_v5_per_conv/](results/260502T1953_phq4_detection_probe_v5_per_conv/) | RQ3 v5; `count_pos` aggregation marginally significant (per commit a4afa36) |

If a re-run diverges by more than rounding, surface the diff to the user before continuing — don't silently overwrite or "fix" it. RQ3 results are *expected* to vary slightly run-to-run because the LLM is stochastic; document the new numbers and the seed/temperature settings used.

### 2. Extend RQ3 v5 (the per-conversation detection probe)

Current state (commits 9257ff3, a524993, a4afa36): per-conversation cascade works, `count_pos` aggregation is marginally significant, but conversation-vs-trait ceiling effect is the main finding. Likely directions the user may ask about:

- Try larger / different local models (`qwen3:14b`, `llama3.1:8b`, `phi4:14b`); compare AUC per item.
- Add inter-rater reliability between v5 model output and the clinical expert's per-conversation ratings (in `Clinical Expert Case Review [#2]-STUDY00021493.xlsx`, sheet-per-pseudonym).
- Per-item ROC curves rather than aggregate-level Pearson.
- Sensitivity analysis on `MAX_CHARS` (currently 8000) — does truncation drive the ceiling?

Don't redesign the cascade prompts (`ITEMS` dict in v5) without checking with the user — those exemplars are clinically curated and changing them invalidates v3/v4/v5 comparability.

## Repo map (short)

| Path | What |
|---|---|
| [analysis/](analysis/) | Active scripts (see RQ table above). Older scripts in [analysis/_archive/](analysis/_archive/). |
| [data/](data/) | Self-contained data snapshot (gitignored). Populated by extracting `aies_bootstrap.tar.gz`. |
| [results/](results/) | Timestamped run outputs (gitignored). Reference runs cited above. |
| [figures/](figures/) | Publication figures. Built by [analysis/make_paper_figures.py](analysis/make_paper_figures.py). |
| [TODO.md](TODO.md) | Older task list (mostly RQ1/RQ2-era — superseded by this file's "What the user wants from you" section). |
| [README.md](README.md) | Public-facing repo description. |
| [Archive/](Archive/) | Old planning docs. Don't edit. |

## Conventions

- **Never commit `data/`, `results/`, or `*.csv` at any path.** They're in `.gitignore` for a reason — survey data is participant-identifiable.
- **The `.tex` is source-of-truth on the `overleaf` remote.** Don't push manuscript edits without asking; the user merges Overleaf → main manually.
- **RQ3 versioning is sequential, not parallel.** `phq4_detection_probe_rq3_v5_per_conv.py` is the live one; v1–v4 are kept for paper appendix comparability. Add v6 as a new file rather than mutating v5.
- **Timestamped result folders use `%y%m%dT%H%M`.** Don't rename; downstream scripts (e.g. v5 reading the v4 regression frame) hard-reference these paths.
