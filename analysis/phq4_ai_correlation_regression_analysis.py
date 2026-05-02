#!/usr/bin/env python3
"""
Mental health x AI attitudes: hierarchical regression with controls.

This mirrors the existing attachment-based analysis while allowing either
PHQ-4 anxiety or PROMIS anxiety as the anxiety predictor. Depression is the
PHQ-4 depression subscale.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, shapiro

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


parser = argparse.ArgumentParser(description="Mental health x AI attitudes analysis")
parser.add_argument(
    "--anxiety-source",
    choices=["phq4", "promis"],
    default="phq4",
    help="Which anxiety measure to use in the regression models.",
)
parser.add_argument(
    "--output-dir",
    default="",
    help="Optional explicit output directory. Otherwise create a timestamped folder.",
)
parser.add_argument(
    "--run-suffix",
    default="",
    help="Optional custom suffix for auto-generated output directories.",
)
args = parser.parse_args()


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SURVEY_DATA = DATA_DIR / "survey.csv"
SURVEY_SCORED = DATA_DIR / "survey_scored.csv"
DEMOGRAPHICS_DIR = DATA_DIR / "demographics"

TIMESTAMP = datetime.now().strftime("%y%m%dT%H%M")
DEFAULT_RUN_SUFFIX = (
    "phq4_ai_controlled_regression"
    if args.anxiety_source == "phq4"
    else "promis_phq4_ai_controlled_regression"
)
OUTPUT_DIR = (
    Path(args.output_dir)
    if args.output_dir
    else BASE_DIR / "results" / f"{TIMESTAMP}_{args.run_suffix or DEFAULT_RUN_SUFFIX}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


LIKERT_MAP = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neither Agree nor Disagree": 3,
    "Agree": 4,
    "Strongly Agree": 5,
}

FREQ_MAP = {
    "Haven't used ChatGPT": 0,
    "Less than 5 times": 1,
    "5-9 times": 2,
    "10-19 times": 3,
    "20+ times": 4,
}

PHQ4_MAP = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}

PROMIS_MAP = {
    "Never": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always": 5,
}

GENDER_MAP = {"Male": 0, "Female": 1}
PARTICIPANT_ID_COL = "Participant id"
SEX_COL = "Sex"

AI_VARS_ALL = [f"AI_Q{i}" for i in range(1, 16)]

AI_LABELS_ALL = {
    "AI_Q1": "Trust",
    "AI_Q2": "Fact-Check",
    "AI_Q3": "Confident Implement",
    "AI_Q4": "Emotionally Understood",
    "AI_Q5": "Emotional Relief",
    "AI_Q6": "Easier to Share",
    "AI_Q7": "Problem Solving",
    "AI_Q8": "More Capable",
    "AI_Q9": "More Efficient",
    "AI_Q10": "Modify Writing",
    "AI_Q11": "Learning Different",
    "AI_Q12": "Communication Changed",
    "AI_Q13": "Worry Relying",
    "AI_Q14": "Less Confident Without",
    "AI_Q15": "Prefer Attempt First",
}

COMPOSITE_VARS = {
    "Emotional_Engagement_Comp": {
        "items": ["AI_Q4", "AI_Q5", "AI_Q6"],
        "label": "Emotional Engagement (Q4-6)",
        "description": "Mean of: Emotionally Understood, Emotional Relief, Easier to Share",
    },
    "Self_Efficacy_Comp": {
        "items": ["AI_Q7", "AI_Q8", "AI_Q9"],
        "label": "Self-Efficacy (Q7-9)",
        "description": "Mean of: Problem Solving, More Capable, More Efficient",
    },
    "Behavior_Change_Comp": {
        "items": ["AI_Q10", "AI_Q11", "AI_Q12"],
        "label": "Behavioral Change (Q10-12)",
        "description": "Mean of: Modify Writing, Learning Different, Communication Changed",
    },
    "Trust_Single": {
        "items": ["AI_Q1"],
        "label": "Trust (Q1)",
        "description": "I trust the answers ChatGPT provides",
    },
    "Dependency_Single": {
        "items": ["AI_Q13"],
        "label": "Dependency Concern (Q13)",
        "description": "I worry I am relying on it too much",
    },
}

AI_VARS = list(COMPOSITE_VARS.keys())
AI_LABELS = {key: value["label"] for key, value in COMPOSITE_VARS.items()}

AI_GROUPS = {
    "Emotional Engagement": ["AI_Q4", "AI_Q5", "AI_Q6"],
    "Trust & Verification": ["AI_Q1", "AI_Q2", "AI_Q3"],
    "Dependency": ["AI_Q13", "AI_Q14", "AI_Q15"],
    "Self-Efficacy": ["AI_Q7", "AI_Q8", "AI_Q9"],
    "Behavioral Change": ["AI_Q10", "AI_Q11", "AI_Q12"],
}


def interpret_r(r: float) -> str:
    r_abs = abs(r)
    if r_abs < 0.10:
        return "negligible"
    if r_abs < 0.30:
        return "small"
    if r_abs < 0.50:
        return "medium"
    return "large"


def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def find_column(columns: list[str], needle: str) -> str:
    matches = [column for column in columns if needle in column]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one column containing {needle!r}, found {matches}")
    return matches[0]


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def compute_vif(x: np.ndarray) -> list[float]:
    n_predictors = x.shape[1]
    vif_values: list[float] = []

    for i in range(n_predictors):
        y_i = x[:, i]
        x_others = np.delete(x, i, axis=1)

        if x_others.shape[1] == 0:
            vif_values.append(1.0)
            continue

        x_others_const = np.column_stack([np.ones(len(x_others)), x_others])

        try:
            beta = np.linalg.lstsq(x_others_const, y_i, rcond=None)[0]
            y_pred = x_others_const @ beta
            ss_res = np.sum((y_i - y_pred) ** 2)
            ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
        except Exception:
            vif = np.nan

        vif_values.append(vif)

    return vif_values


def plot_residual_diagnostics(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    outcome_name: str,
    output_dir: Path,
) -> tuple[float, float]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors="white", linewidth=0.5)
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted (Homoscedasticity)")

    try:
        from scipy.ndimage import uniform_filter1d

        sorted_idx = np.argsort(y_pred)
        smoothed = uniform_filter1d(
            residuals[sorted_idx], size=max(10, len(residuals) // 10)
        )
        ax1.plot(y_pred[sorted_idx], smoothed, color="blue", linewidth=2, label="Trend")
    except Exception:
        pass

    ax2 = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (Normality of Residuals)")

    ax3 = axes[1, 0]
    std_residuals = residuals / np.std(residuals)
    ax3.scatter(
        y_pred,
        np.sqrt(np.abs(std_residuals)),
        alpha=0.6,
        edgecolors="white",
        linewidth=0.5,
    )
    ax3.set_xlabel("Fitted Values")
    ax3.set_ylabel("√|Standardized Residuals|")
    ax3.set_title("Scale-Location (Homoscedasticity)")

    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=20, edgecolor="white", alpha=0.7)
    ax4.axvline(x=0, color="red", linestyle="--", linewidth=1)
    ax4.set_xlabel("Residuals")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution of Residuals")

    if len(residuals) < 5000:
        sw_stat, sw_p = shapiro(residuals)
        ax4.text(
            0.95,
            0.95,
            f"Shapiro-Wilk\nW={sw_stat:.3f}\np={sw_p:.4f}",
            transform=ax4.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
    else:
        sw_stat, sw_p = np.nan, np.nan

    fig.suptitle(f"Regression Diagnostics: {outcome_name}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    safe_name = outcome_name.replace(" ", "_").replace("/", "_")
    fig.savefig(output_dir / f"diag_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return sw_stat, sw_p


def cronbach_alpha(items_df: pd.DataFrame) -> float:
    items = items_df.dropna()
    n_items = items.shape[1]
    if n_items < 2:
        return np.nan

    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan

    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return alpha


def compute_group_analysis(
    df: pd.DataFrame,
    anxiety_col: str,
    depression_col: str,
    output_dir: Path,
) -> pd.DataFrame:
    results = []
    all_p_values: list[float] = []

    for group_name, items in AI_GROUPS.items():
        item_cols = [f"{item}_numeric" for item in items]
        missing_cols = [column for column in item_cols if column not in df.columns]
        if missing_cols:
            print(f"  Warning: missing columns for {group_name}: {missing_cols}")
            continue

        group_df = df[item_cols + [anxiety_col, depression_col]].dropna().copy()
        n = len(group_df)
        if n < 3:
            print(f"  Warning: insufficient data for {group_name} (n={n})")
            continue

        alpha = cronbach_alpha(group_df[item_cols])
        group_df["composite"] = group_df[item_cols].mean(axis=1)

        r_anxiety, p_anxiety = pearsonr(group_df[anxiety_col], group_df["composite"])
        r_depression, p_depression = pearsonr(
            group_df[depression_col], group_df["composite"]
        )

        def r_to_ci(r: float, n_obs: int, alpha_level: float = 0.05) -> tuple[float, float]:
            z = np.arctanh(r)
            se = 1 / np.sqrt(n_obs - 3)
            z_crit = stats.norm.ppf(1 - alpha_level / 2)
            return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)

        ci_anxiety = r_to_ci(r_anxiety, n)
        ci_depression = r_to_ci(r_depression, n)

        all_p_values.extend([p_anxiety, p_depression])

        results.append(
            {
                "Group": group_name,
                "Items": ", ".join(items),
                "N_items": len(items),
                "N": n,
                "Cronbach_Alpha": alpha,
                "Alpha_Interpretation": (
                    "Excellent"
                    if alpha >= 0.9
                    else "Good"
                    if alpha >= 0.8
                    else "Acceptable"
                    if alpha >= 0.7
                    else "Questionable"
                    if alpha >= 0.6
                    else "Poor"
                ),
                "Composite_M": group_df["composite"].mean(),
                "Composite_SD": group_df["composite"].std(),
                "r_Anxiety": r_anxiety,
                "r_Anxiety_CI": f"[{ci_anxiety[0]:.3f}, {ci_anxiety[1]:.3f}]",
                "p_Anxiety": p_anxiety,
                "r_Depression": r_depression,
                "r_Depression_CI": f"[{ci_depression[0]:.3f}, {ci_depression[1]:.3f}]",
                "p_Depression": p_depression,
            }
        )

    results_df = pd.DataFrame(results)
    if len(all_p_values) > 0:
        fdr_corrected = benjamini_hochberg(np.array(all_p_values))
        idx = 0
        for i in range(len(results_df)):
            results_df.loc[i, "p_Anxiety_FDR"] = fdr_corrected[idx]
            results_df.loc[i, "p_Depression_FDR"] = fdr_corrected[idx + 1]
            results_df.loc[i, "sig_Anxiety"] = significance_label(fdr_corrected[idx])
            results_df.loc[i, "sig_Depression"] = significance_label(
                fdr_corrected[idx + 1]
            )
            idx += 2

    results_df.to_csv(output_dir / "06_group_analysis.csv", index=False)
    return results_df


def compute_correlation_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]
    n = len(x_clean)
    if n < 3:
        return {
            "r": np.nan,
            "p": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": n,
            "r_squared": np.nan,
        }

    r, p = pearsonr(x_clean, y_clean)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = np.tanh(z - z_crit * se)
    ci_high = np.tanh(z + z_crit * se)
    return {
        "r": r,
        "p": p,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "r_squared": r**2,
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    adjusted_p = np.zeros(n)
    for i in range(n):
        adjusted_p[sorted_indices[i]] = min(1, sorted_p[i] * n / (i + 1))
    return adjusted_p


def run_regression(y: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray] | None:
    x_with_const = np.column_stack([np.ones(len(x)), x])
    n, p = x_with_const.shape
    k = p - 1

    try:
        beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
    except Exception:
        return None

    y_pred = x_with_const @ beta
    residuals = y - y_pred

    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals**2)
    ss_regression = ss_total - ss_residual

    r_squared = ss_regression / ss_total if ss_total > 0 else 0
    adj_r_squared = (
        1 - (1 - r_squared) * (n - 1) / (n - k - 1) if (n - k - 1) > 0 else 0
    )

    ms_regression = ss_regression / k if k > 0 else 0
    ms_residual = ss_residual / (n - k - 1) if (n - k - 1) > 0 else 1
    f_stat = ms_regression / ms_residual if ms_residual > 0 else 0
    f_p = 1 - stats.f.cdf(f_stat, k, n - k - 1)

    var_residual = ss_residual / (n - k - 1) if (n - k - 1) > 0 else 1
    try:
        var_beta = var_residual * np.linalg.inv(x_with_const.T @ x_with_const)
        se_beta = np.sqrt(np.diag(var_beta))
    except Exception:
        se_beta = np.zeros(len(beta))

    t_stats = np.where(se_beta > 0, beta / se_beta, 0)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

    return {
        "n": n,
        "k": k,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "f_stat": f_stat,
        "f_p": f_p,
        "df1": k,
        "df2": n - k - 1,
        "betas": beta[1:],
        "se_betas": se_beta[1:],
        "t_stats": t_stats[1:],
        "p_values": p_values[1:],
        "intercept": beta[0],
    }


print("=" * 80)
print("MENTAL HEALTH x AI ATTITUDES: HIERARCHICAL REGRESSION")
print("=" * 80)
print(f"Output: {OUTPUT_DIR}")

print("\n--- Loading Data ---")
survey_raw = pd.read_csv(SURVEY_DATA)
survey_raw = survey_raw.rename(
    columns={
        "What is your Prolific ID": "Prolific_ID",
        "How often do you use ChatGPT in the past two weeks?": "ChatGPT_Frequency",
    }
)
survey_raw["Prolific_ID"] = survey_raw["Prolific_ID"].astype(str).str.strip()
survey_raw["Frequency_numeric"] = survey_raw["ChatGPT_Frequency"].map(FREQ_MAP)
survey_raw = survey_raw.drop_duplicates(subset="Prolific_ID", keep="first")
print(f"Raw survey: {len(survey_raw)} unique participants")

phq_anxiety_cols = [
    find_column(survey_raw.columns.tolist(), "Feeling nervous, anxious or on edge"),
    find_column(
        survey_raw.columns.tolist(), "Not being able to stop or control worrying"
    ),
]
phq_depression_cols = [
    find_column(survey_raw.columns.tolist(), "Feeling down, depressed or hopeless"),
    find_column(
        survey_raw.columns.tolist(), "Little interest or pleasure in doing things"
    ),
]
promis_anxiety_cols = [
    find_column(survey_raw.columns.tolist(), "I felt fearful"),
    find_column(survey_raw.columns.tolist(), "I felt anxious."),
    find_column(survey_raw.columns.tolist(), "I felt worried."),
    find_column(
        survey_raw.columns.tolist(),
        "I found it hard to focus on anything other than my anxiety.",
    ),
    find_column(survey_raw.columns.tolist(), "I felt uneasy."),
    find_column(survey_raw.columns.tolist(), "I felt nervous."),
    find_column(survey_raw.columns.tolist(), "I felt tense."),
]

for column in phq_anxiety_cols + phq_depression_cols:
    survey_raw[f"{column}__numeric"] = survey_raw[column].map(PHQ4_MAP)
for column in promis_anxiety_cols:
    survey_raw[f"{column}__numeric"] = survey_raw[column].map(PROMIS_MAP)

survey_raw["PHQ4_Anxiety"] = survey_raw[
    [f"{column}__numeric" for column in phq_anxiety_cols]
].sum(axis=1, min_count=2)
survey_raw["PHQ4_Depression"] = survey_raw[
    [f"{column}__numeric" for column in phq_depression_cols]
].sum(axis=1, min_count=2)
survey_raw["PROMIS_Anxiety"] = survey_raw[
    [f"{column}__numeric" for column in promis_anxiety_cols]
].sum(axis=1, min_count=len(promis_anxiety_cols))
survey_raw["PHQ4_Total"] = survey_raw[["PHQ4_Anxiety", "PHQ4_Depression"]].sum(
    axis=1, min_count=2
)
print(f"  With PHQ-4 anxiety: {survey_raw['PHQ4_Anxiety'].notna().sum()}")
print(f"  With PHQ-4 depression: {survey_raw['PHQ4_Depression'].notna().sum()}")
print(f"  With PROMIS anxiety: {survey_raw['PROMIS_Anxiety'].notna().sum()}")

if args.anxiety_source == "promis":
    anxiety_col = "PROMIS_Anxiety"
    anxiety_label = "PROMIS Anxiety"
    anxiety_summary_prefix = "PROMIS_Anxiety"
else:
    anxiety_col = "PHQ4_Anxiety"
    anxiety_label = "PHQ-4 Anxiety"
    anxiety_summary_prefix = "PHQ4_Anxiety"

depression_col = "PHQ4_Depression"
depression_label = "PHQ-4 Depression"

survey_scored = pd.read_csv(SURVEY_SCORED)
survey_scored["Prolific_ID"] = survey_scored["Prolific_ID"].astype(str).str.strip()
survey_scored = survey_scored.drop_duplicates(subset="Prolific_ID", keep="first")
print(f"Survey_scored: {len(survey_scored)} unique participants")

for var in AI_VARS_ALL:
    if var in survey_scored.columns:
        survey_scored[f"{var}_numeric"] = survey_scored[var].map(LIKERT_MAP)

for comp_name, comp_info in COMPOSITE_VARS.items():
    item_cols = [f"{item}_numeric" for item in comp_info["items"]]
    if len(item_cols) > 1:
        survey_scored[comp_name] = survey_scored[item_cols].mean(axis=1)
    else:
        survey_scored[comp_name] = survey_scored[item_cols[0]]

demo_dfs = []
for csv_path in DEMOGRAPHICS_DIR.glob("prolific_export_*.csv"):
    demo_df = pd.read_csv(csv_path)
    if PARTICIPANT_ID_COL in demo_df.columns and SEX_COL in demo_df.columns:
        demo_dfs.append(demo_df[[PARTICIPANT_ID_COL, SEX_COL]].copy())

demographics = (
    pd.concat(demo_dfs, ignore_index=True)
    .drop_duplicates(subset=PARTICIPANT_ID_COL, keep="first")
    .rename(columns={PARTICIPANT_ID_COL: "Prolific_ID"})
)
demographics["Prolific_ID"] = demographics["Prolific_ID"].astype(str).str.strip()
demographics = demographics[demographics[SEX_COL].isin(["Male", "Female"])]
demographics["Gender_numeric"] = demographics[SEX_COL].map(GENDER_MAP)
print(f"Demographics: {len(demographics)} with valid gender")

print("\n--- Merging Data ---")
merged_df = survey_scored.merge(
    survey_raw[
        [
            "Prolific_ID",
            "Frequency_numeric",
            "PHQ4_Anxiety",
            "PHQ4_Depression",
            "PROMIS_Anxiety",
        ]
    ],
    on="Prolific_ID",
    how="left",
    validate="one_to_one",
)
merged_df = merged_df.merge(
    demographics[["Prolific_ID", "Gender_numeric"]],
    on="Prolific_ID",
    how="left",
    validate="one_to_one",
)
merged_df["Age"] = pd.to_numeric(merged_df["Age"], errors="coerce")

print(f"Merged: {len(merged_df)} participants")
print(f"  With Age: {merged_df['Age'].notna().sum()}")
print(f"  With Frequency: {merged_df['Frequency_numeric'].notna().sum()}")
print(f"  With Gender: {merged_df['Gender_numeric'].notna().sum()}")
print(f"  With PHQ-4 Anxiety: {merged_df['PHQ4_Anxiety'].notna().sum()}")
print(f"  With PHQ-4 Depression: {merged_df['PHQ4_Depression'].notna().sum()}")
print(f"  With PROMIS Anxiety: {merged_df['PROMIS_Anxiety'].notna().sum()}")

ai_numeric_cols_all = [f"{var}_numeric" for var in AI_VARS_ALL]
ai_complete = merged_df[ai_numeric_cols_all].notna().all(axis=1).sum()
print(f"  With all 15 AI attitudes: {ai_complete}")

composite_cols = list(COMPOSITE_VARS.keys())

print("\n--- Handling Missing Data ---")
required_cols = [
    anxiety_col,
    depression_col,
    "Age",
    "Frequency_numeric",
] + composite_cols
complete_mask = merged_df[required_cols].notna().all(axis=1)
analysis_df = merged_df[complete_mask].copy()

n_missing_gender = analysis_df["Gender_numeric"].isna().sum()
gender_mean = analysis_df["Gender_numeric"].mean()
analysis_df["Gender_imputed"] = analysis_df["Gender_numeric"].fillna(gender_mean)
analysis_df["Gender_missing"] = analysis_df["Gender_numeric"].isna().astype(int)

N = len(analysis_df)
print(f"Complete cases ({anxiety_label} + {depression_label} + Age + Freq + AI): N = {N}")
print(f"  Gender observed: {N - n_missing_gender}")
print(f"  Gender imputed (mean={gender_mean:.2f}): {n_missing_gender}")

analysis_df["Age_z"] = zscore(analysis_df["Age"])
analysis_df["Frequency_z"] = zscore(analysis_df["Frequency_numeric"])
analysis_df["Anxiety_z"] = zscore(analysis_df[anxiety_col])
analysis_df["Depression_z"] = zscore(analysis_df[depression_col])

merged_out_path = OUTPUT_DIR / "00_merged_analysis_data.csv"
analysis_df.to_csv(merged_out_path, index=False)

print("\n--- Descriptive Statistics ---")
descriptives = []

for var, label in [
    (anxiety_col, anxiety_label),
    (depression_col, depression_label),
    ("Age", "Age"),
    ("Gender_imputed", "Gender (0=M, 1=F)"),
    ("Frequency_numeric", "ChatGPT Frequency"),
]:
    data = analysis_df[var]
    descriptives.append(
        {
            "Variable": label,
            "N": len(data),
            "Mean": data.mean(),
            "SD": data.std(),
            "Median": data.median(),
            "Min": data.min(),
            "Max": data.max(),
        }
    )
    print(f"  {label}: M={data.mean():.2f}, SD={data.std():.2f}")

for var in AI_VARS:
    data = analysis_df[var]
    comp_info = COMPOSITE_VARS[var]
    descriptives.append(
        {
            "Variable": AI_LABELS[var],
            "N": len(data),
            "Mean": data.mean(),
            "SD": data.std(),
            "Median": data.median(),
            "Min": data.min(),
            "Max": data.max(),
            "N_items": len(comp_info["items"]),
            "Items": ", ".join(comp_info["items"]),
        }
    )
    print(f"  {AI_LABELS[var]}: M={data.mean():.2f}, SD={data.std():.2f}")

pd.DataFrame(descriptives).to_csv(OUTPUT_DIR / "01_descriptive_statistics.csv", index=False)

print("\n--- Zero-Order Correlations (Composites + Singles) ---")
anxiety_results, depression_results = [], []

for var in AI_VARS:
    result = compute_correlation_with_ci(
        analysis_df[anxiety_col].values, analysis_df[var].values
    )
    result.update(
        {
            "AI_Attitude": AI_LABELS[var],
            "Variable": var,
            "Predictor_Dimension": "Anxiety",
            "Effect_Size": interpret_r(result["r"]),
            "Items": ", ".join(COMPOSITE_VARS[var]["items"]),
        }
    )
    anxiety_results.append(result)

    result = compute_correlation_with_ci(
        analysis_df[depression_col].values, analysis_df[var].values
    )
    result.update(
        {
            "AI_Attitude": AI_LABELS[var],
            "Variable": var,
            "Predictor_Dimension": "Depression",
            "Effect_Size": interpret_r(result["r"]),
            "Items": ", ".join(COMPOSITE_VARS[var]["items"]),
        }
    )
    depression_results.append(result)

anxiety_df = pd.DataFrame(anxiety_results)
anxiety_df["p_fdr"] = benjamini_hochberg(anxiety_df["p"].values)
anxiety_df["sig_fdr"] = anxiety_df["p_fdr"].apply(significance_label)
anxiety_df.to_csv(OUTPUT_DIR / "02_anxiety_correlations.csv", index=False)

depression_df = pd.DataFrame(depression_results)
depression_df["p_fdr"] = benjamini_hochberg(depression_df["p"].values)
depression_df["sig_fdr"] = depression_df["p_fdr"].apply(significance_label)
depression_df.to_csv(OUTPUT_DIR / "03_depression_correlations.csv", index=False)

n_sig_anx = (anxiety_df["sig_fdr"] != "ns").sum()
n_sig_dep = (depression_df["sig_fdr"] != "ns").sum()
print(f"  Anxiety: {n_sig_anx}/{len(AI_VARS)} significant (FDR corrected)")
print(f"  Depression: {n_sig_dep}/{len(AI_VARS)} significant (FDR corrected)")

print("\n--- Hierarchical Multiple Regression ---")
print("  Step 1: Controls (Age, Gender, ChatGPT Frequency)")
print(f"  Step 2: + {anxiety_label}, {depression_label}")
print(f"  Note: Gender imputed for {n_missing_gender} participants")

x_full = analysis_df[
    ["Age_z", "Gender_imputed", "Frequency_z", "Anxiety_z", "Depression_z"]
].values
predictor_names = ["Age", "Gender", "Frequency", anxiety_label, depression_label]
vif_values = compute_vif(x_full)

print("\n  VIF (Variance Inflation Factor) for multicollinearity:")
vif_results = []
for name, vif in zip(predictor_names, vif_values):
    interpretation = "OK" if vif < 5 else "Moderate" if vif < 10 else "Severe"
    print(f"    {name}: VIF = {vif:.2f} ({interpretation})")
    vif_results.append({"Predictor": name, "VIF": vif, "Interpretation": interpretation})

pd.DataFrame(vif_results).to_csv(OUTPUT_DIR / "07_vif_multicollinearity.csv", index=False)

diag_dir = OUTPUT_DIR / "diagnostics"
diag_dir.mkdir(exist_ok=True)

hierarchical_results = []
diagnostic_results = []

for var in AI_VARS:
    y = analysis_df[var].values

    x_step1 = analysis_df[["Age_z", "Gender_imputed", "Frequency_z"]].values
    result_step1 = run_regression(y, x_step1)

    x_step2 = analysis_df[
        ["Age_z", "Gender_imputed", "Frequency_z", "Anxiety_z", "Depression_z"]
    ].values
    result_step2 = run_regression(y, x_step2)

    if result_step1 and result_step2:
        delta_r2 = result_step2["r_squared"] - result_step1["r_squared"]
        df1_change = 2
        df2 = result_step2["df2"]
        f_change = (
            (delta_r2 / df1_change) / ((1 - result_step2["r_squared"]) / df2)
            if df2 > 0
            else 0
        )
        p_change = 1 - stats.f.cdf(f_change, df1_change, df2) if f_change > 0 else 1

        x_const = np.column_stack([np.ones(len(x_step2)), x_step2])
        beta = np.linalg.lstsq(x_const, y, rcond=None)[0]
        y_pred = x_const @ beta
        residuals = y - y_pred

        sw_stat, sw_p = plot_residual_diagnostics(y_pred, residuals, AI_LABELS[var], diag_dir)

        diagnostic_results.append(
            {
                "AI_Attitude": AI_LABELS[var],
                "Shapiro_W": sw_stat,
                "Shapiro_p": sw_p,
                "Normality": "Yes" if sw_p > 0.05 else "No",
                "Residual_Mean": residuals.mean(),
                "Residual_SD": residuals.std(),
            }
        )

        hierarchical_results.append(
            {
                "AI_Attitude": AI_LABELS[var],
                "Variable": var,
                "Items": ", ".join(COMPOSITE_VARS[var]["items"]),
                "N": result_step2["n"],
                "Step1_R2": result_step1["r_squared"],
                "Step1_AdjR2": result_step1["adj_r_squared"],
                "Step1_F": result_step1["f_stat"],
                "Step1_F_p": result_step1["f_p"],
                "Beta_Age_Step1": result_step1["betas"][0],
                "Beta_Gender_Step1": result_step1["betas"][1],
                "Beta_Frequency_Step1": result_step1["betas"][2],
                "Step2_R2": result_step2["r_squared"],
                "Step2_AdjR2": result_step2["adj_r_squared"],
                "Step2_F": result_step2["f_stat"],
                "Step2_F_p": result_step2["f_p"],
                "Delta_R2": delta_r2,
                "F_change": f_change,
                "p_change": p_change,
                "Beta_Age": result_step2["betas"][0],
                "t_Age": result_step2["t_stats"][0],
                "p_Age": result_step2["p_values"][0],
                "Beta_Gender": result_step2["betas"][1],
                "t_Gender": result_step2["t_stats"][1],
                "p_Gender": result_step2["p_values"][1],
                "Beta_Frequency": result_step2["betas"][2],
                "t_Frequency": result_step2["t_stats"][2],
                "p_Frequency": result_step2["p_values"][2],
                "Beta_Anxiety": result_step2["betas"][3],
                "t_Anxiety": result_step2["t_stats"][3],
                "p_Anxiety": result_step2["p_values"][3],
                "Beta_Depression": result_step2["betas"][4],
                "t_Depression": result_step2["t_stats"][4],
                "p_Depression": result_step2["p_values"][4],
            }
        )

hierarchical_df = pd.DataFrame(hierarchical_results).sort_values(
    "Step2_R2", ascending=False
)
hierarchical_df["p_change_FDR"] = benjamini_hochberg(hierarchical_df["p_change"].values)
hierarchical_df["sig_change"] = hierarchical_df["p_change_FDR"].apply(significance_label)
hierarchical_df["p_Anxiety_FDR"] = benjamini_hochberg(hierarchical_df["p_Anxiety"].values)
hierarchical_df["p_Depression_FDR"] = benjamini_hochberg(
    hierarchical_df["p_Depression"].values
)
hierarchical_df.to_csv(OUTPUT_DIR / "04_hierarchical_regression.csv", index=False)

diagnostic_df = pd.DataFrame(diagnostic_results)
diagnostic_df.to_csv(OUTPUT_DIR / "08_residual_diagnostics.csv", index=False)

n_vars = len(AI_VARS)
n_normal = (diagnostic_df["Normality"] == "Yes").sum()
print(f"\n  Residual normality (Shapiro-Wilk p > .05): {n_normal}/{n_vars} models")
print(f"  Diagnostic plots saved to: {diag_dir}")

n_sig_change = (hierarchical_df["sig_change"] != "ns").sum()
print(
    f"\n  Models where {anxiety_label.lower()}/{depression_label.lower()} add "
    f"significant ΔR² (FDR): {n_sig_change}/{n_vars}"
)
print(f"  Max ΔR²: {hierarchical_df['Delta_R2'].max():.3f}")
print(f"  Max total R²: {hierarchical_df['Step2_R2'].max():.3f}")

print(
    f"\n--- {anxiety_label}/{depression_label} Effects After Controlling for "
    "Age, Gender, Frequency ---"
)
print("  (FDR corrected p-values)")

controlled_results = []
for _, row in hierarchical_df.iterrows():
    p_anx_fdr = row["p_Anxiety_FDR"]
    p_dep_fdr = row["p_Depression_FDR"]
    p_change_fdr = row["p_change_FDR"]

    controlled_results.append(
        {
            "AI_Attitude": row["AI_Attitude"],
            "Variable": row["Variable"],
            "Items": row["Items"],
            "Beta_Anxiety": row["Beta_Anxiety"],
            "t_Anxiety": row["t_Anxiety"],
            "p_Anxiety": row["p_Anxiety"],
            "p_Anxiety_FDR": p_anx_fdr,
            "sig_Anxiety": significance_label(p_anx_fdr),
            "Beta_Depression": row["Beta_Depression"],
            "t_Depression": row["t_Depression"],
            "p_Depression": row["p_Depression"],
            "p_Depression_FDR": p_dep_fdr,
            "sig_Depression": significance_label(p_dep_fdr),
            "Delta_R2": row["Delta_R2"],
            "F_change": row["F_change"],
            "p_change": row["p_change"],
            "p_change_FDR": p_change_fdr,
            "sig_change": significance_label(p_change_fdr),
        }
    )

controlled_df = pd.DataFrame(controlled_results)
controlled_df.to_csv(OUTPUT_DIR / "05_anxiety_depression_effects_controlled.csv", index=False)

n_sig_anx_ctrl = (controlled_df["sig_Anxiety"] != "ns").sum()
n_sig_dep_ctrl = (controlled_df["sig_Depression"] != "ns").sum()
print(f"  Anxiety significant (FDR): {n_sig_anx_ctrl}/{n_vars}")
print(f"  Depression significant (FDR): {n_sig_dep_ctrl}/{n_vars}")

summary = {
    "N": int(N),
    "n_variables": n_vars,
    "variables": list(AI_VARS),
    "anxiety_source": args.anxiety_source,
    "predictors": [anxiety_col, depression_col],
    "N_gender_observed": int(N - n_missing_gender),
    "N_gender_imputed": int(n_missing_gender),
    f"{anxiety_summary_prefix}_M": float(analysis_df[anxiety_col].mean()),
    f"{anxiety_summary_prefix}_SD": float(analysis_df[anxiety_col].std()),
    "PHQ4_Depression_M": float(analysis_df[depression_col].mean()),
    "PHQ4_Depression_SD": float(analysis_df[depression_col].std()),
    "Age_M": float(analysis_df["Age"].mean()),
    "Age_SD": float(analysis_df["Age"].std()),
    "Gender_pct_female": float(analysis_df["Gender_imputed"].mean() * 100),
    "Frequency_M": float(analysis_df["Frequency_numeric"].mean()),
    "Frequency_SD": float(analysis_df["Frequency_numeric"].std()),
    "n_sig_anxiety_zeroorder_FDR": int(n_sig_anx),
    "n_sig_depression_zeroorder_FDR": int(n_sig_dep),
    "n_sig_anxiety_controlled_FDR": int(n_sig_anx_ctrl),
    "n_sig_depression_controlled_FDR": int(n_sig_dep_ctrl),
    "n_sig_delta_r2_FDR": int(n_sig_change),
    "max_delta_r2": float(hierarchical_df["Delta_R2"].max()),
    "timestamp": datetime.now().isoformat(),
}

print("\n--- AI Attitude Group Analysis ---")
print("  Computing Cronbach's alpha and composite correlations per subscale")
group_results = compute_group_analysis(
    analysis_df, anxiety_col, depression_col, OUTPUT_DIR
)

print("\n  Group Results:")
print(f"  {'Group':<22} {'α':>6} {'r(Anx)':>8} {'r(Dep)':>8}")
print("  " + "-" * 50)
for _, row in group_results.iterrows():
    print(
        f"  {row['Group']:<22} {row['Cronbach_Alpha']:>6.2f} "
        f"{row['r_Anxiety']:>7.2f}{row['sig_Anxiety']:<3} "
        f"{row['r_Depression']:>7.2f}{row['sig_Depression']:<3}"
    )

summary["group_alphas"] = {
    row["Group"]: round(row["Cronbach_Alpha"], 3) for _, row in group_results.iterrows()
}

with open(OUTPUT_DIR / "summary_stats.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print(f"\n{'=' * 80}")
print(f"Results saved to {OUTPUT_DIR}")
print(f"  N = {N} (gender imputed for {n_missing_gender})")
print(f"  Variables analyzed: {n_vars} (3 composites + 2 singles)")
print(f"  Anxiety predictor: {anxiety_label}")
print(f"  Depression predictor: {depression_label}")
print("  All p-values are FDR corrected (Benjamini-Hochberg)")
