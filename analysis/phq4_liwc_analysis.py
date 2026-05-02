#!/usr/bin/env python3
"""PHQ-4 (anxiety + depression) x LIWC-22 linguistic patterns.

Parallels liwc22_analysis.py (which uses ECR attachment) but swaps predictors
for PHQ-4 anxiety and PHQ-4 depression. Mirrors the paper's RQ2 analysis:
do mental health scores predict linguistic patterns in ChatGPT prompts,
controlling for age, gender, and message volume?

Usage:
    python analysis/phq4_liwc_analysis.py
    python analysis/phq4_liwc_analysis.py --anxiety-source promis
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="PHQ-4 x LIWC-22 analysis")
parser.add_argument("--anxiety-source", choices=["phq4", "promis"], default="phq4")
args = parser.parse_args()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LIWC_PATH = DATA_DIR / "liwc22_user_prompts.csv"
SURVEY_RAW = DATA_DIR / "survey.csv"
SURVEY_SCORED = DATA_DIR / "survey_scored.csv"
CHAT_IDS = DATA_DIR / "chat_participants_list.csv"
DEMO_DIR = DATA_DIR / "demographics"

TIMESTAMP = datetime.now().strftime("%y%m%dT%H%M")
OUT_DIR = BASE_DIR / "results" / f"{TIMESTAMP}_phq4_liwc_analysis"
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

PHQ4_MAP = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
PROMIS_MAP = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5}
GENDER_MAP = {"Male": 0, "Female": 1}

LIWC_VARIABLES = {
    "Summary": [("Analytic", "Analytical Thinking"), ("Clout", "Social Status"),
                ("Authentic", "Authenticity"), ("Tone", "Emotional Tone")],
    "Pronouns": [("i", "I-words"), ("we", "We-words"), ("you", "You-words"),
                 ("shehe", "She/He"), ("they", "They")],
    "Affect": [("Affect", "Total Affect"), ("tone_pos", "Positive Tone"),
               ("tone_neg", "Negative Tone"), ("emo_pos", "Positive Emotion"),
               ("emo_neg", "Negative Emotion"), ("emo_anx", "Anxiety"),
               ("emo_anger", "Anger"), ("emo_sad", "Sadness")],
    "Cognition": [("Cognition", "Total Cognition"), ("cogproc", "Cognitive Processes"),
                  ("insight", "Insight"), ("cause", "Causation"), ("discrep", "Discrepancy"),
                  ("tentat", "Tentative"), ("certitude", "Certainty")],
    "Drives": [("Drives", "Total Drives"), ("affiliation", "Affiliation"),
               ("achieve", "Achievement"), ("power", "Power")],
    "Social": [("Social", "Total Social"), ("socbehav", "Social Behavior"),
               ("prosocial", "Prosocial"), ("polite", "Polite"), ("conflict", "Conflict"),
               ("family", "Family"), ("friend", "Friend")],
    "Perception": [("Perception", "Total Perception"), ("feeling", "Feeling"),
                   ("focuspast", "Past Focus"), ("focuspresent", "Present Focus"),
                   ("focusfuture", "Future Focus")],
    "Needs": [("need", "Need"), ("want", "Want"), ("lack", "Lack"),
              ("risk", "Risk"), ("curiosity", "Curiosity"), ("reward", "Reward")],
}
ALL_VARS = [(v, l, c) for c, lst in LIWC_VARIABLES.items() for v, l in lst]


def find_col(cols, needle):
    m = [c for c in cols if needle in c]
    if len(m) != 1:
        raise ValueError(f"Expected exactly one column containing {needle!r}, found {m}")
    return m[0]


def zscore(s):
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd and not pd.isna(sd) else pd.Series(np.nan, index=s.index)


def sig(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def compute_phq4(raw):
    raw = raw.rename(columns={"What is your Prolific ID": "Prolific_ID"})
    raw["Prolific_ID"] = raw["Prolific_ID"].astype(str).str.strip()
    raw = raw.drop_duplicates(subset="Prolific_ID", keep="first")
    cols = raw.columns.tolist()
    phq_anx = [find_col(cols, "Feeling nervous, anxious or on edge"),
               find_col(cols, "Not being able to stop or control worrying")]
    phq_dep = [find_col(cols, "Feeling down, depressed or hopeless"),
               find_col(cols, "Little interest or pleasure in doing things")]
    promis = [find_col(cols, "I felt fearful"), find_col(cols, "I felt anxious."),
              find_col(cols, "I felt worried."),
              find_col(cols, "I found it hard to focus on anything other than my anxiety."),
              find_col(cols, "I felt uneasy."), find_col(cols, "I felt nervous."),
              find_col(cols, "I felt tense.")]
    for c in phq_anx + phq_dep:
        raw[f"{c}__n"] = raw[c].map(PHQ4_MAP)
    for c in promis:
        raw[f"{c}__n"] = raw[c].map(PROMIS_MAP)
    raw["PHQ4_Anxiety"] = raw[[f"{c}__n" for c in phq_anx]].sum(axis=1, min_count=2)
    raw["PHQ4_Depression"] = raw[[f"{c}__n" for c in phq_dep]].sum(axis=1, min_count=2)
    raw["PROMIS_Anxiety"] = raw[[f"{c}__n" for c in promis]].sum(axis=1, min_count=len(promis))
    return raw[["Prolific_ID", "PHQ4_Anxiety", "PHQ4_Depression", "PROMIS_Anxiety"]]


def load_demographics():
    frames = []
    for p in sorted(DEMO_DIR.glob("prolific_export_*.csv")):
        d = pd.read_csv(p)
        if "Participant id" in d.columns and "Sex" in d.columns:
            frames.append(d[["Participant id", "Sex"]])
    if not frames:
        return pd.DataFrame(columns=["Prolific_ID", "Gender_numeric"])
    demo = pd.concat(frames).drop_duplicates("Participant id", keep="first")
    demo = demo.rename(columns={"Participant id": "Prolific_ID"})
    demo["Prolific_ID"] = demo["Prolific_ID"].astype(str).str.strip()
    demo = demo[demo["Sex"].isin(["Male", "Female"])].copy()
    demo["Gender_numeric"] = demo["Sex"].map(GENDER_MAP)
    return demo[["Prolific_ID", "Gender_numeric"]]


print("=" * 80)
print(f"PHQ-4 ({args.anxiety_source.upper()} anxiety + PHQ-4 depression) x LIWC-22")
print("=" * 80)
print(f"Output: {OUT_DIR}")

liwc_df = pd.read_csv(LIWC_PATH)
print(f"LIWC rows: {len(liwc_df):,}")

mh = compute_phq4(pd.read_csv(SURVEY_RAW))
print(f"Mental-health scored: {len(mh)}")

chat_ids = pd.read_csv(CHAT_IDS)
chat_ids["chat_participant_id"] = chat_ids["chat_participant_id"].astype(str).str.strip()
matched = set(mh["Prolific_ID"]) & set(chat_ids["chat_participant_id"])
print(f"Matched (survey + chat): {len(matched)}")

survey_scored = pd.read_csv(SURVEY_SCORED)
survey_scored["Prolific_ID"] = survey_scored["Prolific_ID"].astype(str).str.strip()
survey_scored = survey_scored.drop_duplicates("Prolific_ID", keep="first")
demographics = load_demographics()
print(f"Demographics (M/F): {len(demographics)}")

existing_vars = [v for v, _, _ in ALL_VARS if v in liwc_df.columns]
agg = {v: "mean" for v in existing_vars}
agg["WC"] = "sum"
agg["message"] = "count"
plw = liwc_df.groupby("participant_id").agg(agg).reset_index()
plw.rename(columns={"message": "n_messages", "WC": "total_words"}, inplace=True)
plw["participant_id"] = plw["participant_id"].astype(str).str.strip()

merged = (
    mh.merge(plw, left_on="Prolific_ID", right_on="participant_id", how="inner")
      .merge(survey_scored[["Prolific_ID", "Age"]], on="Prolific_ID", how="left")
      .merge(demographics, on="Prolific_ID", how="left")
)
merged["log_n_messages"] = np.log1p(merged["n_messages"])

anx_col = "PROMIS_Anxiety" if args.anxiety_source == "promis" else "PHQ4_Anxiety"
dep_col = "PHQ4_Depression"
anx_label = "PROMIS Anxiety" if args.anxiety_source == "promis" else "PHQ-4 Anxiety"

analysis = merged.dropna(subset=[anx_col, dep_col, "Age", "log_n_messages"]).copy()
gender_mean = analysis["Gender_numeric"].mean()
if pd.isna(gender_mean):
    gender_mean = 0.5
analysis["Gender_imp"] = analysis["Gender_numeric"].fillna(gender_mean)
analysis["Age_z"] = zscore(analysis["Age"])
analysis["Vol_z"] = zscore(analysis["log_n_messages"])
analysis["Anx_z"] = zscore(analysis[anx_col])
analysis["Dep_z"] = zscore(analysis[dep_col])

print(f"Final N (analysis): {len(analysis)}")
print(f"  Anxiety predictor: {anx_label} (M={analysis[anx_col].mean():.2f}, SD={analysis[anx_col].std():.2f})")
print(f"  Depression: PHQ-4 Depression (M={analysis[dep_col].mean():.2f}, SD={analysis[dep_col].std():.2f})")

def bivariate(predictor_col, label):
    rows = []
    for var, lbl, cat in ALL_VARS:
        if var not in merged.columns:
            continue
        valid = merged[[predictor_col, var]].dropna()
        if len(valid) <= 10:
            continue
        r, p = pearsonr(valid[predictor_col], valid[var])
        rows.append({"Variable": var, "Label": lbl, "Category": cat,
                     "N": len(valid), "r": r, "p": p, "r2": r * r})
    df = pd.DataFrame(rows)
    if len(df):
        _, df["p_fdr"], _, _ = multipletests(df["p"], method="fdr_bh")
        df["sig"] = df["p"].apply(sig)
        df["sig_fdr"] = df["p_fdr"].apply(sig)
        df["abs_r"] = df["r"].abs()
        df = df.sort_values("abs_r", ascending=False).drop(columns="abs_r")
    return df


anx_biv = bivariate(anx_col, anx_label)
dep_biv = bivariate(dep_col, "PHQ-4 Depression")
anx_biv.to_csv(OUT_DIR / "01_anxiety_liwc_correlations.csv", index=False)
dep_biv.to_csv(OUT_DIR / "02_depression_liwc_correlations.csv", index=False)

print(f"\n=== {anx_label} x LIWC (FDR) ===  sig: {(anx_biv['sig_fdr'] != '').sum()}/{len(anx_biv)}")
for _, r in anx_biv.head(10).iterrows():
    print(f"  {r['Label']:<25} r={r['r']:>6.3f} {r['sig_fdr']}")
print(f"\n=== PHQ-4 Depression x LIWC (FDR) ===  sig: {(dep_biv['sig_fdr'] != '').sum()}/{len(dep_biv)}")
for _, r in dep_biv.head(10).iterrows():
    print(f"  {r['Label']:<25} r={r['r']:>6.3f} {r['sig_fdr']}")


def ols(y, X):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()


def model_change(m1, m2, df_num):
    delta = max(m2.rsquared - m1.rsquared, 0.0)
    df_den = int(m2.df_resid)
    if delta <= 0 or df_den <= 0 or m2.rsquared >= 1:
        return delta, np.nan, np.nan
    f = (delta / df_num) / ((1 - m2.rsquared) / df_den)
    return delta, f, stats.f.sf(f, df_num, df_den)


ctrl_rows = []
for var, lbl, cat in ALL_VARS:
    if var not in merged.columns:
        continue
    d = analysis[["Age_z", "Gender_imp", "Vol_z", "Anx_z", "Dep_z", var]].dropna().copy()
    if len(d) <= 10:
        continue
    d["y_z"] = zscore(d[var])
    d = d.dropna()
    if len(d) <= 10:
        continue
    m1 = ols(d["y_z"], d[["Age_z", "Gender_imp", "Vol_z"]])
    m2 = ols(d["y_z"], d[["Age_z", "Gender_imp", "Vol_z", "Anx_z", "Dep_z"]])
    delta, f_ch, p_ch = model_change(m1, m2, 2)
    ctrl_rows.append({
        "Variable": var, "Label": lbl, "Category": cat, "N": len(d),
        "Step1_R2": m1.rsquared, "Step2_R2": m2.rsquared, "Delta_R2": delta,
        "F_change": f_ch, "p_change": p_ch,
        "Beta_Age": m2.params["Age_z"], "p_Age": m2.pvalues["Age_z"],
        "Beta_Gender": m2.params["Gender_imp"], "p_Gender": m2.pvalues["Gender_imp"],
        "Beta_Vol": m2.params["Vol_z"], "p_Vol": m2.pvalues["Vol_z"],
        "Beta_Anxiety": m2.params["Anx_z"], "t_Anxiety": m2.tvalues["Anx_z"],
        "p_Anxiety": m2.pvalues["Anx_z"],
        "Beta_Depression": m2.params["Dep_z"], "t_Depression": m2.tvalues["Dep_z"],
        "p_Depression": m2.pvalues["Dep_z"],
    })

ctrl = pd.DataFrame(ctrl_rows)
if len(ctrl):
    for col in ("p_Anxiety", "p_Depression", "p_change"):
        _, ctrl[f"{col}_fdr"], _, _ = multipletests(ctrl[col].fillna(1.0), method="fdr_bh")
        ctrl[f"sig_{col.replace('p_', '')}"] = ctrl[f"{col}_fdr"].apply(sig)
    ctrl["abs_anx"] = ctrl["Beta_Anxiety"].abs()
    ctrl["abs_dep"] = ctrl["Beta_Depression"].abs()

ctrl.sort_values("abs_anx", ascending=False).drop(columns=["abs_anx", "abs_dep"]).to_csv(
    OUT_DIR / "03_anxiety_liwc_controlled.csv", index=False
)
ctrl.sort_values("abs_dep", ascending=False).drop(columns=["abs_anx", "abs_dep"]).to_csv(
    OUT_DIR / "04_depression_liwc_controlled.csv", index=False
)

n_sig_anx_ctrl = (ctrl["sig_Anxiety"] != "").sum()
n_sig_dep_ctrl = (ctrl["sig_Depression"] != "").sum()
n_sig_change = (ctrl["sig_change"] != "").sum()
print(f"\nControlled (FDR):  Anxiety sig: {n_sig_anx_ctrl}/{len(ctrl)}  "
      f"Depression sig: {n_sig_dep_ctrl}/{len(ctrl)}  Model change sig: {n_sig_change}/{len(ctrl)}")

# Figure: top bivariate correlations side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
for ax, df, title in [
    (ax1, anx_biv.head(15), f"A. {anx_label} x LIWC (Top 15)"),
    (ax2, dep_biv.head(15), "B. PHQ-4 Depression x LIWC (Top 15)"),
]:
    y = np.arange(len(df))
    colors = ["#c44e52" if p < 0.001 else "#dd8452" if p < 0.01
              else "#f0c566" if p < 0.05 else "#b0b0b0" for p in df["p"]]
    ax.barh(y, df["r"], color=colors, edgecolor="white", linewidth=0.8, height=0.7)
    ax.axvline(0, color="#333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row['Label']} ({row['Category'][:4]})" for _, row in df.iterrows()], fontsize=9)
    ax.set_xlabel("Correlation (r)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlim(-0.4, 0.4)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    for i, (_, row) in enumerate(df.iterrows()):
        x = row["r"] + 0.02 if row["r"] >= 0 else row["r"] - 0.02
        ha = "left" if row["r"] >= 0 else "right"
        ax.text(x, i, f"r={row['r']:.2f}{row['sig']}", ha=ha, va="center", fontsize=8)
fig.suptitle(f"Mental Health ({anx_label} + PHQ-4 Depression) x LIWC-22", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "figures" / "FIG_mh_liwc_top15.png", dpi=300, bbox_inches="tight")
plt.close()

merged.to_csv(OUT_DIR / "00_merged_mh_liwc_data.csv", index=False)
print(f"\nDone. Outputs in {OUT_DIR}")
