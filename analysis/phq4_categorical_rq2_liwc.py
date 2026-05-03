#!/usr/bin/env python3
"""RQ2 (categorical): clinically meaningful tiers of PHQ-4 / PROMIS distress
predicting LIWC-22 psycholinguistic markers in user prompts.

Two analytic families (primary vs exploratory) to mitigate FDR-burden:
  PRIMARY (a-priori, hypothesis-driven from Tackman 2019, Guntuku 2022, Thorstad 2023):
    - i, we, they, focusfuture, emo_neg, affiliation, Social
    7 features x 2 predictors = 14 tests, FDR within this family.
  EXPLORATORY (full 46 LIWC):
    46 features x 2 predictors = 92 tests, FDR within this family (reported as
    discovery, not confirmatory).

For each operationalization x outcome:
  - Group test (Welch t / one-way ANOVA F)
  - Effect size (Cohen's d w/ bootstrap CI; eta^2 for ANOVA)
  - OLS with covariates (Age, Gender, log_n_messages, other-PHQ subscale)
  - FDR within family

Operationalizations:
  (A) PHQ-2 Depression Screen (>=3) and GAD-2 Anxiety Screen (>=3)
  (B) 4-tier PHQ-4 Total
  (C) PROMIS Anxiety binary (raw>=18)
  (D) Continuous comparison
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
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

BASE_DIR = Path(__file__).parent.parent
DEFAULT_LIWC = BASE_DIR / "results" / "260419T1358_phq4_liwc_analysis" / "00_merged_mh_liwc_data.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--liwc-merged", default=str(DEFAULT_LIWC))
ap.add_argument("--out-suffix", default="phq4_rq2_liwc_categorical")
args = ap.parse_args()

TS = datetime.now().strftime("%y%m%dT%H%M")
OUT = BASE_DIR / "results" / f"{TS}_{args.out_suffix}"
(OUT / "figures").mkdir(parents=True, exist_ok=True)


# Hypothesis-driven primary set (rationale in module docstring)
PRIMARY_FEATURES = {
    "i":           "I-words",
    "we":          "We-words",
    "they":        "They",
    "focusfuture": "Future Focus",
    "emo_neg":     "Negative Emotion",
    "affiliation": "Affiliation",
    "Social":      "Total Social",
}

# Full LIWC set
ALL_FEATURES = {
    "Analytic": "Analytical Thinking", "Clout": "Social Status", "Authentic": "Authenticity", "Tone": "Emotional Tone",
    "i": "I-words", "we": "We-words", "you": "You-words", "shehe": "She/He", "they": "They",
    "Affect": "Total Affect", "tone_pos": "Positive Tone", "tone_neg": "Negative Tone",
    "emo_pos": "Positive Emotion", "emo_neg": "Negative Emotion", "emo_anx": "Anxiety", "emo_anger": "Anger", "emo_sad": "Sadness",
    "Cognition": "Total Cognition", "cogproc": "Cognitive Processes", "insight": "Insight",
    "cause": "Causation", "discrep": "Discrepancy", "tentat": "Tentative", "certitude": "Certainty",
    "Drives": "Total Drives", "affiliation": "Affiliation", "achieve": "Achievement", "power": "Power",
    "Social": "Total Social", "socbehav": "Social Behavior", "prosocial": "Prosocial", "polite": "Polite", "conflict": "Conflict",
    "family": "Family", "friend": "Friend",
    "Perception": "Total Perception", "feeling": "Feeling",
    "focuspast": "Past Focus", "focuspresent": "Present Focus", "focusfuture": "Future Focus",
    "need": "Need", "want": "Want", "lack": "Lack", "risk": "Risk",
    "curiosity": "Curiosity", "reward": "Reward",
}

print(f"Reading LIWC merged: {args.liwc_merged}")
df = pd.read_csv(args.liwc_merged)
print(f"  N participants = {len(df)}")
print(f"  PROMIS available = {df['PROMIS_Anxiety'].notna().sum()}")


# Categorical operationalizations
def cat_total(x):
    if pd.isna(x): return np.nan
    if x <= 2: return 0
    if x <= 5: return 1
    if x <= 8: return 2
    return 3

df["PHQ4_Total"] = df["PHQ4_Anxiety"] + df["PHQ4_Depression"]
df["PHQ4_Total_Cat"] = df["PHQ4_Total"].apply(cat_total)
df["PHQ2_Dep_Pos"] = (df["PHQ4_Depression"] >= 3).astype(int)
df["GAD2_Anx_Pos"] = (df["PHQ4_Anxiety"] >= 3).astype(int)
df["PROMIS_Anx_Pos"] = (df["PROMIS_Anxiety"] >= 18).astype(int)
df.loc[df["PROMIS_Anxiety"].isna(), "PROMIS_Anx_Pos"] = np.nan

if "log_n_messages" not in df.columns:
    df["log_n_messages"] = np.log1p(df["n_messages"])

COVS = ["Age", "Gender_numeric", "log_n_messages"]


def cohens_d_with_ci(a, b, n_boot=1000, seed=42):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    psd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / psd if psd > 0 else np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ar = rng.choice(a, len(a), replace=True)
        br = rng.choice(b, len(b), replace=True)
        p = np.sqrt(((len(ar) - 1) * ar.var(ddof=1) + (len(br) - 1) * br.var(ddof=1)) / (len(ar) + len(br) - 2))
        boots.append((ar.mean() - br.mean()) / p if p > 0 else np.nan)
    boots = np.array(boots)
    return d, np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)


def benjamini_hochberg(pvals):
    return multipletests(pvals, alpha=0.05, method="fdr_bh")[1]


def sig(p):
    if pd.isna(p): return ""
    if p < .001: return "***"
    if p < .01: return "**"
    if p < .05: return "*"
    if p < .10: return "."
    return ""


def fit_ols(y, predictor, covariates, predictor_name="X"):
    X = pd.concat([predictor.rename(predictor_name), covariates], axis=1)
    X = sm.add_constant(X)
    mask = (~y.isna()) & (~X.isna().any(axis=1))
    if mask.sum() < len(X.columns) + 5:
        return None
    return sm.OLS(y[mask], X[mask]).fit()


# ============================================================
# Compute results for one (predictor_col, predictor_label, family) triple
# ============================================================

def run_binary_family(df, predictor_col, predictor_label, features, other_subscale_col=None):
    rows = []
    for feat, feat_label in features.items():
        if feat not in df.columns:
            continue
        sub = df.dropna(subset=[predictor_col])
        pos = sub.loc[sub[predictor_col] == 1, feat].dropna()
        neg = sub.loc[sub[predictor_col] == 0, feat].dropna()
        if len(pos) < 5 or len(neg) < 5:
            continue
        t, p = stats.ttest_ind(pos, neg, equal_var=False)
        d, dlo, dhi = cohens_d_with_ci(pos.values, neg.values, n_boot=500)
        u, p_u = stats.mannwhitneyu(pos, neg, alternative="two-sided")

        covs_cols = COVS + ([other_subscale_col] if other_subscale_col and other_subscale_col in sub.columns else [])
        ols = fit_ols(sub[feat], sub[predictor_col].astype(float), sub[covs_cols], predictor_name="binary")
        rows.append({
            "Predictor": predictor_label,
            "Feature": feat,
            "Label": feat_label,
            "N_pos": len(pos), "N_neg": len(neg),
            "M_pos": pos.mean(), "M_neg": neg.mean(),
            "SD_pos": pos.std(), "SD_neg": neg.std(),
            "t": t, "p_t": p,
            "d": d, "d_CI_low": dlo, "d_CI_high": dhi,
            "MWU_p": p_u,
            "ols_beta": ols.params.get("binary", np.nan) if ols is not None else np.nan,
            "ols_p":    ols.pvalues.get("binary", np.nan) if ols is not None else np.nan,
            "ols_n":    int(ols.nobs) if ols is not None else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_t_FDR"]   = benjamini_hochberg(out["p_t"].values)
        out["p_ols_FDR"] = benjamini_hochberg(out["ols_p"].values)
        out["sig_t_FDR"] = out["p_t_FDR"].apply(sig)
        out["sig_ols_FDR"] = out["p_ols_FDR"].apply(sig)
    return out


def run_tier_family(df, features):
    rows = []
    for feat, feat_label in features.items():
        if feat not in df.columns:
            continue
        groups = [df.loc[df["PHQ4_Total_Cat"] == k, feat].dropna().values for k in range(4)]
        if any(len(g) < 3 for g in groups):
            continue
        f, p_f = stats.f_oneway(*groups)
        df_b, df_w = 3, sum(len(g) for g in groups) - 4
        eta2 = (f * df_b) / (f * df_b + df_w) if (f * df_b + df_w) > 0 else np.nan
        # Linear trend: pearson r between tier (0..3) and feature
        cat_n = df["PHQ4_Total_Cat"].astype(float)
        mask = (~cat_n.isna()) & (~df[feat].isna())
        r_trend, p_trend = stats.pearsonr(cat_n[mask], df[feat][mask])
        # Kruskal-Wallis
        kw, p_kw = stats.kruskal(*groups)
        # OLS with covariates
        ols = fit_ols(df[feat], cat_n, df[COVS], predictor_name="tier")
        rows.append({
            "Feature": feat, "Label": feat_label,
            "N": sum(len(g) for g in groups),
            "M_None": groups[0].mean(), "M_Mild": groups[1].mean(),
            "M_Mod": groups[2].mean(), "M_Sev": groups[3].mean(),
            "F": f, "p_F": p_f, "eta2": eta2,
            "r_trend": r_trend, "p_trend": p_trend,
            "KW_H": kw, "p_KW": p_kw,
            "ols_beta": ols.params.get("tier", np.nan) if ols is not None else np.nan,
            "ols_p":    ols.pvalues.get("tier", np.nan) if ols is not None else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out):
        for c in ["p_F", "p_trend", "ols_p"]:
            out[f"{c}_FDR"] = benjamini_hochberg(out[c].values)
            out[f"sig_{c}_FDR"] = out[f"{c}_FDR"].apply(sig)
    return out


# ============================================================
# Run primary (hypothesis-driven) analyses
# ============================================================

print("\n" + "=" * 60)
print("PRIMARY (hypothesis-driven, 7 features)")
print("=" * 60)

primary_dep = run_binary_family(df, "PHQ2_Dep_Pos", "PHQ-2 Depression+", PRIMARY_FEATURES, "PHQ4_Anxiety")
primary_anx = run_binary_family(df, "GAD2_Anx_Pos", "GAD-2 Anxiety+",    PRIMARY_FEATURES, "PHQ4_Depression")
# Combined FDR within (predictor x feature) family of 14 tests
primary = pd.concat([primary_dep, primary_anx], ignore_index=True)
primary["p_t_FDR_combined"]   = benjamini_hochberg(primary["p_t"].values)
primary["p_ols_FDR_combined"] = benjamini_hochberg(primary["ols_p"].values)
primary["sig_t_FDR_combined"]   = primary["p_t_FDR_combined"].apply(sig)
primary["sig_ols_FDR_combined"] = primary["p_ols_FDR_combined"].apply(sig)
primary.to_csv(OUT / "01_primary_binary_screen.csv", index=False)
print()
print(primary[["Predictor", "Label", "N_pos", "N_neg", "M_pos", "M_neg", "d", "p_t", "p_t_FDR_combined", "sig_t_FDR_combined", "ols_p", "p_ols_FDR_combined", "sig_ols_FDR_combined"]].round(3).to_string(index=False))

# Primary tier (4-tier total)
print("\n--- Primary 4-tier PHQ-4 total ---")
primary_tier = run_tier_family(df, PRIMARY_FEATURES)
primary_tier.to_csv(OUT / "02_primary_tier_total.csv", index=False)
print()
print(primary_tier[["Label", "M_None", "M_Mild", "M_Mod", "M_Sev", "F", "p_F", "p_F_FDR", "sig_p_F_FDR", "eta2", "r_trend", "p_trend_FDR", "ols_p", "ols_p_FDR"]].round(3).to_string(index=False))


# ============================================================
# Run exploratory (full 46 LIWC) analyses
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY (full LIWC, FDR over all 46 features)")
print("=" * 60)

expl_dep = run_binary_family(df, "PHQ2_Dep_Pos", "PHQ-2 Depression+", ALL_FEATURES, "PHQ4_Anxiety")
expl_anx = run_binary_family(df, "GAD2_Anx_Pos", "GAD-2 Anxiety+",    ALL_FEATURES, "PHQ4_Depression")
expl_dep.to_csv(OUT / "03_exploratory_dep_binary.csv", index=False)
expl_anx.to_csv(OUT / "04_exploratory_anx_binary.csv", index=False)

print("\n--- Top-10 depression+ effects (by |d|, FDR within family of 46) ---")
print(expl_dep.reindex(expl_dep["d"].abs().sort_values(ascending=False).index)[["Label", "M_pos", "M_neg", "d", "p_t", "p_t_FDR", "sig_t_FDR", "ols_p", "p_ols_FDR"]].head(10).round(3).to_string(index=False))

print("\n--- Top-10 anxiety+ effects (by |d|, FDR within family of 46) ---")
print(expl_anx.reindex(expl_anx["d"].abs().sort_values(ascending=False).index)[["Label", "M_pos", "M_neg", "d", "p_t", "p_t_FDR", "sig_t_FDR", "ols_p", "p_ols_FDR"]].head(10).round(3).to_string(index=False))


# Tier exploratory
expl_tier = run_tier_family(df, ALL_FEATURES)
expl_tier.to_csv(OUT / "05_exploratory_tier_total.csv", index=False)


# ============================================================
# PROMIS robustness
# ============================================================

print("\n" + "=" * 60)
print("PROMIS BINARY ROBUSTNESS  (LIWC subset)")
print("=" * 60)

n_promis_liwc = df["PROMIS_Anxiety"].notna().sum()
print(f"PROMIS available in LIWC subset: N={n_promis_liwc}")
if n_promis_liwc >= 30:
    promis_primary = run_binary_family(df, "PROMIS_Anx_Pos", "PROMIS Anxiety+", PRIMARY_FEATURES, "PHQ4_Depression")
    promis_primary.to_csv(OUT / "06_promis_primary_binary.csv", index=False)
    print()
    print(promis_primary[["Label", "N_pos", "N_neg", "M_pos", "M_neg", "d", "p_t", "p_t_FDR", "sig_t_FDR", "ols_p", "p_ols_FDR"]].round(3).to_string(index=False))


# ============================================================
# Comparison: continuous vs categorical (depression only, primary features)
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON: continuous PHQ-4 dep r  vs  binary screen d")
print("=" * 60)

cmp_rows = []
for feat, feat_label in PRIMARY_FEATURES.items():
    if feat not in df.columns:
        continue
    sub_cmp = df.dropna(subset=["PHQ4_Depression", feat])
    r, p_r = stats.pearsonr(sub_cmp["PHQ4_Depression"], sub_cmp[feat])
    pos = df.loc[df["PHQ2_Dep_Pos"] == 1, feat].dropna()
    neg = df.loc[df["PHQ2_Dep_Pos"] == 0, feat].dropna()
    t, p_t = stats.ttest_ind(pos, neg, equal_var=False)
    d, _, _ = cohens_d_with_ci(pos.values, neg.values, n_boot=500)
    cmp_rows.append({"Feature": feat_label, "Cont_r": r, "Cont_p": p_r,
                     "Bin_d": d, "Bin_p": p_t})
cmp = pd.DataFrame(cmp_rows)
cmp["Cont_p_FDR"] = benjamini_hochberg(cmp["Cont_p"].values)
cmp["Bin_p_FDR"] = benjamini_hochberg(cmp["Bin_p"].values)
cmp.to_csv(OUT / "07_continuous_vs_binary_comparison.csv", index=False)
print()
print(cmp.round(3).to_string(index=False))


# ============================================================
# Figures
# ============================================================

# Forest of d-values, primary features, depression
fig, ax = plt.subplots(figsize=(8, 5))
sub = primary_dep.sort_values("d", ascending=True)
y = np.arange(len(sub))
ax.errorbar(sub["d"], y, xerr=[sub["d"] - sub["d_CI_low"], sub["d_CI_high"] - sub["d"]],
            fmt="o", color="#d77e7e", capsize=4, markersize=8)
ax.axvline(0, color="grey", linestyle="--", linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(sub["Label"])
ax.set_xlabel("Cohen's $d$ (PHQ-2 Depression+ vs Depression-)")
ax.set_title("Depression-Positive vs Negative on Primary LIWC Features\n95% bootstrap CIs")
plt.tight_layout()
plt.savefig(OUT / "figures" / "fig_primary_d_forest_dep.png", dpi=150, bbox_inches="tight")
plt.close()

# Forest for anxiety
fig, ax = plt.subplots(figsize=(8, 5))
sub = primary_anx.sort_values("d", ascending=True)
y = np.arange(len(sub))
ax.errorbar(sub["d"], y, xerr=[sub["d"] - sub["d_CI_low"], sub["d_CI_high"] - sub["d"]],
            fmt="o", color="#7ea6cf", capsize=4, markersize=8)
ax.axvline(0, color="grey", linestyle="--", linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(sub["Label"])
ax.set_xlabel("Cohen's $d$ (GAD-2 Anxiety+ vs Anxiety-)")
ax.set_title("Anxiety-Positive vs Negative on Primary LIWC Features\n95% bootstrap CIs")
plt.tight_layout()
plt.savefig(OUT / "figures" / "fig_primary_d_forest_anx.png", dpi=150, bbox_inches="tight")
plt.close()


manifest = {
    "timestamp": TS,
    "input": str(args.liwc_merged),
    "N_LIWC": int(len(df)),
    "N_PROMIS_in_LIWC": int(n_promis_liwc),
    "PHQ-2 Dep+ N": int(df["PHQ2_Dep_Pos"].sum()),
    "GAD-2 Anx+ N": int(df["GAD2_Anx_Pos"].sum()),
    "tier_dist": df["PHQ4_Total_Cat"].value_counts().sort_index().to_dict(),
    "primary_features": list(PRIMARY_FEATURES.keys()),
    "files": [
        "01_primary_binary_screen.csv",
        "02_primary_tier_total.csv",
        "03_exploratory_dep_binary.csv",
        "04_exploratory_anx_binary.csv",
        "05_exploratory_tier_total.csv",
        "06_promis_primary_binary.csv",
        "07_continuous_vs_binary_comparison.csv",
        "figures/fig_primary_d_forest_dep.png",
        "figures/fig_primary_d_forest_anx.png",
    ],
}
with open(OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print(f"\nDone. Output: {OUT}")
