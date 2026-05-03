#!/usr/bin/env python3
"""RQ1 (categorical): clinically meaningful tiers of PHQ-4 / PROMIS distress
predicting ChatGPT-experience composites.

Operationalizations (all reported, primary = binary screen):
  (A) Binary subscale screen (PHQ-2 dep >=3, GAD-2 anx >=3) -- Kroenke 2009
  (B) 4-tier total PHQ-4 (0-2 none / 3-5 mild / 6-8 mod / 9-12 severe) -- Loewe 2010
  (C) PROMIS Anxiety binary (raw >=18, approx T>=55 elevated) -- HealthMeasures
  (D) Continuous (baseline for comparison)

For each operationalization x outcome:
  - Group test (t / Welch / ANOVA F)
  - Effect size (Cohen's d with bootstrap CI; eta^2 for ANOVA)
  - OLS with categorical predictor + covariates (age, gender, freq, other-PHQ subscale)
  - Linear trend across ordered tiers (4-tier only)
  - Benjamini-Hochberg FDR within each predictor x outcome family

Reads the latest merged analysis CSV from the existing pipeline (so we
don't re-do the data join). Writes a timestamped folder under results/.
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
DEFAULT_MERGED = BASE_DIR / "results" / "260419T1358_phq4_ai_controlled_regression" / "00_merged_analysis_data.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--merged", default=str(DEFAULT_MERGED))
ap.add_argument("--out-suffix", default="phq4_rq1_categorical")
args = ap.parse_args()

TS = datetime.now().strftime("%y%m%dT%H%M")
OUT = BASE_DIR / "results" / f"{TS}_{args.out_suffix}"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

OUTCOMES = {
    "Emotional_Engagement_Comp": "Emotional Engagement",
    "Trust_Single": "Trust",
    "Behavior_Change_Comp": "Behavioral Change",
    "Self_Efficacy_Comp": "Self-Efficacy",
    "Dependency_Single": "Dependency Concern",
}

COVARIATES = ["Age", "Gender_imputed", "Frequency_numeric"]

print(f"Reading merged data: {args.merged}")
df = pd.read_csv(args.merged)
print(f"  N = {len(df)}")
print(f"  PROMIS available: {df['PROMIS_Anxiety'].notna().sum()}")


# --- Operationalizations ---
def cat_total(x):
    if pd.isna(x):
        return np.nan
    if x <= 2: return 0  # None
    if x <= 5: return 1  # Mild
    if x <= 8: return 2  # Moderate
    return 3              # Severe

df["PHQ4_Total"] = df["PHQ4_Anxiety"] + df["PHQ4_Depression"]
df["PHQ4_Total_Cat"] = df["PHQ4_Total"].apply(cat_total)
df["PHQ2_Dep_Pos"] = (df["PHQ4_Depression"] >= 3).astype(int)
df["GAD2_Anx_Pos"] = (df["PHQ4_Anxiety"] >= 3).astype(int)
# PROMIS Anxiety SF7a: raw 7-35; T~50 maps to raw ~13, T~55 to raw ~18
# (HealthMeasures conversion table). We use raw>=18 as 'elevated' cutoff.
df["PROMIS_Anx_Pos"] = (df["PROMIS_Anxiety"] >= 18).astype(int)
df.loc[df["PROMIS_Anxiety"].isna(), "PROMIS_Anx_Pos"] = np.nan


# ============================================================
# Helper functions
# ============================================================

def cohens_d_with_ci(a, b, n_boot=2000, seed=42):
    """Cohen's d for independent samples with bootstrap 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ar = rng.choice(a, len(a), replace=True)
        br = rng.choice(b, len(b), replace=True)
        psd = np.sqrt(((len(ar) - 1) * ar.var(ddof=1) + (len(br) - 1) * br.var(ddof=1)) / (len(ar) + len(br) - 2))
        boots.append((ar.mean() - br.mean()) / psd if psd > 0 else np.nan)
    boots = np.array(boots)
    ci_low, ci_high = np.nanpercentile(boots, [2.5, 97.5])
    return d, ci_low, ci_high


def eta_squared(f, df_between, df_within):
    """Eta-squared for one-way ANOVA from F statistic."""
    return (f * df_between) / (f * df_between + df_within)


def benjamini_hochberg(pvals):
    return multipletests(pvals, alpha=0.05, method="fdr_bh")[1]


def sig(p):
    if pd.isna(p): return ""
    if p < .001: return "***"
    if p < .01: return "**"
    if p < .05: return "*"
    if p < .10: return "."
    return ""


def fit_ols_with_covariates(y, predictor, covariates, predictor_name="X"):
    """OLS regression with covariates and bonus stats. Returns dict of beta, SE, t, p, partial eta-sq."""
    X = pd.concat([predictor, covariates], axis=1)
    X.columns = [predictor_name] + list(covariates.columns)
    X = sm.add_constant(X)
    mask = (~y.isna()) & (~X.isna().any(axis=1))
    y2, X2 = y[mask], X[mask]
    if len(y2) < len(X2.columns) + 5:
        return None
    model = sm.OLS(y2, X2).fit()
    return {
        "n": int(model.nobs),
        "beta": model.params.get(predictor_name, np.nan),
        "se": model.bse.get(predictor_name, np.nan),
        "t": model.tvalues.get(predictor_name, np.nan),
        "p": model.pvalues.get(predictor_name, np.nan),
        "ci_low": model.conf_int().loc[predictor_name, 0] if predictor_name in model.params.index else np.nan,
        "ci_high": model.conf_int().loc[predictor_name, 1] if predictor_name in model.params.index else np.nan,
        "r2": model.rsquared,
        "r2_adj": model.rsquared_adj,
    }


# ============================================================
# (A) BINARY SUBSCALE SCREEN
# ============================================================

print("\n" + "=" * 60)
print("(A) BINARY SUBSCALE SCREEN  (PHQ-2 dep >=3, GAD-2 anx >=3)")
print("=" * 60)

binary_results = []
for predictor_name, predictor_col, label in [
    ("PHQ-2 Depression Screen", "PHQ2_Dep_Pos", "Dep+"),
    ("GAD-2 Anxiety Screen", "GAD2_Anx_Pos", "Anx+"),
]:
    print(f"\n{predictor_name}: positive N={int(df[predictor_col].sum())} / total {len(df)}")
    for outcome_col, outcome_label in OUTCOMES.items():
        pos = df.loc[df[predictor_col] == 1, outcome_col].dropna()
        neg = df.loc[df[predictor_col] == 0, outcome_col].dropna()
        # Welch's t (heteroscedastic)
        t, p_t = stats.ttest_ind(pos, neg, equal_var=False)
        d, d_lo, d_hi = cohens_d_with_ci(pos.values, neg.values)
        # Mann-Whitney U (nonparametric robustness)
        u, p_u = stats.mannwhitneyu(pos, neg, alternative="two-sided")

        # OLS with covariates: outcome ~ binary + age + gender + freq + (other PHQ subscale)
        other_subscale = "PHQ4_Anxiety" if predictor_col == "PHQ2_Dep_Pos" else "PHQ4_Depression"
        covs = df[COVARIATES + [other_subscale]]
        ols = fit_ols_with_covariates(df[outcome_col], df[predictor_col], covs, predictor_name="binary")

        binary_results.append({
            "Predictor": predictor_name,
            "Outcome": outcome_label,
            "N_pos": len(pos), "N_neg": len(neg),
            "M_pos": pos.mean(), "M_neg": neg.mean(),
            "SD_pos": pos.std(), "SD_neg": neg.std(),
            "t": t, "p_t": p_t,
            "Cohens_d": d, "d_CI_low": d_lo, "d_CI_high": d_hi,
            "MannWhitneyU": u, "p_U": p_u,
            "ols_beta": ols["beta"] if ols else np.nan,
            "ols_se":   ols["se"] if ols else np.nan,
            "ols_t":    ols["t"]  if ols else np.nan,
            "ols_p":    ols["p"]  if ols else np.nan,
            "ols_n":    ols["n"]  if ols else np.nan,
        })

binary_df = pd.DataFrame(binary_results)
# FDR within each predictor x outcome family (5 outcomes per predictor)
for pname in binary_df["Predictor"].unique():
    mask = binary_df["Predictor"] == pname
    binary_df.loc[mask, "p_t_FDR"] = benjamini_hochberg(binary_df.loc[mask, "p_t"].values)
    binary_df.loc[mask, "p_ols_FDR"] = benjamini_hochberg(binary_df.loc[mask, "ols_p"].values)
    binary_df.loc[mask, "sig_t_FDR"] = binary_df.loc[mask, "p_t_FDR"].apply(sig)
    binary_df.loc[mask, "sig_ols_FDR"] = binary_df.loc[mask, "p_ols_FDR"].apply(sig)

binary_df.to_csv(OUT / "01_binary_screen.csv", index=False)
print()
print(binary_df[["Predictor", "Outcome", "N_pos", "N_neg", "M_pos", "M_neg", "Cohens_d", "p_t", "p_t_FDR", "sig_t_FDR", "ols_p", "p_ols_FDR", "sig_ols_FDR"]].round(3).to_string(index=False))


# ============================================================
# (B) 4-TIER TOTAL PHQ-4
# ============================================================

print("\n" + "=" * 60)
print("(B) 4-TIER TOTAL PHQ-4  (Loewe 2010: 0-2/3-5/6-8/9-12)")
print("=" * 60)

cat_labels = ["0_None", "1_Mild", "2_Moderate", "3_Severe"]
print(f"\nDistribution: {df['PHQ4_Total_Cat'].value_counts().sort_index().to_dict()}")

tier_results = []
for outcome_col, outcome_label in OUTCOMES.items():
    groups = [df.loc[df["PHQ4_Total_Cat"] == k, outcome_col].dropna().values for k in range(4)]
    f, p_f = stats.f_oneway(*groups)
    df_b = 3
    df_w = sum(len(g) for g in groups) - 4
    eta2 = eta_squared(f, df_b, df_w)
    means = [g.mean() for g in groups]
    sds = [g.std() for g in groups]
    Ns = [len(g) for g in groups]

    # Linear trend test (treats tier as ordered numeric)
    cat_numeric = df["PHQ4_Total_Cat"].astype(float)
    mask = (~cat_numeric.isna()) & (~df[outcome_col].isna())
    r_trend, p_trend = stats.pearsonr(cat_numeric[mask], df[outcome_col][mask])

    # Kruskal-Wallis (nonparametric robustness)
    kw_h, p_kw = stats.kruskal(*groups)

    # OLS with ordinal tier as continuous predictor + covariates
    ols = fit_ols_with_covariates(df[outcome_col], cat_numeric, df[COVARIATES], predictor_name="tier")

    tier_results.append({
        "Outcome": outcome_label,
        "N_None": Ns[0], "N_Mild": Ns[1], "N_Mod": Ns[2], "N_Sev": Ns[3],
        "M_None": means[0], "M_Mild": means[1], "M_Mod": means[2], "M_Sev": means[3],
        "F": f, "p_F": p_f, "eta2": eta2,
        "r_trend": r_trend, "p_trend": p_trend,
        "KW_H": kw_h, "p_KW": p_kw,
        "ols_trend_beta": ols["beta"] if ols else np.nan,
        "ols_trend_p":    ols["p"]    if ols else np.nan,
        "ols_n":          ols["n"]    if ols else np.nan,
    })

tier_df = pd.DataFrame(tier_results)
for col_pf in ["p_F", "p_trend", "ols_trend_p"]:
    tier_df[f"{col_pf}_FDR"] = benjamini_hochberg(tier_df[col_pf].values)
    tier_df[f"sig_{col_pf}_FDR"] = tier_df[f"{col_pf}_FDR"].apply(sig)

tier_df.to_csv(OUT / "02_tier_total_phq4.csv", index=False)
print()
print(tier_df[["Outcome", "M_None", "M_Mild", "M_Mod", "M_Sev", "F", "p_F", "p_F_FDR", "sig_p_F_FDR", "eta2", "r_trend", "p_trend_FDR", "ols_trend_beta", "ols_trend_p", "ols_trend_p_FDR"]].round(3).to_string(index=False))


# ============================================================
# (C) PROMIS BINARY (robustness for anxiety)
# ============================================================

print("\n" + "=" * 60)
print("(C) PROMIS ANXIETY BINARY  (raw >=18, ~T>=55)")
print("=" * 60)

n_promis = df["PROMIS_Anxiety"].notna().sum()
n_pos = int(df["PROMIS_Anx_Pos"].sum())
print(f"\nPROMIS available N={n_promis}, elevated N={n_pos}")

promis_results = []
for outcome_col, outcome_label in OUTCOMES.items():
    sub = df.dropna(subset=["PROMIS_Anx_Pos"])
    pos = sub.loc[sub["PROMIS_Anx_Pos"] == 1, outcome_col].dropna()
    neg = sub.loc[sub["PROMIS_Anx_Pos"] == 0, outcome_col].dropna()
    t, p_t = stats.ttest_ind(pos, neg, equal_var=False)
    d, d_lo, d_hi = cohens_d_with_ci(pos.values, neg.values)
    u, p_u = stats.mannwhitneyu(pos, neg, alternative="two-sided")

    # OLS with covariates + PHQ-4 depression as additional control
    covs = sub[COVARIATES + ["PHQ4_Depression"]]
    ols = fit_ols_with_covariates(sub[outcome_col], sub["PROMIS_Anx_Pos"], covs, predictor_name="promis_pos")

    # PROMIS continuous OLS too
    ols_cont = fit_ols_with_covariates(sub[outcome_col], sub["PROMIS_Anxiety"], covs, predictor_name="promis_cont")

    promis_results.append({
        "Outcome": outcome_label,
        "N_pos": len(pos), "N_neg": len(neg),
        "M_pos": pos.mean(), "M_neg": neg.mean(),
        "t": t, "p_t": p_t,
        "Cohens_d": d, "d_CI_low": d_lo, "d_CI_high": d_hi,
        "MWU_p": p_u,
        "ols_pos_beta": ols["beta"] if ols else np.nan,
        "ols_pos_p":    ols["p"]    if ols else np.nan,
        "ols_cont_beta": ols_cont["beta"] if ols_cont else np.nan,
        "ols_cont_p":    ols_cont["p"]    if ols_cont else np.nan,
    })

promis_df = pd.DataFrame(promis_results)
promis_df["p_t_FDR"] = benjamini_hochberg(promis_df["p_t"].values)
promis_df["sig_t_FDR"] = promis_df["p_t_FDR"].apply(sig)
promis_df["ols_pos_p_FDR"] = benjamini_hochberg(promis_df["ols_pos_p"].values)
promis_df["sig_ols_pos_FDR"] = promis_df["ols_pos_p_FDR"].apply(sig)
promis_df.to_csv(OUT / "03_promis_binary.csv", index=False)
print()
print(promis_df[["Outcome", "N_pos", "N_neg", "M_pos", "M_neg", "Cohens_d", "p_t", "p_t_FDR", "sig_t_FDR", "ols_pos_beta", "ols_pos_p", "ols_pos_p_FDR", "sig_ols_pos_FDR"]].round(3).to_string(index=False))


# ============================================================
# (D) CUTOFF SENSITIVITY (binary screen)
# ============================================================

print("\n" + "=" * 60)
print("(D) CUTOFF SENSITIVITY  (subscale binary at >=2, >=3, >=4)")
print("=" * 60)

cutoff_results = []
for cutoff in [2, 3, 4]:
    for predictor_col, label in [("PHQ4_Depression", "Depression"), ("PHQ4_Anxiety", "Anxiety")]:
        positive = (df[predictor_col] >= cutoff).astype(int)
        for outcome_col, outcome_label in OUTCOMES.items():
            pos = df.loc[positive == 1, outcome_col].dropna()
            neg = df.loc[positive == 0, outcome_col].dropna()
            if len(pos) < 5 or len(neg) < 5:
                continue
            t, p = stats.ttest_ind(pos, neg, equal_var=False)
            d, d_lo, d_hi = cohens_d_with_ci(pos.values, neg.values, n_boot=500)
            cutoff_results.append({
                "Cutoff": f">={cutoff}",
                "Predictor": label,
                "Outcome": outcome_label,
                "N_pos": len(pos), "N_neg": len(neg),
                "Cohens_d": d, "p": p,
            })

cutoff_df = pd.DataFrame(cutoff_results)
cutoff_df.to_csv(OUT / "04_cutoff_sensitivity.csv", index=False)
print()
pivot = cutoff_df.pivot_table(index=["Predictor", "Outcome"], columns="Cutoff", values=["Cohens_d", "p"]).round(3)
print(pivot.to_string())


# ============================================================
# Comparison summary table: continuous (orig) vs binary screen
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY: continuous (current paper) vs categorical (proposed)")
print("=" * 60)

summary = []
for outcome_col, outcome_label in OUTCOMES.items():
    # continuous
    r_anx, p_anx = stats.pearsonr(df["PHQ4_Anxiety"], df[outcome_col])
    r_dep, p_dep = stats.pearsonr(df["PHQ4_Depression"], df[outcome_col])
    # binary depression t
    pos = df.loc[df["PHQ2_Dep_Pos"] == 1, outcome_col]
    neg = df.loc[df["PHQ2_Dep_Pos"] == 0, outcome_col]
    t_dep, p_dep_b = stats.ttest_ind(pos, neg, equal_var=False)
    d_dep, _, _ = cohens_d_with_ci(pos.values, neg.values, n_boot=500)
    # binary anxiety t
    pos = df.loc[df["GAD2_Anx_Pos"] == 1, outcome_col]
    neg = df.loc[df["GAD2_Anx_Pos"] == 0, outcome_col]
    t_anx, p_anx_b = stats.ttest_ind(pos, neg, equal_var=False)
    d_anx, _, _ = cohens_d_with_ci(pos.values, neg.values, n_boot=500)
    summary.append({
        "Outcome": outcome_label,
        "Cont_r_Anx": r_anx, "Cont_p_Anx": p_anx,
        "Cont_r_Dep": r_dep, "Cont_p_Dep": p_dep,
        "Bin_d_Dep+": d_dep, "Bin_p_Dep+": p_dep_b,
        "Bin_d_Anx+": d_anx, "Bin_p_Anx+": p_anx_b,
    })
sumdf = pd.DataFrame(summary)
# FDR within each column family
for col in ["Cont_p_Anx", "Cont_p_Dep", "Bin_p_Dep+", "Bin_p_Anx+"]:
    sumdf[f"{col}_FDR"] = benjamini_hochberg(sumdf[col].values)
sumdf.to_csv(OUT / "05_summary_continuous_vs_binary.csv", index=False)
print()
print(sumdf.round(3).to_string(index=False))


# ============================================================
# Figure: depression-positive vs negative bar plot for all outcomes
# ============================================================

fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
for i, (outcome_col, outcome_label) in enumerate(OUTCOMES.items()):
    ax = axes[i]
    pos = df.loc[df["PHQ2_Dep_Pos"] == 1, outcome_col].dropna()
    neg = df.loc[df["PHQ2_Dep_Pos"] == 0, outcome_col].dropna()
    means = [neg.mean(), pos.mean()]
    sems = [neg.std() / np.sqrt(len(neg)), pos.std() / np.sqrt(len(pos))]
    bars = ax.bar(["Dep-", "Dep+"], means, yerr=sems, capsize=5,
                  color=["#7ea6cf", "#d77e7e"], edgecolor="black")
    t, p = stats.ttest_ind(pos, neg, equal_var=False)
    d, _, _ = cohens_d_with_ci(pos.values, neg.values, n_boot=500)
    ax.set_title(f"{outcome_label}\nd={d:.2f}, p={p:.3f}", fontsize=10)
    ax.set_ylim(1, 5)
plt.tight_layout()
plt.savefig(OUT / "figures" / "fig_binary_dep_vs_outcomes.png", dpi=150, bbox_inches="tight")
plt.close()


# 4-tier total figure
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
for i, (outcome_col, outcome_label) in enumerate(OUTCOMES.items()):
    ax = axes[i]
    means = [df.loc[df["PHQ4_Total_Cat"] == k, outcome_col].mean() for k in range(4)]
    sems = [df.loc[df["PHQ4_Total_Cat"] == k, outcome_col].std() / np.sqrt(max((df["PHQ4_Total_Cat"] == k).sum(), 1)) for k in range(4)]
    ax.bar(["None", "Mild", "Mod", "Sev"], means, yerr=sems, capsize=5,
           color=["#7ea6cf", "#a8c0d4", "#cf9d7e", "#d77e7e"], edgecolor="black")
    f, p = stats.f_oneway(*[df.loc[df["PHQ4_Total_Cat"] == k, outcome_col].dropna() for k in range(4)])
    df_b, df_w = 3, len(df) - 4
    eta2 = eta_squared(f, df_b, df_w)
    ax.set_title(f"{outcome_label}\n$\\eta^2$={eta2:.3f}, p={p:.3f}", fontsize=10)
    ax.set_ylim(1, 5)
plt.tight_layout()
plt.savefig(OUT / "figures" / "fig_tier_total_vs_outcomes.png", dpi=150, bbox_inches="tight")
plt.close()


# Save manifest
manifest = {
    "timestamp": TS,
    "merged_input": str(args.merged),
    "N_total": int(len(df)),
    "N_PROMIS": int(n_promis),
    "PHQ-2 Dep+ (>=3)": int(df["PHQ2_Dep_Pos"].sum()),
    "GAD-2 Anx+ (>=3)": int(df["GAD2_Anx_Pos"].sum()),
    "PROMIS Anx+ (raw>=18)": int(df["PROMIS_Anx_Pos"].dropna().sum()),
    "Tier_distribution": df["PHQ4_Total_Cat"].value_counts().sort_index().to_dict(),
    "files": [
        "01_binary_screen.csv",
        "02_tier_total_phq4.csv",
        "03_promis_binary.csv",
        "04_cutoff_sensitivity.csv",
        "05_summary_continuous_vs_binary.csv",
        "figures/fig_binary_dep_vs_outcomes.png",
        "figures/fig_tier_total_vs_outcomes.png",
    ],
}
with open(OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)

print(f"\nDone. Output: {OUT}")
