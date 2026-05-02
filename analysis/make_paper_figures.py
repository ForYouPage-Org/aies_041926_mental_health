#!/usr/bin/env python3
"""Generate paper-parallel Figure 1 (regression forest) + Figure 2 (LIWC scatter)
for PHQ-4 anxiety + depression, mirroring Fyorin et al. (2026).

Reads latest PHQ-4 regression + LIWC result directories and saves PNG + PDF
into figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BASE = Path(__file__).parent.parent
RESULTS = BASE / "results"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def latest(pattern: str) -> Path:
    dirs = sorted(RESULTS.glob(pattern))
    if not dirs:
        sys.exit(f"No directory matching {pattern!r}")
    return dirs[-1]


REG_DIR = latest("*_phq4_ai_controlled_regression")
LIWC_DIR = latest("*_phq4_liwc_analysis")
print(f"Regression: {REG_DIR.name}")
print(f"LIWC:       {LIWC_DIR.name}")

# ---------------------------------------------------------------------------
# Figure 1: Forest plot of standardized betas (Anxiety vs. Depression × 5 outcomes)
# ---------------------------------------------------------------------------
eff = pd.read_csv(REG_DIR / "05_anxiety_depression_effects_controlled.csv")

# Compute SE and 95% CI from beta / t
eff["SE_Anx"] = (eff["Beta_Anxiety"] / eff["t_Anxiety"]).abs()
eff["CI_Anx"] = 1.96 * eff["SE_Anx"]
eff["SE_Dep"] = (eff["Beta_Depression"] / eff["t_Depression"]).abs()
eff["CI_Dep"] = 1.96 * eff["SE_Dep"]

# Order outcomes as in paper Fig 1
order = ["Emotional Engagement (Q4-6)", "Trust (Q1)", "Behavioral Change (Q10-12)",
         "Dependency Concern (Q13)", "Self-Efficacy (Q7-9)"]
eff["__ord"] = eff["AI_Attitude"].apply(lambda x: order.index(x) if x in order else 99)
eff = eff.sort_values("__ord").reset_index(drop=True)

labels_display = {
    "Emotional Engagement (Q4-6)": "Emotional Engagement",
    "Trust (Q1)": "Trust",
    "Behavioral Change (Q10-12)": "Behavioral Change",
    "Dependency Concern (Q13)": "Dependency Concern",
    "Self-Efficacy (Q7-9)": "Self-Efficacy",
}


def sig_tier(p_raw, p_fdr):
    """3-tier significance: 'fdr' | 'raw' | 'ns'."""
    if pd.notna(p_fdr) and p_fdr < 0.05: return "fdr"
    if pd.notna(p_raw) and p_raw < 0.05: return "raw"
    return "ns"


def stars_fdr(p_fdr):
    if pd.isna(p_fdr): return ""
    if p_fdr < 0.001: return "***"
    if p_fdr < 0.01:  return "**"
    if p_fdr < 0.05:  return "*"
    return ""


def p_label(p_raw, p_fdr, tier):
    """Annotation next to each point — shows FDR stars OR raw p† for marginal effects."""
    if tier == "fdr":
        return stars_fdr(p_fdr)
    if tier == "raw":
        # Use dagger to mark "uncorrected only" - standard in meta-analytic forest plots
        if p_raw < 0.01: return "††"
        return "†"
    return ""


fig, ax = plt.subplots(figsize=(10.5, 6.2))
y = np.arange(len(eff))
offset = 0.20
RED = "#c44e52"
BLUE = "#4c72b0"

# Visual tiers
STYLE = {
    "fdr": dict(alpha=1.0, mfc_mode="solid", edge_lw=0, ebar_lw=2.0),
    "raw": dict(alpha=0.95, mfc_mode="hollow", edge_lw=1.8, ebar_lw=1.6),
    "ns":  dict(alpha=0.35, mfc_mode="solid", edge_lw=0, ebar_lw=1.2),
}


def draw_point(ax, x, y, ci, color, marker, tier, size):
    st = STYLE[tier]
    mfc = "white" if st["mfc_mode"] == "hollow" else color
    ax.errorbar(x, y, xerr=ci,
                fmt=marker, markersize=size,
                markerfacecolor=mfc,
                markeredgecolor=color,
                markeredgewidth=st["edge_lw"],
                color=color,
                alpha=st["alpha"],
                elinewidth=st["ebar_lw"],
                capsize=0)


for i, row in eff.iterrows():
    tier_a = sig_tier(row["p_Anxiety"], row["p_Anxiety_FDR"])
    tier_d = sig_tier(row["p_Depression"], row["p_Depression_FDR"])

    draw_point(ax, row["Beta_Anxiety"], y[i] + offset, row["CI_Anx"],
               RED, "o", tier_a, 10)
    lbl = p_label(row["p_Anxiety"], row["p_Anxiety_FDR"], tier_a)
    if lbl:
        ax.text(row["Beta_Anxiety"] + row["CI_Anx"] + 0.015, y[i] + offset,
                lbl, ha="left", va="center", color=RED,
                fontsize=12, fontweight="bold")

    draw_point(ax, row["Beta_Depression"], y[i] - offset, row["CI_Dep"],
               BLUE, "D", tier_d, 9)
    lbl = p_label(row["p_Depression"], row["p_Depression_FDR"], tier_d)
    if lbl:
        ax.text(row["Beta_Depression"] + row["CI_Dep"] + 0.015, y[i] - offset,
                lbl, ha="left", va="center", color=BLUE,
                fontsize=12, fontweight="bold")

# Reference line at 0
ax.axvline(0, color="black", linewidth=1.1, alpha=0.5)

# Shaded background bands to separate outcomes
for i in range(len(eff)):
    if i % 2 == 0:
        ax.axhspan(i - 0.45, i + 0.45, color="#f7f7f7", zorder=-1)

ax.set_yticks(y)
ax.set_yticklabels([labels_display[a] for a in eff["AI_Attitude"]], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel(r"Standardized Beta Coefficient ($\beta$) $\pm$ 95% CI", fontsize=11)
ax.set_xlim(-0.42, 0.65)
ax.set_ylim(len(eff) - 0.5, -0.5)
n = pd.read_csv(REG_DIR / "00_merged_analysis_data.csv").shape[0]
ax.set_title(f"How PHQ-4 Mental Health Predicts ChatGPT Experiences (N={n})",
             fontsize=13, fontweight="bold", pad=12)

# Custom legend: predictor colors + significance tiers
from matplotlib.lines import Line2D
pred_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=10,
           markeredgecolor=RED, markeredgewidth=0, label="PHQ-4 Anxiety"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=BLUE, markersize=9,
           markeredgecolor=BLUE, markeredgewidth=0, label="PHQ-4 Depression"),
]
sig_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", markersize=10,
           label="FDR-significant (*)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#555", markeredgewidth=1.8, markersize=10,
           label="Uncorrected p<.05 (†)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", alpha=0.35,
           markersize=10, label="Not significant"),
]
leg1 = ax.legend(handles=pred_handles, loc="lower right", frameon=True,
                 title="Predictor", fontsize=9, title_fontsize=9,
                 bbox_to_anchor=(0.995, 0.02))
ax.add_artist(leg1)
ax.legend(handles=sig_handles, loc="lower right", frameon=True,
          title="Significance", fontsize=9, title_fontsize=9,
          bbox_to_anchor=(0.78, 0.02))

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.grid(axis="x", alpha=0.25, linestyle="-", linewidth=0.5)

fig.text(0.5, -0.01,
         "* p_FDR<.05, ** p_FDR<.01, *** p_FDR<.001   "
         "† p<.05 uncorrected, †† p<.01 uncorrected",
         ha="center", fontsize=9, style="italic")

plt.tight_layout()
f1_png = FIG_DIR / "FIG1_phq4_regression_forest.png"
f1_pdf = FIG_DIR / "FIG1_phq4_regression_forest.pdf"
fig.savefig(f1_png, dpi=300, bbox_inches="tight")
fig.savefig(f1_pdf, bbox_inches="tight")
plt.close(fig)
print(f"Saved {f1_png.name}")

# ---------------------------------------------------------------------------
# Figure 2: LIWC scatter - top 4 depression associations
# ---------------------------------------------------------------------------
dep_biv = pd.read_csv(LIWC_DIR / "02_depression_liwc_correlations.csv")
merged = pd.read_csv(LIWC_DIR / "00_merged_mh_liwc_data.csv")

# Paper's top-4 pattern: pick top-4 by |r| from depression FDR-sig
sig = dep_biv[dep_biv["sig_fdr"] != ""].copy()
sig["abs_r"] = sig["r"].abs()
top4 = sig.nlargest(4, "abs_r")
if len(top4) < 4:
    remaining = dep_biv[~dep_biv["Variable"].isin(top4["Variable"])].copy()
    remaining["abs_r"] = remaining["r"].abs()
    top4 = pd.concat([top4, remaining.nlargest(4 - len(top4), "abs_r")])

top4 = top4.reset_index(drop=True)
panels = list("ABCD")

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
for i, (_, row) in enumerate(top4.iterrows()):
    ax = axes[i]
    x = merged["PHQ4_Depression"]
    yv = merged[row["Variable"]]
    m = x.notna() & yv.notna()
    x, yv = x[m], yv[m]
    r, p = pearsonr(x, yv)

    ax.scatter(x, yv, s=24, alpha=0.55, color="#c44e52",
               edgecolor="white", linewidth=0.5)
    # Regression line
    coef = np.polyfit(x, yv, 1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, np.polyval(coef, xl), color="#2a79c0", linewidth=2)
    ax.set_title(f"{panels[i]}. Depression × {row['Label']}",
                 fontsize=10.5, loc="left")
    ax.text(0.04, 0.93, f"r = {r:.3f}", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="top")
    ax.set_xlabel("PHQ-4 Depression Score", fontsize=10)
    ax.set_ylabel(f"Avg. % {row['Label']}", fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(alpha=0.25)

fig.suptitle("PHQ-4 Depression Is Associated with Linguistic Patterns in User Messages to ChatGPT",
             fontsize=12, fontweight="bold", y=1.05)
plt.tight_layout()
f2_png = FIG_DIR / "FIG2_phq4_liwc_scatter.png"
f2_pdf = FIG_DIR / "FIG2_phq4_liwc_scatter.pdf"
fig.savefig(f2_png, dpi=300, bbox_inches="tight")
fig.savefig(f2_pdf, bbox_inches="tight")
plt.close(fig)
print(f"Saved {f2_png.name}")

# ---------------------------------------------------------------------------
# Figure 3 (supplementary): PHQ-4 distribution panel
# ---------------------------------------------------------------------------
survey_cols = pd.read_csv(REG_DIR / "00_merged_analysis_data.csv")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

ax = axes[0]
ax.hist(survey_cols["PHQ4_Anxiety"].dropna(), bins=np.arange(-0.5, 7, 1),
        color="#c44e52", alpha=0.8, edgecolor="white")
ax.axvline(3, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Clinical cutoff (≥3)")
ax.set_xlabel("PHQ-4 Anxiety (0–6)")
ax.set_ylabel("Participants")
ax.set_title("A. PHQ-4 Anxiety", loc="left", fontweight="bold")
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(survey_cols["PHQ4_Depression"].dropna(), bins=np.arange(-0.5, 7, 1),
        color="#4c72b0", alpha=0.8, edgecolor="white")
ax.axvline(3, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Clinical cutoff (≥3)")
ax.set_xlabel("PHQ-4 Depression (0–6)")
ax.set_title("B. PHQ-4 Depression", loc="left", fontweight="bold")
ax.legend(fontsize=9)

ax = axes[2]
x = survey_cols["PHQ4_Anxiety"]
yv = survey_cols["PHQ4_Depression"]
m = x.notna() & yv.notna()
r, _ = pearsonr(x[m], yv[m])
jitter_x = x[m] + np.random.uniform(-0.15, 0.15, m.sum())
jitter_y = yv[m] + np.random.uniform(-0.15, 0.15, m.sum())
ax.scatter(jitter_x, jitter_y, s=22, alpha=0.45, color="#666")
ax.text(0.04, 0.93, f"r = {r:.3f}", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top")
ax.set_xlabel("PHQ-4 Anxiety")
ax.set_ylabel("PHQ-4 Depression")
ax.set_title("C. Anxiety × Depression", loc="left", fontweight="bold")

for a in axes:
    for spine in ("top", "right"):
        a.spines[spine].set_visible(False)
    a.grid(alpha=0.25)

plt.tight_layout()
f3_png = FIG_DIR / "FIG3_phq4_distribution.png"
fig.savefig(f3_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {f3_png.name}")
print(f"\nAll figures in {FIG_DIR}")
