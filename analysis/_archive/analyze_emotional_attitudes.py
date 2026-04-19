#!/usr/bin/env python3
"""
Emotional AI Attitudes Analysis
--------------------------------
Analyzes the three key emotional attitude survey questions:
1. "I feel emotionally understood when interacting with ChatGPT"
2. "I experience emotional relief after discussing personal matters"
3. "I find it easier to share personal struggles with ChatGPT than with people"

Outputs: Half-column publication figure (3.5" wide) for FAccT
Author: UW HCI Research Team
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Configuration ===
DATA_PATH = Path(__file__).parent.parent / "data" / "survey.csv"
OUTPUT_DIR = Path(__file__).parent
FIGURE_DPI = 600

# Survey column indices (0-indexed)
EMOTIONAL_COLS = {
    67: "Emotionally\nUnderstood",
    70: "Emotional\nRelief", 
    71: "Easier to Share\nwith AI"
}

# 5-point Likert scale mapping (exact text from survey)
LIKERT_MAP = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neither Agree nor Disagree": 3,
    "Agree": 4,
    "Strongly Agree": 5
}

LIKERT_LABELS = ["Strongly\nDisagree", "Disagree", "Neither", "Agree", "Strongly\nAgree"]

# === Load and Process Data ===
print("=" * 60)
print("EMOTIONAL AI ATTITUDES ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nTotal survey responses: {len(df)}")

# Extract and process the three emotional attitude columns
data = {}
for col_idx, short_name in EMOTIONAL_COLS.items():
    col_name = df.columns[col_idx]
    print(f"\n[{col_idx}] {short_name.replace(chr(10), ' ')}")
    print(f"    Full: {col_name[:80]}...")
    
    # Map text responses to numeric
    values = df.iloc[:, col_idx].map(LIKERT_MAP)
    valid = values.dropna()
    data[short_name] = valid
    
    print(f"    Valid N: {len(valid)}")
    print(f"    Mean: {valid.mean():.2f} (SD: {valid.std():.2f})")
    print(f"    Median: {valid.median():.1f}")

# === Calculate Response Distributions ===
print("\n" + "=" * 60)
print("RESPONSE DISTRIBUTIONS (%)")
print("=" * 60)

dist_data = {}
for name, values in data.items():
    counts = values.value_counts().sort_index()
    percentages = (counts / len(values) * 100).reindex([1, 2, 3, 4, 5], fill_value=0)
    dist_data[name] = percentages
    
    print(f"\n{name.replace(chr(10), ' ')}:")
    for score, pct in percentages.items():
        label = LIKERT_LABELS[score - 1].replace('\n', ' ')
        bar = "█" * int(pct / 2)
        print(f"  {score} ({label:18}): {pct:5.1f}% {bar}")

# Convert to DataFrame for plotting
dist_df = pd.DataFrame(dist_data)

# === Calculate Agreement Rates ===
print("\n" + "=" * 60)
print("AGREEMENT RATES (Agree + Strongly Agree)")
print("=" * 60)

for name, values in data.items():
    agree_rate = (values >= 4).sum() / len(values) * 100
    disagree_rate = (values <= 2).sum() / len(values) * 100
    neutral_rate = (values == 3).sum() / len(values) * 100
    print(f"\n{name.replace(chr(10), ' ')}:")
    print(f"  Agree/Strongly Agree: {agree_rate:.1f}%")
    print(f"  Neutral:              {neutral_rate:.1f}%")
    print(f"  Disagree/Strongly:    {disagree_rate:.1f}%")

# === Create Publication Figure ===
print("\n" + "=" * 60)
print("GENERATING FIGURE")
print("=" * 60)

# Figure setup: half-column width (3.5 inches for typical ACM format)
fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=150)

# Color palette - warm gradient for emotional constructs
colors = ['#8B0000', '#CD5C5C', '#D3D3D3', '#5B8A72', '#2E5A4C']  # Dark red → Gray → Teal

# Prepare data for stacked horizontal bar
questions = list(EMOTIONAL_COLS.values())
n_questions = len(questions)
y_pos = np.arange(n_questions)

# Create stacked horizontal bars
left = np.zeros(n_questions)
for i, (score, label) in enumerate(zip([1, 2, 3, 4, 5], LIKERT_LABELS)):
    widths = [dist_df.loc[score, q] for q in questions]
    bars = ax.barh(y_pos, widths, left=left, height=0.65, 
                   label=label.replace('\n', ' '), color=colors[i],
                   edgecolor='white', linewidth=0.5)
    
    # Add percentage labels for segments > 10%
    for j, (w, l) in enumerate(zip(widths, left)):
        if w > 12:
            ax.text(l + w/2, j, f'{w:.0f}%', 
                   ha='center', va='center', fontsize=7,
                   color='white' if i in [0, 1, 3, 4] else 'black',
                   fontweight='medium')
    left += widths

# Styling
ax.set_yticks(y_pos)
ax.set_yticklabels(questions, fontsize=8, fontweight='medium')
ax.set_xlabel('Response Distribution (%)', fontsize=9, fontweight='medium')
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.tick_params(axis='x', labelsize=8)

# Add vertical line at 50% for reference
ax.axvline(x=50, color='black', linestyle=':', linewidth=0.5, alpha=0.5)

# Legend - compact, below the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=5, fontsize=6.5, frameon=False,
          handlelength=1.2, handletextpad=0.3, columnspacing=0.8)

# Add sample size annotation
n_valid = min(len(v) for v in data.values())
ax.text(0.98, 0.98, f'N = {n_valid}', transform=ax.transAxes,
        fontsize=7, ha='right', va='top', style='italic',
        color='gray')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Title (optional - can comment out for paper)
# ax.set_title('Emotional Attitudes Toward AI', fontsize=10, fontweight='bold', pad=10)

plt.tight_layout()

# Save figure
output_path = OUTPUT_DIR / "FIG_emotional_attitudes_distribution.png"
plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"\n✅ Saved: {output_path}")

# Also save PDF for vector graphics
pdf_path = OUTPUT_DIR / "FIG_emotional_attitudes_distribution.pdf"
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✅ Saved: {pdf_path}")

plt.close()

# === Summary Statistics Table ===
print("\n" + "=" * 60)
print("SUMMARY TABLE (for paper)")
print("=" * 60)

summary_data = []
for name, values in data.items():
    agree_rate = (values >= 4).sum() / len(values) * 100
    summary_data.append({
        'Measure': name.replace('\n', ' '),
        'N': len(values),
        'M': f"{values.mean():.2f}",
        'SD': f"{values.std():.2f}",
        'Agree %': f"{agree_rate:.1f}%"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary CSV
summary_path = OUTPUT_DIR / "emotional_attitudes_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\n✅ Saved: {summary_path}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

