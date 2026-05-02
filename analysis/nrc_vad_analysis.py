#!/usr/bin/env python3
"""
ECR Attachment × Chat Behavior Analysis using NRC VAD Lexicon v2.1 (2025)
=========================================================================

Purpose:
- Apply validated NRC VAD Lexicon (Mohammad, 2025) to ChatGPT user prompts
- Correlate Valence, Arousal, Dominance with attachment styles (ECR-SF)
- Filter out copy-paste messages (word count >= 20)

NRC VAD Lexicon Citation:
    Mohammad, S. M. (2025). NRC VAD Lexicon v2: Norms for Valence, Arousal, 
    and Dominance for over 55k English Terms. arXiv preprint arXiv:2503.23547.

Date: December 29, 2025
Institution: University of Washington
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')
import os
import re

# ==============================================================================
# Configuration
# ==============================================================================

# Paths
NRC_VAD_PATH = '/Users/marxw/Sirius/FaccT_122925/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt'
CHAT_DATA_PATH = '/Users/marxw/Sirius/data/conversations_full_20251007.csv'
ECR_DATA_PATH = '/Users/marxw/Sirius/analysis_102025/ecr_scores_validated_clean.csv'
CHAT_IDS_PATH = '/Users/marxw/Sirius/analysis_102025/chat_participants_list.csv'

# Output directory
timestamp = datetime.now().strftime('%y%m%dT%H%M')
OUTPUT_DIR = f'/Users/marxw/Sirius/FaccT_122925/results/{timestamp}_NRC_VAD_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)

# Filter threshold: exclude messages with 20+ words (likely copy-paste)
WORD_COUNT_THRESHOLD = 20

# Visualization settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

print("=" * 80)
print("NRC VAD LEXICON v2.1 × ATTACHMENT STYLE ANALYSIS")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Word count filter: < {WORD_COUNT_THRESHOLD} words (excluding copy-paste)")
print()

# ==============================================================================
# STEP 1: Load NRC VAD Lexicon
# ==============================================================================

print("STEP 1: Loading NRC VAD Lexicon v2.1...")

vad_df = pd.read_csv(NRC_VAD_PATH, sep='\t')
print(f"  Loaded {len(vad_df):,} terms from NRC VAD Lexicon")
print(f"  Dimensions: Valence, Arousal, Dominance (scores: -1 to +1)")

# Create dictionary for fast lookup
vad_dict = {}
for _, row in vad_df.iterrows():
    term = str(row['term']).lower().strip()
    vad_dict[term] = {
        'valence': row['valence'],
        'arousal': row['arousal'],
        'dominance': row['dominance']
    }

print(f"  VAD dictionary created with {len(vad_dict):,} entries")

# ==============================================================================
# STEP 2: Load and match participants (ECR + Chat)
# ==============================================================================

print("\nSTEP 2: Loading and matching participants...")

ecr_df = pd.read_csv(ECR_DATA_PATH)
print(f"  ECR data: {len(ecr_df)} participants")

chat_ids = pd.read_csv(CHAT_IDS_PATH)
ecr_set = set(ecr_df['Prolific_ID'].astype(str))
chat_set = set(chat_ids['chat_participant_id'].astype(str))
matched_ids = list(ecr_set.intersection(chat_set))

print(f"  Matched participants (ECR + Chat): {len(matched_ids)}")

ecr_matched = ecr_df[ecr_df['Prolific_ID'].isin(matched_ids)].copy()

# ==============================================================================
# STEP 3: Extract and filter user prompts
# ==============================================================================

print(f"\nSTEP 3: Extracting user prompts (word_count < {WORD_COUNT_THRESHOLD})...")

chunk_size = 500000
all_user_messages = []

print("  Reading chat data in chunks...")
for chunk_num, chunk in enumerate(pd.read_csv(CHAT_DATA_PATH, chunksize=chunk_size), 1):
    # Filter to matched participants, user messages only
    chunk_filtered = chunk[
        (chunk['participant_id'].isin(matched_ids)) & 
        (chunk['user_or_ai'] == 'user')
    ].copy()
    
    if len(chunk_filtered) == 0:
        continue
    
    # Filter by word count (exclude copy-paste)
    chunk_filtered = chunk_filtered[chunk_filtered['word_count'] < WORD_COUNT_THRESHOLD]
    
    if len(chunk_filtered) > 0:
        all_user_messages.append(chunk_filtered)
        print(f"    Chunk {chunk_num}: {len(chunk_filtered):,} messages after filtering")

# Combine all chunks
user_messages_df = pd.concat(all_user_messages, ignore_index=True)
print(f"\n  Total filtered user messages: {len(user_messages_df):,}")
print(f"  Unique participants: {user_messages_df['participant_id'].nunique()}")

# Save filtered user prompts
filtered_csv_path = f'{OUTPUT_DIR}/filtered_user_prompts_lt{WORD_COUNT_THRESHOLD}words.csv'
user_messages_df.to_csv(filtered_csv_path, index=False)
print(f"  ✓ Saved: {filtered_csv_path}")

# ==============================================================================
# STEP 4: Apply NRC VAD Lexicon to each message
# ==============================================================================

print("\nSTEP 4: Applying NRC VAD Lexicon to user messages...")

def calculate_vad_scores(text):
    """
    Calculate average VAD scores for a text using NRC VAD Lexicon.
    Returns mean valence, arousal, dominance for matched words.
    """
    if pd.isna(text) or not isinstance(text, str):
        return np.nan, np.nan, np.nan, 0
    
    # Tokenize: lowercase, remove punctuation, split on whitespace
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    valence_scores = []
    arousal_scores = []
    dominance_scores = []
    
    for word in words:
        if word in vad_dict:
            valence_scores.append(vad_dict[word]['valence'])
            arousal_scores.append(vad_dict[word]['arousal'])
            dominance_scores.append(vad_dict[word]['dominance'])
    
    n_matched = len(valence_scores)
    
    if n_matched == 0:
        return np.nan, np.nan, np.nan, 0
    
    return (
        np.mean(valence_scores),
        np.mean(arousal_scores),
        np.mean(dominance_scores),
        n_matched
    )

# Apply VAD scoring to each message
print("  Calculating VAD scores for each message...")
vad_results = user_messages_df['message'].apply(calculate_vad_scores)

user_messages_df['vad_valence'] = vad_results.apply(lambda x: x[0])
user_messages_df['vad_arousal'] = vad_results.apply(lambda x: x[1])
user_messages_df['vad_dominance'] = vad_results.apply(lambda x: x[2])
user_messages_df['vad_matched_words'] = vad_results.apply(lambda x: x[3])

# Calculate coverage statistics
total_words = user_messages_df['word_count'].sum()
matched_words = user_messages_df['vad_matched_words'].sum()
coverage = (matched_words / total_words) * 100

print(f"  VAD lexicon coverage: {coverage:.1f}% of words matched")
print(f"  Messages with VAD scores: {user_messages_df['vad_valence'].notna().sum():,}")

# ==============================================================================
# STEP 5: Aggregate VAD scores at participant level
# ==============================================================================

print("\nSTEP 5: Aggregating VAD scores by participant...")

participant_vad = user_messages_df.groupby('participant_id').agg({
    'vad_valence': ['mean', 'std'],
    'vad_arousal': ['mean', 'std'],
    'vad_dominance': ['mean', 'std'],
    'vad_matched_words': 'sum',
    'word_count': 'sum',
    'message': 'count'
}).reset_index()

# Flatten column names
participant_vad.columns = [
    'participant_id',
    'valence_mean', 'valence_std',
    'arousal_mean', 'arousal_std',
    'dominance_mean', 'dominance_std',
    'total_vad_words', 'total_words', 'total_messages'
]

# Calculate VAD coverage per participant
participant_vad['vad_coverage'] = (participant_vad['total_vad_words'] / participant_vad['total_words']) * 100

print(f"  Participants with VAD data: {len(participant_vad)}")

# ==============================================================================
# STEP 6: Merge with ECR attachment scores
# ==============================================================================

print("\nSTEP 6: Merging VAD data with ECR attachment scores...")

merged_df = ecr_matched.merge(
    participant_vad,
    left_on='Prolific_ID',
    right_on='participant_id',
    how='inner'
)

print(f"  Final sample: N = {len(merged_df)} participants")

# ==============================================================================
# STEP 7: Correlational analysis with FDR correction
# ==============================================================================

print("\nSTEP 7: Correlational analysis (Attachment × VAD)...")

# Define VAD variables
vad_vars = [
    ('valence_mean', 'Mean Valence', 'Positive-Negative'),
    ('arousal_mean', 'Mean Arousal', 'Excited-Calm'),
    ('dominance_mean', 'Mean Dominance', 'Powerful-Weak'),
    ('valence_std', 'Valence Variability', 'Emotional Range'),
    ('arousal_std', 'Arousal Variability', 'Activation Range'),
    ('dominance_std', 'Dominance Variability', 'Control Range'),
]

# Correlations with Attachment Anxiety
anxiety_results = []
for var, label, desc in vad_vars:
    valid = merged_df[['ECR_Anxiety', var]].dropna()
    if len(valid) > 10:
        r, p = pearsonr(valid['ECR_Anxiety'], valid[var])
        anxiety_results.append({
            'VAD_Variable': label,
            'Description': desc,
            'Variable': var,
            'N': len(valid),
            'r': r,
            'p_uncorrected': p,
            'r_squared': r**2
        })

# Correlations with Attachment Avoidance
avoidance_results = []
for var, label, desc in vad_vars:
    valid = merged_df[['ECR_Avoidance', var]].dropna()
    if len(valid) > 10:
        r, p = pearsonr(valid['ECR_Avoidance'], valid[var])
        avoidance_results.append({
            'VAD_Variable': label,
            'Description': desc,
            'Variable': var,
            'N': len(valid),
            'r': r,
            'p_uncorrected': p,
            'r_squared': r**2
        })

# Convert to DataFrames
anxiety_df = pd.DataFrame(anxiety_results)
avoidance_df = pd.DataFrame(avoidance_results)

# Apply FDR correction (Benjamini-Hochberg)
if len(anxiety_df) > 0:
    _, anxiety_df['p_fdr'], _, _ = multipletests(anxiety_df['p_uncorrected'], method='fdr_bh')
    anxiety_df['sig_uncorrected'] = anxiety_df['p_uncorrected'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )
    anxiety_df['sig_fdr'] = anxiety_df['p_fdr'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )

if len(avoidance_df) > 0:
    _, avoidance_df['p_fdr'], _, _ = multipletests(avoidance_df['p_uncorrected'], method='fdr_bh')
    avoidance_df['sig_uncorrected'] = avoidance_df['p_uncorrected'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )
    avoidance_df['sig_fdr'] = avoidance_df['p_fdr'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )

# Print results
print("\n  === ATTACHMENT ANXIETY × VAD ===")
for _, row in anxiety_df.iterrows():
    print(f"    {row['VAD_Variable']}: r = {row['r']:.3f}, p = {row['p_uncorrected']:.4f} {row['sig_uncorrected']} (FDR: {row['sig_fdr']})")

print("\n  === ATTACHMENT AVOIDANCE × VAD ===")
for _, row in avoidance_df.iterrows():
    print(f"    {row['VAD_Variable']}: r = {row['r']:.3f}, p = {row['p_uncorrected']:.4f} {row['sig_uncorrected']} (FDR: {row['sig_fdr']})")

# ==============================================================================
# STEP 8: Save key data tables as TXT
# ==============================================================================

print("\nSTEP 8: Saving key data tables...")

# Table 1: Anxiety correlations
table1_path = f'{OUTPUT_DIR}/TABLE1_anxiety_vad_correlations.txt'
with open(table1_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("TABLE 1: Attachment Anxiety × NRC VAD Dimensions\n")
    f.write("=" * 80 + "\n")
    f.write(f"N = {anxiety_df['N'].iloc[0]} participants\n")
    f.write(f"NRC VAD Lexicon v2.1 (Mohammad, 2025)\n")
    f.write(f"FDR correction applied (Benjamini-Hochberg)\n")
    f.write("-" * 80 + "\n\n")
    f.write(f"{'VAD Dimension':<25} {'r':>8} {'p':>10} {'Sig':>6} {'p_FDR':>10} {'FDR':>6} {'R²':>8}\n")
    f.write("-" * 80 + "\n")
    for _, row in anxiety_df.iterrows():
        f.write(f"{row['VAD_Variable']:<25} {row['r']:>8.3f} {row['p_uncorrected']:>10.4f} {row['sig_uncorrected']:>6} {row['p_fdr']:>10.4f} {row['sig_fdr']:>6} {row['r_squared']:>8.3f}\n")
    f.write("-" * 80 + "\n")
    f.write("\nNote: Valence (-1=negative, +1=positive), Arousal (-1=calm, +1=excited),\n")
    f.write("      Dominance (-1=weak/submissive, +1=powerful/in-control)\n")
    f.write("      *** p<.001, ** p<.01, * p<.05\n")

print(f"  ✓ Saved: {table1_path}")

# Table 2: Avoidance correlations
table2_path = f'{OUTPUT_DIR}/TABLE2_avoidance_vad_correlations.txt'
with open(table2_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("TABLE 2: Attachment Avoidance × NRC VAD Dimensions\n")
    f.write("=" * 80 + "\n")
    f.write(f"N = {avoidance_df['N'].iloc[0]} participants\n")
    f.write(f"NRC VAD Lexicon v2.1 (Mohammad, 2025)\n")
    f.write(f"FDR correction applied (Benjamini-Hochberg)\n")
    f.write("-" * 80 + "\n\n")
    f.write(f"{'VAD Dimension':<25} {'r':>8} {'p':>10} {'Sig':>6} {'p_FDR':>10} {'FDR':>6} {'R²':>8}\n")
    f.write("-" * 80 + "\n")
    for _, row in avoidance_df.iterrows():
        f.write(f"{row['VAD_Variable']:<25} {row['r']:>8.3f} {row['p_uncorrected']:>10.4f} {row['sig_uncorrected']:>6} {row['p_fdr']:>10.4f} {row['sig_fdr']:>6} {row['r_squared']:>8.3f}\n")
    f.write("-" * 80 + "\n")
    f.write("\nNote: Valence (-1=negative, +1=positive), Arousal (-1=calm, +1=excited),\n")
    f.write("      Dominance (-1=weak/submissive, +1=powerful/in-control)\n")
    f.write("      *** p<.001, ** p<.01, * p<.05\n")

print(f"  ✓ Saved: {table2_path}")

# ==============================================================================
# STEP 9: Create key visualizations
# ==============================================================================

print("\nSTEP 9: Creating visualizations...")

# Color scheme matching the original analysis
COLORS = {
    'sig_001': '#c44e52',    # dark red for p < 0.001
    'sig_01': '#dd8452',     # orange for p < 0.01
    'sig_05': '#f0c566',     # yellow for p < 0.05
    'ns': '#b0b0b0',         # gray for non-significant
}

def get_color(p):
    if p < 0.001:
        return COLORS['sig_001']
    elif p < 0.01:
        return COLORS['sig_01']
    elif p < 0.05:
        return COLORS['sig_05']
    else:
        return COLORS['ns']

# === FIGURE 1: Two-panel forest plot ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Anxiety
y_pos = np.arange(len(anxiety_df))
colors_anx = [get_color(p) for p in anxiety_df['p_uncorrected']]

ax1.barh(y_pos, anxiety_df['r'], color=colors_anx, edgecolor='white', linewidth=1, height=0.6)
ax1.axvline(0, color='#333333', linewidth=1, linestyle='-')
ax1.axvline(0.3, color='#cccccc', linewidth=0.8, linestyle='--', alpha=0.6)
ax1.axvline(-0.3, color='#cccccc', linewidth=0.8, linestyle='--', alpha=0.6)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(anxiety_df['VAD_Variable'], fontsize=10)
ax1.set_xlabel('Correlation coefficient (r)', fontsize=11, fontweight='bold')
ax1.set_title('A. Attachment Anxiety × VAD', fontsize=12, fontweight='bold', loc='left')
ax1.set_xlim(-0.4, 0.4)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

# Add correlation values
for i, (_, row) in enumerate(anxiety_df.iterrows()):
    x_pos = row['r'] + 0.02 if row['r'] >= 0 else row['r'] - 0.02
    ha = 'left' if row['r'] >= 0 else 'right'
    ax1.text(x_pos, i, f"r={row['r']:.2f}{row['sig_uncorrected']}", 
             ha=ha, va='center', fontsize=9, fontweight='medium')

# Panel B: Avoidance
y_pos = np.arange(len(avoidance_df))
colors_avo = [get_color(p) for p in avoidance_df['p_uncorrected']]

ax2.barh(y_pos, avoidance_df['r'], color=colors_avo, edgecolor='white', linewidth=1, height=0.6)
ax2.axvline(0, color='#333333', linewidth=1, linestyle='-')
ax2.axvline(0.3, color='#cccccc', linewidth=0.8, linestyle='--', alpha=0.6)
ax2.axvline(-0.3, color='#cccccc', linewidth=0.8, linestyle='--', alpha=0.6)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(avoidance_df['VAD_Variable'], fontsize=10)
ax2.set_xlabel('Correlation coefficient (r)', fontsize=11, fontweight='bold')
ax2.set_title('B. Attachment Avoidance × VAD', fontsize=12, fontweight='bold', loc='left')
ax2.set_xlim(-0.4, 0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

# Add correlation values
for i, (_, row) in enumerate(avoidance_df.iterrows()):
    x_pos = row['r'] + 0.02 if row['r'] >= 0 else row['r'] - 0.02
    ha = 'left' if row['r'] >= 0 else 'right'
    ax2.text(x_pos, i, f"r={row['r']:.2f}{row['sig_uncorrected']}", 
             ha=ha, va='center', fontsize=9, fontweight='medium')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS['sig_001'], label='p < .001'),
    Patch(facecolor=COLORS['sig_01'], label='p < .01'),
    Patch(facecolor=COLORS['sig_05'], label='p < .05'),
    Patch(facecolor=COLORS['ns'], label='n.s.'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Attachment Dimensions × Linguistic Affect (NRC VAD v2.1)', 
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
fig1_path = f'{OUTPUT_DIR}/figures/FIG1_attachment_vad_correlations.png'
fig.savefig(fig1_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✓ Saved: {fig1_path}")

# === FIGURE 2: Scatterplot matrix for significant findings ===
# Find the strongest correlations
all_results = []
for _, row in anxiety_df.iterrows():
    all_results.append({**row.to_dict(), 'attachment': 'Anxiety', 'ecr_var': 'ECR_Anxiety'})
for _, row in avoidance_df.iterrows():
    all_results.append({**row.to_dict(), 'attachment': 'Avoidance', 'ecr_var': 'ECR_Avoidance'})

all_results_df = pd.DataFrame(all_results)
top_results = all_results_df.nsmallest(4, 'p_uncorrected')  # Top 4 by significance

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()

for idx, (_, result) in enumerate(top_results.iterrows()):
    ax = axes[idx]
    ecr_var = result['ecr_var']
    vad_var = result['Variable']
    
    valid = merged_df[[ecr_var, vad_var]].dropna()
    
    # Scatter plot
    ax.scatter(valid[ecr_var], valid[vad_var], alpha=0.5, s=40, 
               color='#1f77b4', edgecolor='white', linewidth=0.5)
    
    # Regression line
    z = np.polyfit(valid[ecr_var], valid[vad_var], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid[ecr_var].min(), valid[ecr_var].max(), 100)
    ax.plot(x_line, p(x_line), color='#c44e52', linewidth=2.5, alpha=0.9)
    
    # Stats box
    stats_text = f"r = {result['r']:.3f} {result['sig_uncorrected']}\nN = {result['N']}\np = {result['p_uncorrected']:.4f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
    
    ax.set_xlabel(f"Attachment {result['attachment']}", fontsize=10, fontweight='bold')
    ax.set_ylabel(result['VAD_Variable'], fontsize=10, fontweight='bold')
    ax.set_title(f"{'ABCD'[idx]}. {result['attachment']} × {result['VAD_Variable']}", 
                 fontsize=11, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)

fig.suptitle('Top Attachment × VAD Associations (Scatterplots)', 
             fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout()
fig2_path = f'{OUTPUT_DIR}/figures/FIG2_attachment_vad_scatterplots.png'
fig.savefig(fig2_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✓ Saved: {fig2_path}")

# ==============================================================================
# STEP 10: Save merged data for further analysis
# ==============================================================================

print("\nSTEP 10: Saving merged dataset...")

merged_df.to_csv(f'{OUTPUT_DIR}/merged_ecr_vad_data.csv', index=False)
participant_vad.to_csv(f'{OUTPUT_DIR}/participant_vad_scores.csv', index=False)

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "=" * 80)
print("NRC VAD ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nSample:")
print(f"  • N = {len(merged_df)} participants")
print(f"  • {len(user_messages_df):,} user messages (< {WORD_COUNT_THRESHOLD} words)")
print(f"  • VAD lexicon coverage: {coverage:.1f}%")

print(f"\nKey Findings (Attachment Anxiety):")
sig_anx = anxiety_df[anxiety_df['sig_uncorrected'] != 'ns']
if len(sig_anx) > 0:
    for _, row in sig_anx.iterrows():
        print(f"  • {row['VAD_Variable']}: r = {row['r']:.3f} {row['sig_uncorrected']}")
else:
    print("  • No significant associations")

print(f"\nKey Findings (Attachment Avoidance):")
sig_avo = avoidance_df[avoidance_df['sig_uncorrected'] != 'ns']
if len(sig_avo) > 0:
    for _, row in sig_avo.iterrows():
        print(f"  • {row['VAD_Variable']}: r = {row['r']:.3f} {row['sig_uncorrected']}")
else:
    print("  • No significant associations")

print(f"\nOutput files:")
print(f"  • {OUTPUT_DIR}/")
print(f"    ├── TABLE1_anxiety_vad_correlations.txt")
print(f"    ├── TABLE2_avoidance_vad_correlations.txt")
print(f"    ├── figures/FIG1_attachment_vad_correlations.png")
print(f"    ├── figures/FIG2_attachment_vad_scatterplots.png")
print(f"    ├── filtered_user_prompts_lt{WORD_COUNT_THRESHOLD}words.csv")
print(f"    ├── participant_vad_scores.csv")
print(f"    └── merged_ecr_vad_data.csv")

print("\n" + "=" * 80)
print("Citation: Mohammad, S.M. (2025). NRC VAD Lexicon v2. arXiv:2503.23547")
print("=" * 80)

