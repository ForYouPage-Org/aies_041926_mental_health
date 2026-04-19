import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from datetime import datetime

# 1. Setup Directories
timestamp = datetime.now().strftime('%y%m%dT%H%M')
OUTPUT_DIR = f'/Users/marxw/Sirius/FaccT_122925/results/{timestamp}_LIWC22_analysis'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'

# Create the directory if it doesn't exist to prevent FileNotFoundError
os.makedirs(FIGURES_DIR, exist_ok=True) 

# Read the csv data
merged_df = pd.read_csv("/Users/marxw/Sirius/FaccT_122925/results/251229T1220_LIWC22_analysis/merged_ecr_liwc22_data.csv")

# 2. Configuration
ecr_var = 'ECR_Anxiety'
liwc_vars_to_plot = [
    {'var': 'i', 'label': 'I Words'},
    {'var': 'Affect', 'label': 'Total Affect'},
    # {'var': 'emo_neg', 'label': 'Negative Emotion'}, 
    {'var': 'focusfuture', 'label': 'Thinking About the Future'},
    {'var': 'Analytic', 'label': 'Analytical Thinking'}, 

]

# 3. Create Plot
fig, axes = plt.subplots(1, len(liwc_vars_to_plot), figsize=(15, 5))
axes = axes.flatten()

for idx, item in enumerate(liwc_vars_to_plot):
    liwc_var = item['var']
    label = item['label']
    ax = axes[idx]
    
    valid = merged_df[[ecr_var, liwc_var]].dropna()
    
    # Calculate stats (only need r now)
    r, p_val = stats.pearsonr(valid[ecr_var], valid[liwc_var])
    
    # Scatter plot
    ax.scatter(valid[ecr_var], valid[liwc_var], alpha=0.6, s=60, 
               color='#c44e52', edgecolor='white', linewidth=0.5)
    
    # Regression line
    z = np.polyfit(valid[ecr_var], valid[liwc_var], 1)
    p = np.poly1d(z)
    
    x_range_buffer = (valid[ecr_var].max() - valid[ecr_var].min()) * 0.1
    x_line = np.linspace(valid[ecr_var].min() - x_range_buffer, 
                         valid[ecr_var].max() + x_range_buffer, 100)
    
    ax.plot(x_line, p(x_line), color='#1f77b4', linewidth=3, alpha=0.8)
    
    # Stats box (Removed N and p, kept r)
    stats_text = f"r = {r:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
    
    # Refined Labels
    ax.set_xlabel("Individual Attachment Anxiety Score", fontsize=11, fontweight='bold')
    ax.set_ylabel(f"Avg. % {label} Words", fontsize=11, fontweight='bold')
    
    letter = chr(65 + idx)
    ax.set_title(f"{letter}. Anxiety × {label}", 
                 fontsize=12, fontweight='bold', loc='left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)

fig.suptitle('Association between Attachment Anxiety and LIWC Variables', 
             fontsize=14, fontweight='bold', y=1.05)

plt.tight_layout()

# Save Figure
fig2_path = f'{FIGURES_DIR}/FIG2_attachment_liwc22_scatterplots.png'
fig.savefig(fig2_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved: {fig2_path}")