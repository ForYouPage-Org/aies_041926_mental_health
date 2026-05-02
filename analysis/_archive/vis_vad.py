import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from datetime import datetime
import io

# 1. Setup Directories & Data Path
# ---------------------------------------------------------
timestamp = datetime.now().strftime('%y%m%dT%H%M')
# UPDATE THIS PATH to your project folder
BASE_DIR = '/Users/marxw/Sirius/FaccT_122925' 
OUTPUT_DIR = f'{BASE_DIR}/results/{timestamp}_VAD_analysis'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'

# Create the directory if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True) 

# 2. Load Data
# ---------------------------------------------------------

# UNCOMMENT the line below and point to your actual file
# merged_df = pd.read_csv(f"{BASE_DIR}/results/your_data_file.csv")
merged_df = pd.read_csv("/Users/marxw/Sirius/FaccT_122925/results/251229T1212_NRC_VAD_analysis/merged_ecr_vad_data.csv") # Loading sample data for demo

# 3. Configuration
# ---------------------------------------------------------
# We want two specific plots: Anxiety vs Arousal, and Avoidance vs Arousal
plot_configs = [
    {
        'x_var': 'ECR_Anxiety',
        'y_var': 'arousal_mean',
        'x_label': 'Attachment Anxiety',
        'title_label': 'Anxiety',
        'label_letter': 'A',
        'color': '#c44e52',
        'line_color': '#1f77b4'
    },
    {
        'x_var': 'ECR_Avoidance',
        'y_var': 'arousal_mean',
        'x_label': 'Attachment Avoidance',
        'title_label': 'Avoidance',
        'label_letter': 'B',
        'color': '#1f77b4',
        'line_color': '#1f77b4'

    }
]

# 4. Generate Plots
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5)) # 1 row, 2 columns
axes = axes.flatten()

for idx, config in enumerate(plot_configs):
    ax = axes[idx]
    x_var = config['x_var']
    y_var = config['y_var']
    
    # Filter valid data
    valid = merged_df[[x_var, y_var]].dropna()
    
    # Calculate Stats (Pearson r, N, p)
    r, p_val = stats.pearsonr(valid[x_var], valid[y_var])
    N = len(valid)
    
    # --- Plotting ---
    
    # Scatter plot
    ax.scatter(valid[x_var], valid[y_var], alpha=0.5, s=40, 
               color=config['color'], edgecolor='white', linewidth=0.5)
    
    # Regression line
    if len(valid) > 1: # Only plot line if enough points
        z = np.polyfit(valid[x_var], valid[y_var], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[x_var].min(), valid[x_var].max(), 100)
        ax.plot(x_line, p(x_line), color=config['line_color'], linewidth=2.5, alpha=0.9)
    
    # Stats box (Matching your example format)
    sig_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    stats_text = f"r = {r:.3f} "
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
    
    # Labels and Titles
    ax.set_xlabel(config['x_label'], fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean Arousal", fontsize=10, fontweight='bold')
    
    # Title format: "A. Anxiety x Mean Arousal"
    ax.set_title(f"{config['label_letter']}. {config['title_label']} × Mean Arousal", 
                 fontsize=11, fontweight='bold', loc='left')
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)

# Overall Title
fig.suptitle('Attachment Style × Arousal in User Messages to ChatGPT', 
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()

# 5. Save Figure
# ---------------------------------------------------------
fig2_path = f'{FIGURES_DIR}/FIG2_attachment_vad_scatterplots.png'
fig.savefig(fig2_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved: {fig2_path}")