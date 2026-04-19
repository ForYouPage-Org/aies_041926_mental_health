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
BASE_DIR = '/Users/marxw/Sirius/FaccT_122925' 
OUTPUT_DIR = f'{BASE_DIR}/results/{timestamp}_hr'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'

# Create the directory explicitly
os.makedirs(FIGURES_DIR, exist_ok=True) 

# ---------------------------------------------------------
# 1. LOAD AND PREP DATA
# ---------------------------------------------------------
df = pd.read_csv('/Users/marxw/Sirius/FaccT_122925/results/260110T1635_ecr_ai_controlled_regression/05_ecr_effects_controlled.csv')

# Sort by the magnitude of Anxiety Beta
df = df.sort_values('Beta_Anxiety', ascending=True)

# ---------------------------------------------------------
# 2. PLOT CONFIGURATION (REFINED SIZE)
# ---------------------------------------------------------
# CHANGED: Reduced height from 12 to 5 (approx 60% reduction)
fig, ax = plt.subplots(figsize=(10, 5.5))

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#333333'

y_pos = np.arange(len(df))
# CHANGED: Reduced offset slightly (0.3 -> 0.25) so dots stay tighter to the line
height = 0.25  

# ---------------------------------------------------------
# 3. DRAW THE ELEMENTS
# ---------------------------------------------------------

# Vertical line at 0
ax.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.3)

for i, (_, row) in enumerate(df.iterrows()):
    y = y_pos[i]
    
    # --- ANXIETY (Red #c44e52) ---
    anx_alpha = 1.0 if row['p_Anxiety'] < 0.05 else 0.3
    
    # CHANGED: Reduced s (size) to 80
    ax.scatter(row['Beta_Anxiety'], y + height/2, 
               s=80, color='#c44e52', alpha=anx_alpha, 
               edgecolor='none', label='Anxiety' if i == 0 else "")
    
    if row['p_Anxiety'] < 0.05:
        sig_symbol = "***" if row['p_Anxiety'] < 0.001 else "**" if row['p_Anxiety'] < 0.01 else "*"
        # CHANGED: Adjusted vertical text offset and font size
        ax.text(row['Beta_Anxiety'], y + height/2 + 0.12, sig_symbol, 
                ha='center', va='bottom', fontsize=8, color='#c44e52', fontweight='bold')

    # --- AVOIDANCE (Blue #1f77b4) ---
    av_alpha = 1.0 if row['p_Avoidance'] < 0.05 else 0.3
    
    # CHANGED: Reduced s (size) to 70 (diamonds look bigger visually, so slightly smaller)
    ax.scatter(row['Beta_Avoidance'], y - height/2, 
               s=70, color='#1f77b4', marker='D', alpha=av_alpha, 
               edgecolor='none', label='Avoidance' if i == 0 else "")

    if row['p_Avoidance'] < 0.05:
        sig_symbol = "***" if row['p_Avoidance'] < 0.001 else "**" if row['p_Avoidance'] < 0.01 else "*"
        # CHANGED: Adjusted vertical text offset and font size
        ax.text(row['Beta_Avoidance'], y - height/2 - 0.2, sig_symbol, 
                ha='center', va='top', fontsize=8, color='#1f77b4', fontweight='bold')

# ---------------------------------------------------------
# 4. FORMATTING
# ---------------------------------------------------------

ax.set_yticks(y_pos)
ax.set_yticklabels(df['AI_Attitude'], fontsize=10, fontweight='medium')

ax.set_xlabel('Standardized Beta Coefficient (β)', fontsize=10, fontweight='bold', labelpad=8)
ax.set_xlim(-0.6, 0.6)
ax.grid(axis='x', linestyle='--', alpha=0.3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.5)

ax.set_title('How Attachment Styles Predict ChatGPT Experiences (N=168)', 
             fontsize=12, fontweight='bold', loc='left', pad=15)

# Compact Legend
legend = ax.legend(loc='lower right', frameon=True, fontsize=9, ncol=1)
legend.get_frame().set_edgecolor('#cccccc')
legend.get_frame().set_alpha(0.9)

ax.text(0.98, 0.02, 'Faded points = non-significant (p > .05)', 
        transform=ax.transAxes, ha='right', fontsize=8, color='#666666', style='italic')

plt.tight_layout()

# ---------------------------------------------------------
# SAVE FIGURE
fig2_path = f'{FIGURES_DIR}/FIG3_coefficients_plot.png'
fig.savefig(fig2_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved: {fig2_path}")