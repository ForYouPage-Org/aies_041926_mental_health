import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime

# ---------------------------------------------------------
# 1. SETUP & DATA LOADING
# ---------------------------------------------------------
timestamp = datetime.now().strftime('%y%m%dT%H%M')
# UPDATE THIS PATH to your project folder
BASE_DIR = '/Users/marxw/Sirius/FaccT_122925' 
OUTPUT_DIR = f'{BASE_DIR}/results/{timestamp}_hr_composites'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'

# Create directory
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load the data provided
data_str = """AI_Attitude,Variable,Items,Beta_Anxiety,t_Anxiety,p_Anxiety,p_Anxiety_FDR,sig_Anxiety,Beta_Avoidance,t_Avoidance,p_Avoidance,p_Avoidance_FDR,sig_Avoidance,Delta_R2,F_change,p_change,p_change_FDR,sig_change
Emotional Engagement,Emotional_Engagement_Comp,"AI_Q4, AI_Q5, AI_Q6",0.43446390199068186,4.69719657521579,5.854253882553451e-06,2.9271269412767253e-05,***,0.004708638092065391,0.05025714747899736,0.9599834532014617,0.9599834532014617,ns,0.13509544038940874,13.327078941469722,4.646382344986044e-06,2.323191172493022e-05,***
Behavioral Change,Behavior_Change_Comp,"AI_Q10, AI_Q11, AI_Q12",0.1900221952630519,2.819523822126752,0.0054507744538736524,0.00908462408978942,**,-0.028753278129367843,-0.4211882903864665,0.674212568910576,1.0,ns,0.04607616642430731,4.288612011239965,0.015421719668341827,0.019277149585427283,*
Trust,Trust_Single,AI_Q1,0.2706049746883901,4.33533590961846,2.6368137942212044e-05,6.592034485553011e-05,***,-0.1510469395896352,-2.3890013397427947,0.018120693856321424,0.04530173464080356,*,0.10826943691541854,9.633950118593997,0.00011501082423881481,0.00028752706059703703,***
Self-Efficacy,Self_Efficacy_Comp,"AI_Q7, AI_Q8, AI_Q9",0.12875391860026664,2.3297418615680026,0.021135631899491525,0.026419539874364406,*,-0.16809600003264907,-3.0027710012842546,0.003128964027197201,0.015644820135986004,*,0.06099637378818509,5.246373041381726,0.006262249125519981,0.010437081875866635,*
Dependency Concern,Dependency_Single,AI_Q13,0.18200525427500822,1.7272047146021388,0.08616226317554498,0.08616226317554498,ns,-0.020976901458423482,-0.19652541103575097,0.844461461406331,1.0,ns,0.019923841368481188,1.6442751121224024,0.19656925593667596,0.19656925593667596,ns"""

df = pd.read_csv(io.StringIO(data_str))

# Calculate Standard Errors and 95% Confidence Intervals
# SE = Beta / t
# 95% CI width = 1.96 * SE
df['SE_Anxiety'] = (df['Beta_Anxiety'] / df['t_Anxiety']).abs()
df['CI_Anxiety'] = 1.96 * df['SE_Anxiety']

df['SE_Avoidance'] = (df['Beta_Avoidance'] / df['t_Avoidance']).abs()
df['CI_Avoidance'] = 1.96 * df['SE_Avoidance']

# Sort by Anxiety Effect Size for visual hierarchy
df = df.sort_values('Beta_Anxiety', ascending=True)

# ---------------------------------------------------------
# 2. PLOT GENERATION
# ---------------------------------------------------------
# Figure size: Compact but readable
fig, ax = plt.subplots(figsize=(9, 5))

# Plot settings
y_pos = np.arange(len(df))
offset = 0.15  # Shift bars slightly so they don't overlap

# Clean naming for Y-axis (Remove the 'Q4-6' part if desired, or keep it)
# Let's keep it as it adds context
labels = df['AI_Attitude']

# --- Reference Line at 0 ---
ax.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.9)

# --- PLOT ANXIETY (RED) ---
# Filter for significance for opacity
anx_alphas = [1.0 if p < 0.05 else 0.4 for p in df['p_Anxiety_FDR']]

for i, (_, row) in enumerate(df.iterrows()):
    # Error Bar + Marker
    ax.errorbar(row['Beta_Anxiety'], y_pos[i] + offset, 
                xerr=row['CI_Anxiety'], 
                fmt='o', color='#c44e52', ecolor='#c44e52', 
                elinewidth=1.5, capsize=3, markersize=8, 
                alpha=anx_alphas[i], label='Anxiety' if i == 0 else "")
    
    # Sig Star
    if row['p_Anxiety_FDR'] < 0.05:
        sig = "***" if row['p_Anxiety_FDR'] < 0.001 else "**" if row['p_Anxiety_FDR'] < 0.01 else "*"
        # Place star slightly to the right of the error bar
        ax.text(row['Beta_Anxiety'] + row['CI_Anxiety'] + 0.02, y_pos[i] + offset - 0.05, 
                sig, color='#c44e52', fontsize=10, fontweight='bold', va='center')

# --- PLOT AVOIDANCE (BLUE) ---
av_alphas = [1.0 if p < 0.05 else 0.4 for p in df['p_Avoidance_FDR']]

for i, (_, row) in enumerate(df.iterrows()):
    # Error Bar + Marker (Diamond shape for distinction)
    ax.errorbar(row['Beta_Avoidance'], y_pos[i] - offset, 
                xerr=row['CI_Avoidance'], 
                fmt='D', color='#1f77b4', ecolor='#1f77b4', 
                elinewidth=1.5, capsize=3, markersize=7, 
                alpha=av_alphas[i], label='Avoidance' if i == 0 else "")
    
    # Sig Star
    if row['p_Avoidance_FDR'] < 0.05:
        sig = "***" if row['p_Avoidance_FDR'] < 0.001 else "**" if row['p_Avoidance_FDR'] < 0.01 else "*"
        # Place star slightly to the left if negative, right if positive
        # To avoid clutter, let's put it below the marker or to the side
        pos_x = row['Beta_Avoidance'] - row['CI_Avoidance'] - 0.05 if row['Beta_Avoidance'] < 0 else row['Beta_Avoidance'] + row['CI_Avoidance'] + 0.02
        ha = 'right' if row['Beta_Avoidance'] < 0 else 'left'
        ax.text(pos_x, y_pos[i] - offset - 0.02, 
                sig, color='#1f77b4', fontsize=10, fontweight='bold', va='center', ha=ha)

# ---------------------------------------------------------
# 3. STYLING
# ---------------------------------------------------------
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11, fontweight='medium')
ax.set_xlabel('Standardized Beta Coefficient (β) ± 95% CI', fontsize=10, fontweight='bold')
ax.set_title('How Attachment Styles Predict ChatGPT Experiences (N=168)', fontsize=13, fontweight='bold', pad=15)

# Add grid
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.set_xlim(-0.4, 0.6) # Adjust to fit data

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Legend
legend = ax.legend(loc='lower right', frameon=True)
legend.get_frame().set_alpha(0.9)
legend.get_frame().set_edgecolor('#cccccc')

plt.tight_layout()

# Save
fig_path = f'{FIGURES_DIR}/FIG4_composites_forest_plot.png'
plt.savefig(fig_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved: {fig_path}")