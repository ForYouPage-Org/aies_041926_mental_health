#!/usr/bin/env python3
"""
FAccT Publication Forest Plot: Regression Coefficients (β)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "260107T1230_ecr_ai_rigorous_analysis"
VIS_DIR = BASE_DIR / "vis"

reg_df = pd.read_csv(RESULTS_DIR / "04_multiple_regression.csv")

# Define categories with spacing for headers
DISPLAY_ORDER = [
    ('EMOTIONAL SUPPORT', [
        ('AI_Q6', 'Easier to share struggles with ChatGPT than with people'),
        ('AI_Q5', 'Experience emotional relief'),
        ('AI_Q4', 'Feel emotionally understood'),
    ]),
    ('TRUST & VERIFICATION', [
        ('AI_Q1', 'Trust ChatGPT for accurate information'),
        ('AI_Q2', 'Regularly fact-check ChatGPT responses'),
        ('AI_Q3', 'Confident implementing suggestions'),
    ]),
    ('DEPENDENCY', [
        ('AI_Q14', 'Less confident without ChatGPT'),
        ('AI_Q13', 'Worry about over-reliance'),
        ('AI_Q15', 'Prefer attempting tasks alone'),
    ]),
    ('SELF-EFFICACY', [
        ('AI_Q7', 'Improved problem-solving'),
        ('AI_Q8', 'Feel more capable'),
        ('AI_Q9', 'More efficient at tasks'),
    ]),
    ('BEHAVIORAL CHANGE', [
        ('AI_Q12', 'Changed professional communication'),
        ('AI_Q10', 'Modify writing style'),
        ('AI_Q11', 'Approach learning differently'),
    ]),
]

COLORS = {
    'p<0.001': '#C41E3A',
    'p<0.01': '#E07020',
    'p<0.05': '#D4A017',
    'ns': '#9E9E9E'
}

BG_COLORS = ['#FAFAFA', '#F2F2F2']

def get_sig_color(p):
    if p < 0.001: return COLORS['p<0.001']
    elif p < 0.01: return COLORS['p<0.01']
    elif p < 0.05: return COLORS['p<0.05']
    else: return COLORS['ns']

def get_sig_label(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''

# Build positions with category headers
y_positions = {}
y_labels = {}
category_header_y = {}
current_y = 0

for cat_idx, (cat, items) in enumerate(reversed(DISPLAY_ORDER)):
    for var, label in reversed(items):
        y_positions[var] = current_y
        y_labels[var] = label
        current_y += 1
    category_header_y[cat] = current_y + 0.3
    current_y += 1.2  # Gap for category header

total_height = current_y

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')

def plot_panel(ax, beta_col, p_col, title, show_labels=True):
    # Background bands
    y_start = -0.5
    for cat_idx, (cat, items) in enumerate(reversed(DISPLAY_ORDER)):
        n = len(items)
        rect = Rectangle((-0.55, y_start), 1.25, n,
                        facecolor=BG_COLORS[cat_idx % 2], edgecolor='none', zorder=0)
        ax.add_patch(rect)
        y_start += n + 1.2
    
    # Plot items
    for var, y in y_positions.items():
        row = reg_df[reg_df['Variable'] == var].iloc[0]
        beta = row[beta_col]
        p = row[p_col]
        label = y_labels[var]
        
        color = get_sig_color(p)
        sig = get_sig_label(p)
        
        # Line
        ax.hlines(y, 0, beta, colors=color, linewidth=2.5, zorder=2)
        
        # Dot
        ax.scatter(beta, y, color=color, s=80, zorder=3, edgecolors='white', linewidth=0.8)
        
        # Beta label
        offset = 0.03 if beta >= 0 else -0.03
        ha = 'left' if beta >= 0 else 'right'
        ax.text(beta + offset, y, f'β={beta:.2f}{sig}', va='center', ha=ha,
                fontsize=9, color=color, fontweight='medium')
        
        # Item label
        if show_labels:
            ax.text(-0.53, y, label, va='center', ha='right', fontsize=9.5, color='#333333')
    
    # Category headers
    for cat, y in category_header_y.items():
        if show_labels:
            ax.text(-0.53, y - 0.5, cat, va='center', ha='right', fontsize=9,
                    fontweight='bold', color='#555555', style='italic')
    
    # Zero line
    ax.axvline(0, color='#555555', linewidth=1, linestyle='-', zorder=1, alpha=0.6)
    
    # Styling
    ax.set_xlim(-0.55, 0.70)
    ax.set_ylim(-0.7, total_height - 0.5)
    ax.set_xlabel('Standardized coefficient (β)', fontsize=10.5, fontweight='medium', labelpad=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=9.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#AAAAAA')
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
    ax.grid(axis='x', linestyle=':', alpha=0.4, color='#BBBBBB', zorder=0)

# Plot panels
plot_panel(ax1, 'Beta_Anxiety', 'p_Anxiety', 'A. Attachment Anxiety', show_labels=True)
plot_panel(ax2, 'Beta_Avoidance', 'p_Avoidance', 'B. Attachment Avoidance', show_labels=True)

# Legend
legend_elements = [
    Patch(facecolor=COLORS['p<0.001'], edgecolor='none', label='p < 0.001'),
    Patch(facecolor=COLORS['p<0.01'], edgecolor='none', label='p < 0.01'),
    Patch(facecolor=COLORS['p<0.05'], edgecolor='none', label='p < 0.05'),
    Patch(facecolor=COLORS['ns'], edgecolor='none', label='n.s.')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.01),
           handlelength=1.8, handleheight=1.2)

# Title
fig.suptitle('Attachment Dimensions × ChatGPT Experiences: Multiple Regression',
             fontsize=15, fontweight='bold', y=0.97)
fig.text(0.5, 0.93, 'Standardized coefficients (β) from simultaneous regression, N = 168',
         ha='center', fontsize=10, color='#555555', style='italic')

plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.90])
plt.subplots_adjust(wspace=0.55)

# Save
fig.savefig(VIS_DIR / 'FIG_regression_forest_beta.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig(VIS_DIR / 'FIG_regression_forest_beta.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"✓ Saved: {VIS_DIR / 'FIG_regression_forest_beta.png'}")
print(f"✓ Saved: {VIS_DIR / 'FIG_regression_forest_beta.pdf'}")

plt.close()
