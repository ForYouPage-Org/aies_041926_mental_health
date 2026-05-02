#!/usr/bin/env python3
"""
Emotional Attachment to AI: Two-Panel Visualization
====================================================
FAccT 2026 Submission

Left Panel: Distribution of emotional AI attitude survey responses
Right Panel: Variance explained (R²) by attachment dimensions

Author: UW HCI Research Team
Date: December 29, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from pathlib import Path

# ==============================================================================
# Configuration
# ==============================================================================

# Paths
DATA_DIR = Path("/Users/marxw/Sirius/data")
RESULTS_DIR = Path("/Users/marxw/Sirius/results/251020T1120_ecr_ai_attitudes_CORRECTED")
OUTPUT_DIR = Path("/Users/marxw/Sirius/FaccT_122925/vis")

# Survey column mapping (0-indexed from AI questions section)
# Based on survey.csv structure - AI questions start around column 63
AI_COLUMNS = {
    'AI_Trust': 'I trust ChatGPT to provide accurate information for my needs',
    'AI_FactCheck': 'I regularly fact-check or verify ChatGPT\'s responses',
    'AI_Confident_Implement': 'I feel confident implementing ChatGPT\'s suggestions without modification',
    'AI_Emotional_Understanding': 'I feel emotionally understood when interacting with ChatGPT',
    'AI_Emotional_Relief': 'I experience emotional relief after discussing personal matters with ChatGPT',
    'AI_Easier_Share': 'I find it easier to share personal struggles with ChatGPT than with people',
    'AI_Problem_Solving': 'Using ChatGPT has improved my problem-solving abilities',
    'AI_More_Capable': 'I feel more capable of tackling complex tasks with ChatGPT\'s assistance',
    'AI_More_Efficient': 'I\'ve become more efficient at completing tasks since using ChatGPT',
    'AI_Modify_Writing': 'I modify my writing style based on ChatGPT\'s suggestions',
    'AI_Learning_Different': 'I approach learning new concepts differently since using ChatGPT',
    'AI_Communication_Changed': 'ChatGPT has changed how I communicate professionally',
    'AI_Worry_Relying': 'I worry about relying too heavily on ChatGPT for tasks',
    'AI_Less_Confident': 'I feel less confident solving problems without ChatGPT\'s help',
    'AI_Prefer_Attempt_First': 'I prefer to attempt tasks on my own before consulting ChatGPT'
}

# Focus variables - SAME ORDER for both panels (top to bottom visually = last to first in list)
# Order: Easier to Share, Emotionally Understood, Emotional Relief, Trust AI, Less Confident
FOCUS_VARS = [
    'AI_Less_Confident',        # Bottom
    'AI_Trust',                 # 
    'AI_Emotional_Relief',      # 
    'AI_Emotional_Understanding', # 
    'AI_Easier_Share'           # Top
]

FOCUS_LABELS = {
    'AI_Easier_Share': 'Easier to Share\nwith AI',
    'AI_Emotional_Relief': 'Emotional Relief\nfrom AI',
    'AI_Emotional_Understanding': 'Emotionally\nUnderstood',
    'AI_Trust': 'Trust AI',
    'AI_Less_Confident': 'Less Confident\nWithout AI'
}

# Response scale (5-point Likert)
RESPONSE_LABELS = ['Strongly\nDisagree', 'Disagree', 'Neutral', 'Agree', 'Strongly\nAgree']

# Unified color palette - teal/blue gradient for cleaner look
# Each variable gets a consistent color across both panels
VAR_COLORS = {
    'AI_Easier_Share': '#2c7bb6',           # Deep blue - strongest effect
    'AI_Emotional_Understanding': '#41ae76', # Teal
    'AI_Emotional_Relief': '#78c679',        # Green
    'AI_Trust': '#c2e699',                   # Light green
    'AI_Less_Confident': '#ffffbf'           # Yellow/cream - weakest
}

# Likert response colors - diverging palette (red = disagree, blue = agree)
LIKERT_COLORS = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']

COLORS = {
    'sig_p001': '#2c7bb6',      # deep blue for p < 0.001
    'sig_p01': '#41ae76',       # teal for p < 0.01  
    'sig_p05': '#c2e699',       # light green for p < 0.05
    'ns': '#d9d9d9',            # light gray for non-significant
}

# ==============================================================================
# Data Loading
# ==============================================================================

def load_survey_data():
    """Load and preprocess survey data for AI attitude questions."""
    survey = pd.read_csv(DATA_DIR / "survey.csv")
    
    # Response mapping for Likert scale
    response_map = {
        'Strongly Disagree': 1,
        'Disagree': 2,
        'Neither Agree nor Disagree': 3,
        'Neutral': 3,
        'Agree': 4,
        'Strongly Agree': 5
    }
    
    # Find AI question columns by searching for key phrases
    ai_data = {}
    for col in survey.columns:
        # Match AI questions
        if 'I trust ChatGPT to provide accurate' in col:
            ai_data['AI_Trust'] = survey[col].map(response_map)
        elif 'I regularly fact-check' in col:
            ai_data['AI_FactCheck'] = survey[col].map(response_map)
        elif 'I feel confident implementing' in col:
            ai_data['AI_Confident_Implement'] = survey[col].map(response_map)
        elif 'I feel emotionally understood' in col:
            ai_data['AI_Emotional_Understanding'] = survey[col].map(response_map)
        elif 'I experience emotional relief' in col:
            ai_data['AI_Emotional_Relief'] = survey[col].map(response_map)
        elif 'I find it easier to share personal struggles' in col:
            ai_data['AI_Easier_Share'] = survey[col].map(response_map)
        elif 'improved my problem-solving' in col:
            ai_data['AI_Problem_Solving'] = survey[col].map(response_map)
        elif 'I feel more capable of tackling' in col:
            ai_data['AI_More_Capable'] = survey[col].map(response_map)
        elif 'become more efficient' in col:
            ai_data['AI_More_Efficient'] = survey[col].map(response_map)
        elif 'I modify my writing style' in col:
            ai_data['AI_Modify_Writing'] = survey[col].map(response_map)
        elif 'approach learning new concepts' in col:
            ai_data['AI_Learning_Different'] = survey[col].map(response_map)
        elif 'changed how I communicate professionally' in col:
            ai_data['AI_Communication_Changed'] = survey[col].map(response_map)
        elif 'I worry about relying too heavily' in col:
            ai_data['AI_Worry_Relying'] = survey[col].map(response_map)
        elif 'I feel less confident solving problems' in col:
            ai_data['AI_Less_Confident'] = survey[col].map(response_map)
        elif 'I prefer to attempt tasks on my own' in col:
            ai_data['AI_Prefer_Attempt_First'] = survey[col].map(response_map)
    
    return pd.DataFrame(ai_data)


def load_regression_results():
    """Load multiple regression results with R² values."""
    return pd.read_csv(RESULTS_DIR / "multiple_regressions.csv")


# ==============================================================================
# Visualization Functions
# ==============================================================================

def get_significance_color(p_value):
    """Return color based on significance level."""
    if p_value < 0.001:
        return COLORS['sig_p001']
    elif p_value < 0.01:
        return COLORS['sig_p01']
    elif p_value < 0.05:
        return COLORS['sig_p05']
    else:
        return COLORS['ns']


def plot_left_panel(ax, survey_data):
    """
    Left panel: Stacked horizontal bar chart showing response distributions
    for key emotional attachment indicators.
    """
    # Calculate response percentages for focus variables
    dist_data = []
    for var in FOCUS_VARS:
        if var in survey_data.columns:
            counts = survey_data[var].dropna().value_counts(normalize=True).sort_index()
            # Ensure all 5 response levels are present
            pcts = [counts.get(i, 0) * 100 for i in range(1, 6)]
            dist_data.append(pcts)
        else:
            dist_data.append([0, 0, 0, 0, 0])
    
    dist_data = np.array(dist_data)
    
    # Plot stacked horizontal bars
    y_pos = np.arange(len(FOCUS_VARS))
    bar_height = 0.65
    
    # Diverging color scheme: red (disagree) → yellow (neutral) → green (agree)
    colors = LIKERT_COLORS
    
    # Start positions for stacking
    left = np.zeros(len(FOCUS_VARS))
    
    for i, (pct, color, label) in enumerate(zip(dist_data.T, colors, RESPONSE_LABELS)):
        bars = ax.barh(y_pos, pct, bar_height, left=left, color=color, 
                       label=label.replace('\n', ' '), edgecolor='white', linewidth=0.8)
        
        # Add percentage labels for larger segments
        for j, (bar, p) in enumerate(zip(bars, pct)):
            if p > 12:  # Only label segments > 12%
                x_center = left[j] + p / 2
                text_color = 'white' if i in [0, 4] else '#333333'
                ax.text(x_center, j, f'{p:.0f}%', ha='center', va='center',
                       fontsize=8, fontweight='bold', color=text_color)
        
        left += pct
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FOCUS_LABELS[v] for v in FOCUS_VARS], fontsize=9, fontweight='medium')
    ax.set_xlabel('Percentage of Respondents', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(FOCUS_VARS) - 0.5)
    ax.set_title('Survey Response Distributions\n(N = 169)', fontsize=13, fontweight='bold', pad=12)
    
    # Add legend below
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5, 
              frameon=False, fontsize=8, columnspacing=1)
    
    # Add vertical line at 50% for reference
    ax.axvline(x=50, color='#888888', linestyle='--', linewidth=1, alpha=0.6)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)


def plot_right_panel(ax, regression_data):
    """
    Right panel: Horizontal bar chart showing R² values for FOCUS variables only,
    in the same order as left panel, with consistent colors.
    """
    # Filter to only focus variables and maintain same order
    df = regression_data[regression_data['AI_Attitude'].isin(FOCUS_VARS)].copy()
    
    # Set order to match left panel (FOCUS_VARS order)
    df['order'] = df['AI_Attitude'].map({v: i for i, v in enumerate(FOCUS_VARS)})
    df = df.sort_values('order')
    
    y_pos = np.arange(len(df))
    r_squared = df['R²'].values
    p_values = df['p_F'].values
    attitudes = df['AI_Attitude'].values
    
    # Use consistent colors matching left panel variable colors
    bar_colors = [VAR_COLORS[v] for v in attitudes]
    
    # Create bars
    bar_height = 0.65
    bars = ax.barh(y_pos, r_squared, color=bar_colors, edgecolor='white', 
                   linewidth=0.8, height=bar_height)
    
    # Add R² and statistics annotations
    for i, (bar, r2, p, f, n, var) in enumerate(zip(bars, r_squared, p_values, 
                                                     df['F'].values, df['N'].values, attitudes)):
        # Significance stars
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''
        
        # Right-side annotation - cleaner format
        annotation = f'R²={r2:.2f}{sig}\n(F={f:.1f})'
        ax.text(r2 + 0.008, i, annotation, ha='left', va='center', fontsize=9, 
               color='#333333', linespacing=1.2, fontweight='medium')
    
    # Formatting - use same labels as left panel
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FOCUS_LABELS[v] for v in attitudes], fontsize=9, fontweight='medium')
    ax.set_xlabel('R² (Variance Explained by Attachment)', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 0.30)
    ax.set_ylim(-0.5, len(FOCUS_VARS) - 0.5)
    ax.set_title('Attachment Predicts AI Attitudes\n(Anxiety + Avoidance → R²)', fontsize=13, fontweight='bold', pad=12)
    
    # Add significance legend in bottom right corner
    legend_text = '*** p<.001   ** p<.01   * p<.05'
    ax.text(0.98, 0.02, legend_text, ha='right', va='bottom', fontsize=8, 
           color='#555555', style='italic', transform=ax.transAxes)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add reference lines at 10% and 20%
    ax.axvline(x=0.10, color='#aaaaaa', linestyle=':', linewidth=1, alpha=0.7)
    ax.axvline(x=0.20, color='#aaaaaa', linestyle=':', linewidth=1, alpha=0.7)


def create_two_panel_figure():
    """Create the complete two-panel figure."""
    # Load data
    print("Loading survey data...")
    survey_data = load_survey_data()
    print(f"  Survey data shape: {survey_data.shape}")
    print(f"  Available columns: {list(survey_data.columns)}")
    
    print("\nLoading regression results...")
    regression_data = load_regression_results()
    print(f"  Regression data shape: {regression_data.shape}")
    
    # Create figure with two panels - equal width for easy comparison
    fig = plt.figure(figsize=(14, 5))
    
    # Use gridspec for precise layout control - equal widths
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.4)
    
    ax1 = fig.add_subplot(gs[0])  # Left: Distributions
    ax2 = fig.add_subplot(gs[1])  # Right: Variance explained
    
    # Plot panels
    print("\nCreating visualizations...")
    plot_left_panel(ax1, survey_data)
    plot_right_panel(ax2, regression_data)
    
    # Add panel labels
    fig.text(0.01, 0.92, 'A', fontsize=16, fontweight='bold')
    fig.text(0.51, 0.92, 'B', fontsize=16, fontweight='bold')
    
    # Main title
    fig.suptitle('Emotional Attachment to AI: Survey Responses & Attachment Predictors', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout
    fig.subplots_adjust(left=0.12, right=0.92, top=0.80, bottom=0.05, wspace=0.45)
    
    # Save figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save in multiple formats
    output_path_png = OUTPUT_DIR / "FIG_emotional_attachment_two_panel.png"
    
    fig.savefig(output_path_png, dpi=600, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Saved: {output_path_png}")
    
    plt.close()
    
    return fig


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Emotional Attachment to AI: Two-Panel Figure")
    print("FAccT 2026 Submission")
    print("=" * 60)
    
    create_two_panel_figure()
    
    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print("=" * 60)

