#!/usr/bin/env python3
"""
Grouped Correlation Forest Plot: Attachment × AI Attitudes
===========================================================
FAccT 2026 Submission

Two-panel forest plot showing correlations between attachment dimensions
and AI attitudes, grouped by theoretical categories.

Author: UW HCI Research Team
Date: December 29, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ==============================================================================
# Configuration
# ==============================================================================

RESULTS_DIR = Path("/Users/marxw/Sirius/results/251020T1120_ecr_ai_attitudes_CORRECTED")
OUTPUT_DIR = Path("/Users/marxw/Sirius/FaccT_122925/vis")

# Variable groupings with display labels
GROUPS = {
    'Emotional Support & Self-Disclosure': [
        ('AI_Easier_Share', 'Easier to share struggles with AI'),
        ('AI_Emotional_Relief', 'Experience emotional relief'),
        ('AI_Emotional_Understanding', 'Feel emotionally understood'),
    ],
    'Trust & Verification': [
        ('AI_Trust', 'Trust AI for accurate information'),
        ('AI_FactCheck', 'Regularly fact-check AI responses'),
        ('AI_Confident_Implement', 'Confident implementing suggestions'),
    ],
    'Dependency': [
        ('AI_Less_Confident', 'Less confident without AI'),
        ('AI_Worry_Relying', 'Worry about over-reliance'),
        ('AI_Prefer_Attempt_First', 'Prefer attempting tasks alone'),
    ],
    'Self-Efficacy': [
        ('AI_Problem_Solving', 'Improved problem-solving'),
        ('AI_More_Capable', 'Feel more capable'),
        ('AI_More_Efficient', 'More efficient at tasks'),
    ],
    'Behavioral Change': [
        ('AI_Communication_Changed', 'Changed professional communication'),
        ('AI_Modify_Writing', 'Modify writing style'),
        ('AI_Learning_Different', 'Approach learning differently'),
    ],
}

# Colors matching original plot
COLORS = {
    'sig_p001': '#c44e52',      # dark red for p < 0.001 (***)
    'sig_p01': '#dd8452',       # orange for p < 0.01 (**)
    'sig_p05': '#f0c566',       # yellow for p < 0.05 (*)
    'ns': '#b0b0b0',            # gray for non-significant
    'group_bg': '#f5f5f5',      # light gray for group backgrounds
    'zero_line': '#666666',     # dark gray for zero reference line
    'threshold': '#cccccc',     # light gray for threshold lines
}

# ==============================================================================
# Data Loading
# ==============================================================================

def load_correlation_data():
    """Load anxiety and avoidance correlation results."""
    anxiety = pd.read_csv(RESULTS_DIR / "anxiety_correlations.csv")
    avoidance = pd.read_csv(RESULTS_DIR / "avoidance_correlations.csv")
    return anxiety, avoidance


def get_color(p_value):
    """Return color based on significance level."""
    if p_value < 0.001:
        return COLORS['sig_p001']
    elif p_value < 0.01:
        return COLORS['sig_p01']
    elif p_value < 0.05:
        return COLORS['sig_p05']
    else:
        return COLORS['ns']


def get_sig_stars(p_value):
    """Return significance stars."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_grouped_forest(ax, corr_data, title, show_group_labels=False):
    """
    Create a grouped forest plot for one attachment dimension.
    
    Parameters:
    -----------
    ax : matplotlib axis
    corr_data : DataFrame with correlation results
    title : str for subplot title
    show_group_labels : bool, whether to show group category labels
    """
    # Short group names for labels
    GROUP_SHORT_NAMES = {
        'Emotional Support & Self-Disclosure': 'EMOTIONAL SUPPORT',
        'Trust & Verification': 'TRUST & VERIFICATION',
        'Dependency': 'DEPENDENCY',
        'Self-Efficacy': 'SELF-EFFICACY',
        'Behavioral Change': 'BEHAVIORAL CHANGE',
    }
    
    # Build y-positions and data in grouped order
    # Include space for group headers
    y_positions = []
    labels = []
    r_values = []
    p_values = []
    colors = []
    group_spans = {}  # Track y-range for each group
    group_header_positions = {}  # Track header positions
    
    y = 0
    for group_name, variables in GROUPS.items():
        # Add space for group header
        group_header_positions[group_name] = y
        y += 0.6  # Space for header
        
        group_start = y
        for var_code, var_label in variables:
            row = corr_data[corr_data['AI_Attitude'] == var_code]
            if len(row) > 0:
                r = row['r'].values[0]
                p = row['p'].values[0]
                y_positions.append(y)
                labels.append(var_label)
                r_values.append(r)
                p_values.append(p)
                colors.append(get_color(p))
                y += 1
        group_spans[group_name] = (group_start, y - 1)
        y += 0.3  # Small gap between groups
    
    # Convert to arrays
    y_positions = np.array(y_positions)
    r_values = np.array(r_values)
    
    # Draw group headers and backgrounds
    for group_name, (start, end) in group_spans.items():
        # Find actual y positions for this group
        mask = (y_positions >= start) & (y_positions <= end)
        if np.any(mask):
            y_min = y_positions[mask].min() - 0.35
            y_max = y_positions[mask].max() + 0.35
            # Light background rectangle
            rect = plt.Rectangle((-0.35, y_min), 0.92, y_max - y_min, 
                                  facecolor=COLORS['group_bg'], edgecolor='none', 
                                  alpha=0.5, zorder=0)
            ax.add_patch(rect)
        
        # Add group header text
        header_y = group_header_positions[group_name]
        ax.text(0.1, header_y, GROUP_SHORT_NAMES[group_name],
               ha='center', va='center', fontsize=8, fontweight='bold',
               color='#444444', style='italic')
            
    
    # Draw reference lines
    ax.axvline(x=0, color=COLORS['zero_line'], linestyle='-', linewidth=1.2, zorder=1)
    ax.axvline(x=0.3, color=COLORS['threshold'], linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
    ax.axvline(x=-0.3, color=COLORS['threshold'], linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
    
    # Plot points with error bars (simplified - just points and lines)
    for i, (y_pos, r, p, color) in enumerate(zip(y_positions, r_values, p_values, colors)):
        # Horizontal line from 0 to r
        ax.plot([0, r], [y_pos, y_pos], color=color, linewidth=2, zorder=2)
        # Point marker
        ax.scatter(r, y_pos, s=100, c=color, edgecolors='white', linewidths=1, zorder=3)
        
        # Add significance stars and r value
        sig = get_sig_stars(p)
        if r >= 0:
            text_x = r + 0.02
            ha = 'left'
        else:
            text_x = r - 0.02
            ha = 'right'
        
        # Format: r=0.XX***
        label_text = f'r={r:.2f}{sig}'
        ax.text(text_x, y_pos, label_text, ha=ha, va='center', fontsize=8, 
               color='#333333', fontweight='medium')
    
    # Set labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Correlation coefficient (r)', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Set axis limits
    ax.set_xlim(-0.35, 0.55)
    # Account for header space at top
    first_header_y = list(group_header_positions.values())[0]
    ax.set_ylim(first_header_y - 0.5, y_positions.max() + 0.5)
    
    # Invert y-axis so first group is at top
    ax.invert_yaxis()
    
    # Style spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def add_group_labels(fig, ax):
    """Add group category labels on the left side."""
    # Calculate positions for group labels based on the inverted y-axis
    y = 0
    group_centers = {}
    for group_name, variables in GROUPS.items():
        group_start = y
        y += len(variables)
        group_end = y - 1
        group_centers[group_name] = (group_start + group_end) / 2
        y += 0.5
    
    # Transform to figure coordinates and add text
    # We'll add brackets/labels using the axis
    for group_name, center_y in group_centers.items():
        # Short group names for margin
        short_names = {
            'Emotional Support & Self-Disclosure': 'Emotional\nSupport',
            'Trust & Verification': 'Trust &\nVerification',
            'Dependency': 'Dependency',
            'Self-Efficacy': 'Self-\nEfficacy',
            'Behavioral Change': 'Behavioral\nChange',
        }


def create_grouped_forest_plot():
    """Create the complete two-panel grouped forest plot."""
    # Load data
    print("Loading correlation data...")
    anxiety_data, avoidance_data = load_correlation_data()
    
    # Create figure with extra space for group labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), sharey=False)
    
    # Plot both panels - show group labels on left panel only
    print("Creating forest plots...")
    plot_grouped_forest(ax1, anxiety_data, 'A. Attachment Anxiety', show_group_labels=True)
    plot_grouped_forest(ax2, avoidance_data, 'B. Attachment Avoidance', show_group_labels=False)
    
    # Add legend for significance levels
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['sig_p001'], edgecolor='white', label='p < 0.001'),
        mpatches.Patch(facecolor=COLORS['sig_p01'], edgecolor='white', label='p < 0.01'),
        mpatches.Patch(facecolor=COLORS['sig_p05'], edgecolor='white', label='p < 0.05'),
        mpatches.Patch(facecolor=COLORS['ns'], edgecolor='white', label='n.s.'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.02))
    
    # Main title
    fig.suptitle('Attachment Dimensions × AI Attitudes: Correlation Analysis', 
                 fontsize=14, fontweight='bold', y=0.96)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "FIG_grouped_correlation_forest.png"
    fig.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {output_path}")
    
    plt.close()
    return fig


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Grouped Correlation Forest Plot")
    print("FAccT 2026 Submission")
    print("=" * 60)
    
    create_grouped_forest_plot()
    
    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print("=" * 60)

