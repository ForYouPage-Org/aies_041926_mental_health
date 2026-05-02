#!/usr/bin/env python3
"""
ECR Attachment × AI Attitudes: Hierarchical Regression with Controls
Handles missing gender gracefully via mean imputation

Usage:
    python ecr_ai_correlation_regression_analysis.py              # Run full analysis
    python ecr_ai_correlation_regression_analysis.py --groups     # Run group analysis only
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, shapiro
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from datetime import datetime
import json
import argparse

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Argument parsing
parser = argparse.ArgumentParser(description='ECR × AI Attitudes Analysis')
parser.add_argument('--groups', action='store_true', help='Run group analysis (Cronbach alpha + composite correlations)')
parser.add_argument('--all', action='store_true', default=True, help='Run full analysis (default)')
args = parser.parse_args()

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "data"
ECR_DATA = BASE_DIR / "data" / "ecr_scores_validated_clean.csv"
SURVEY_DATA = BASE_DIR / "data" / "survey.csv"  # FaccT folder has the frequency data
SURVEY_SCORED = DATA_DIR / "survey_scored.csv"
DEMOGRAPHICS_DIR = DATA_DIR / "demographics"

TIMESTAMP = datetime.now().strftime("%y%m%dT%H%M")
OUTPUT_DIR = BASE_DIR / "results" / f"{TIMESTAMP}_ecr_ai_controlled_regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mappings
LIKERT_MAP = {
    "Strongly Disagree": 1, "Disagree": 2,
    "Neither Agree nor Disagree": 3, "Agree": 4, "Strongly Agree": 5
}

FREQ_MAP = {
    "Haven't used ChatGPT": 0, "Less than 5 times": 1,
    "5-9 times": 2, "10-19 times": 3, "20+ times": 4
}

GENDER_MAP = {'Male': 0, 'Female': 1}

AI_VARS_ALL = [f'AI_Q{i}' for i in range(1, 16)]

AI_LABELS_ALL = {
    'AI_Q1': 'Trust', 'AI_Q2': 'Fact-Check', 'AI_Q3': 'Confident Implement',
    'AI_Q4': 'Emotionally Understood', 'AI_Q5': 'Emotional Relief',
    'AI_Q6': 'Easier to Share', 'AI_Q7': 'Problem Solving',
    'AI_Q8': 'More Capable', 'AI_Q9': 'More Efficient',
    'AI_Q10': 'Modify Writing', 'AI_Q11': 'Learning Different',
    'AI_Q12': 'Communication Changed', 'AI_Q13': 'Worry Relying',
    'AI_Q14': 'Less Confident Without', 'AI_Q15': 'Prefer Attempt First'
}

# Composite and Single Item Variables (5 total)
COMPOSITE_VARS = {
    'Emotional_Engagement_Comp': {
        'items': ['AI_Q4', 'AI_Q5', 'AI_Q6'],
        'label': 'Emotional Engagement (Q4-6)',
        'description': 'Mean of: Emotionally Understood, Emotional Relief, Easier to Share'
    },
    'Self_Efficacy_Comp': {
        'items': ['AI_Q7', 'AI_Q8', 'AI_Q9'],
        'label': 'Self-Efficacy (Q7-9)',
        'description': 'Mean of: Problem Solving, More Capable, More Efficient'
    },
    'Behavior_Change_Comp': {
        'items': ['AI_Q10', 'AI_Q11', 'AI_Q12'],
        'label': 'Behavioral Change (Q10-12)',
        'description': 'Mean of: Modify Writing, Learning Different, Communication Changed'
    },
    'Trust_Single': {
        'items': ['AI_Q1'],
        'label': 'Trust (Q1)',
        'description': 'I trust the answers ChatGPT provides'
    },
    'Dependency_Single': {
        'items': ['AI_Q13'],
        'label': 'Dependency Concern (Q13)',
        'description': 'I worry I am relying on it too much'
    }
}

# Analysis variables (composites + singles)
AI_VARS = list(COMPOSITE_VARS.keys())
AI_LABELS = {k: v['label'] for k, v in COMPOSITE_VARS.items()}

# AI Attitude Groups (subscales) - for internal consistency analysis
AI_GROUPS = {
    'Emotional Engagement': ['AI_Q4', 'AI_Q5', 'AI_Q6'],
    'Trust & Verification': ['AI_Q1', 'AI_Q2', 'AI_Q3'],
    'Dependency': ['AI_Q13', 'AI_Q14', 'AI_Q15'],
    'Self-Efficacy': ['AI_Q7', 'AI_Q8', 'AI_Q9'],
    'Behavioral Change': ['AI_Q10', 'AI_Q11', 'AI_Q12']
}

def interpret_r(r):
    r_abs = abs(r)
    if r_abs < 0.10: return "negligible"
    elif r_abs < 0.30: return "small"
    elif r_abs < 0.50: return "medium"
    else: return "large"

def compute_vif(X):
    """
    Compute Variance Inflation Factor for each predictor.
    VIF > 5 suggests moderate multicollinearity, > 10 severe.
    """
    n_predictors = X.shape[1]
    vif_values = []
    
    for i in range(n_predictors):
        # Regress predictor i on all other predictors
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        
        if X_others.shape[1] == 0:
            vif_values.append(1.0)
            continue
        
        # Add constant
        X_others_const = np.column_stack([np.ones(len(X_others)), X_others])
        
        try:
            beta = np.linalg.lstsq(X_others_const, y_i, rcond=None)[0]
            y_pred = X_others_const @ beta
            ss_res = np.sum((y_i - y_pred)**2)
            ss_tot = np.sum((y_i - np.mean(y_i))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
        except:
            vif = np.nan
        
        vif_values.append(vif)
    
    return vif_values

def plot_residual_diagnostics(y, y_pred, residuals, outcome_name, output_dir):
    """
    Create diagnostic plots for regression assumptions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Residuals vs Fitted (Homoscedasticity)
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='white', linewidth=0.5)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted (Homoscedasticity)')
    
    # Add lowess smoother
    try:
        from scipy.ndimage import uniform_filter1d
        sorted_idx = np.argsort(y_pred)
        smoothed = uniform_filter1d(residuals[sorted_idx], size=max(10, len(residuals)//10))
        ax1.plot(y_pred[sorted_idx], smoothed, color='blue', linewidth=2, label='Trend')
    except:
        pass
    
    # 2. Q-Q Plot (Normality)
    ax2 = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality of Residuals)')
    
    # 3. Scale-Location (Sqrt of standardized residuals)
    ax3 = axes[1, 0]
    std_residuals = residuals / np.std(residuals)
    ax3.scatter(y_pred, np.sqrt(np.abs(std_residuals)), alpha=0.6, edgecolors='white', linewidth=0.5)
    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel('√|Standardized Residuals|')
    ax3.set_title('Scale-Location (Homoscedasticity)')
    
    # 4. Histogram of Residuals
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=20, edgecolor='white', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Residuals')
    
    # Shapiro-Wilk test result
    if len(residuals) < 5000:
        sw_stat, sw_p = shapiro(residuals)
        ax4.text(0.95, 0.95, f'Shapiro-Wilk\nW={sw_stat:.3f}\np={sw_p:.4f}',
                transform=ax4.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f'Regression Diagnostics: {outcome_name}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    safe_name = outcome_name.replace(' ', '_').replace('/', '_')
    fig.savefig(output_dir / f'diag_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return sw_stat if len(residuals) < 5000 else np.nan, sw_p if len(residuals) < 5000 else np.nan

def cronbach_alpha(items_df):
    """
    Compute Cronbach's alpha for internal consistency.
    items_df: DataFrame where each column is an item
    """
    items = items_df.dropna()
    n_items = items.shape[1]
    if n_items < 2:
        return np.nan
    
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    
    if total_var == 0:
        return np.nan
    
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return alpha

def compute_group_analysis(df, ecr_anxiety_col, ecr_avoidance_col, output_dir):
    """
    Compute correlations and Cronbach's alpha for each AI attitude group.
    Applies FDR correction for multiple comparisons (10 tests: 5 groups × 2 ECR dimensions).
    Returns DataFrame with results.
    """
    results = []
    all_p_values = []  # Collect all p-values for FDR correction
    
    for group_name, items in AI_GROUPS.items():
        item_cols = [f'{item}_numeric' for item in items]
        
        missing_cols = [c for c in item_cols if c not in df.columns]
        if missing_cols:
            print(f"  Warning: Missing columns for {group_name}: {missing_cols}")
            continue
        
        group_df = df[item_cols + [ecr_anxiety_col, ecr_avoidance_col]].dropna()
        n = len(group_df)
        
        if n < 3:
            print(f"  Warning: Insufficient data for {group_name} (n={n})")
            continue
        
        alpha = cronbach_alpha(group_df[item_cols])
        group_df['composite'] = group_df[item_cols].mean(axis=1)
        
        r_anxiety, p_anxiety = pearsonr(group_df[ecr_anxiety_col], group_df['composite'])
        r_avoidance, p_avoidance = pearsonr(group_df[ecr_avoidance_col], group_df['composite'])
        
        def r_to_ci(r, n, alpha=0.05):
            z = np.arctanh(r)
            se = 1 / np.sqrt(n - 3)
            z_crit = stats.norm.ppf(1 - alpha/2)
            return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)
        
        ci_anx = r_to_ci(r_anxiety, n)
        ci_avo = r_to_ci(r_avoidance, n)
        
        all_p_values.extend([p_anxiety, p_avoidance])
        
        results.append({
            'Group': group_name,
            'Items': ', '.join(items),
            'N_items': len(items),
            'N': n,
            'Cronbach_Alpha': alpha,
            'Alpha_Interpretation': 'Excellent' if alpha >= 0.9 else 'Good' if alpha >= 0.8 else 'Acceptable' if alpha >= 0.7 else 'Questionable' if alpha >= 0.6 else 'Poor',
            'Composite_M': group_df['composite'].mean(),
            'Composite_SD': group_df['composite'].std(),
            'r_Anxiety': r_anxiety,
            'r_Anxiety_CI': f'[{ci_anx[0]:.3f}, {ci_anx[1]:.3f}]',
            'p_Anxiety': p_anxiety,
            'r_Avoidance': r_avoidance,
            'r_Avoidance_CI': f'[{ci_avo[0]:.3f}, {ci_avo[1]:.3f}]',
            'p_Avoidance': p_avoidance
        })
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction (Benjamini-Hochberg)
    if len(all_p_values) > 0:
        fdr_corrected = benjamini_hochberg(np.array(all_p_values))
        
        # Assign corrected p-values back
        idx = 0
        for i in range(len(results_df)):
            results_df.loc[i, 'p_Anxiety_FDR'] = fdr_corrected[idx]
            results_df.loc[i, 'p_Avoidance_FDR'] = fdr_corrected[idx + 1]
            results_df.loc[i, 'sig_Anxiety'] = '***' if fdr_corrected[idx] < 0.001 else '**' if fdr_corrected[idx] < 0.01 else '*' if fdr_corrected[idx] < 0.05 else 'ns'
            results_df.loc[i, 'sig_Avoidance'] = '***' if fdr_corrected[idx+1] < 0.001 else '**' if fdr_corrected[idx+1] < 0.01 else '*' if fdr_corrected[idx+1] < 0.05 else 'ns'
            idx += 2
    
    results_df.to_csv(output_dir / "06_group_analysis.csv", index=False)
    
    return results_df

def compute_correlation_with_ci(x, y, alpha=0.05):
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]
    n = len(x_clean)
    if n < 3:
        return {'r': np.nan, 'p': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'n': n}
    r, p = pearsonr(x_clean, y_clean)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha/2)
    ci_low = np.tanh(z - z_crit * se)
    ci_high = np.tanh(z + z_crit * se)
    return {'r': r, 'p': p, 'ci_low': ci_low, 'ci_high': ci_high, 'n': n, 'r_squared': r**2}

def benjamini_hochberg(p_values):
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    adjusted_p = np.zeros(n)
    for i in range(n):
        adjusted_p[sorted_indices[i]] = min(1, sorted_p[i] * n / (sorted_indices.tolist().index(sorted_indices[i]) + 1))
    return adjusted_p

def run_regression(y, X):
    """OLS regression with full statistics"""
    X_with_const = np.column_stack([np.ones(len(X)), X])
    n, p = X_with_const.shape
    k = p - 1
    
    try:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    except:
        return None
    
    y_pred = X_with_const @ beta
    residuals = y - y_pred
    
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum(residuals**2)
    ss_regression = ss_total - ss_residual
    
    r_squared = ss_regression / ss_total if ss_total > 0 else 0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if (n - k - 1) > 0 else 0
    
    ms_regression = ss_regression / k if k > 0 else 0
    ms_residual = ss_residual / (n - k - 1) if (n - k - 1) > 0 else 1
    f_stat = ms_regression / ms_residual if ms_residual > 0 else 0
    f_p = 1 - stats.f.cdf(f_stat, k, n - k - 1)
    
    var_residual = ss_residual / (n - k - 1) if (n - k - 1) > 0 else 1
    try:
        var_beta = var_residual * np.linalg.inv(X_with_const.T @ X_with_const)
        se_beta = np.sqrt(np.diag(var_beta))
    except:
        se_beta = np.zeros(len(beta))
    
    t_stats = np.where(se_beta > 0, beta / se_beta, 0)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
    
    return {
        'n': n, 'k': k, 'r_squared': r_squared, 'adj_r_squared': adj_r_squared,
        'f_stat': f_stat, 'f_p': f_p, 'df1': k, 'df2': n - k - 1,
        'betas': beta[1:], 'se_betas': se_beta[1:],
        't_stats': t_stats[1:], 'p_values': p_values[1:], 'intercept': beta[0]
    }

print("=" * 70)
print("ECR × AI ATTITUDES: HIERARCHICAL REGRESSION (CONTROLLED)")
print("=" * 70)
print(f"Output: {OUTPUT_DIR}")

# =============================================================================
# STEP 1: LOAD AND MERGE DATA
# =============================================================================
print("\n--- Loading Data ---")

# ECR scores (197 unique)
ecr_df = pd.read_csv(ECR_DATA)
ecr_df['Prolific_ID'] = ecr_df['Prolific_ID'].astype(str).str.strip()
print(f"ECR: {len(ecr_df)} participants")

# Survey with frequency (from FaccT folder)
survey_df = pd.read_csv(SURVEY_DATA)
survey_df = survey_df.rename(columns={
    'What is your Prolific ID': 'Prolific_ID',
    'How often do you use ChatGPT in the past two weeks?': 'ChatGPT_Frequency'
})
survey_df['Prolific_ID'] = survey_df['Prolific_ID'].astype(str).str.strip()
survey_df['Frequency_numeric'] = survey_df['ChatGPT_Frequency'].map(FREQ_MAP)
survey_df = survey_df.drop_duplicates(subset='Prolific_ID', keep='first')
print(f"Survey: {len(survey_df)} unique, {survey_df['Frequency_numeric'].notna().sum()} with frequency")

# Survey scored (has AI attitudes) - REMOVE DUPLICATES
survey_scored = pd.read_csv(SURVEY_SCORED)
survey_scored['Prolific_ID'] = survey_scored['Prolific_ID'].astype(str).str.strip()
survey_scored = survey_scored.drop_duplicates(subset='Prolific_ID', keep='first')
print(f"Survey_scored: {len(survey_scored)} unique participants")

# Convert AI attitudes to numeric
for var in AI_VARS_ALL:
    if var in survey_scored.columns:
        survey_scored[f'{var}_numeric'] = survey_scored[var].map(LIKERT_MAP)

# Create composite variables
for comp_name, comp_info in COMPOSITE_VARS.items():
    item_cols = [f'{item}_numeric' for item in comp_info['items']]
    if len(item_cols) > 1:
        # Composite: mean of items
        survey_scored[comp_name] = survey_scored[item_cols].mean(axis=1)
    else:
        # Single item
        survey_scored[comp_name] = survey_scored[item_cols[0]]

# Load demographics (gender)
demo_dfs = []
for f in DEMOGRAPHICS_DIR.glob("prolific_export_*.csv"):
    df = pd.read_csv(f)
    if 'Participant id' in df.columns and 'Sex' in df.columns:
        demo_dfs.append(df[['Participant id', 'Sex']].copy())

demographics = pd.concat(demo_dfs, ignore_index=True).drop_duplicates(subset='Participant id', keep='first')
demographics = demographics.rename(columns={'Participant id': 'Prolific_ID'})
demographics['Prolific_ID'] = demographics['Prolific_ID'].astype(str).str.strip()
demographics = demographics[demographics['Sex'].isin(['Male', 'Female'])]
demographics['Gender_numeric'] = demographics['Sex'].map(GENDER_MAP)
print(f"Demographics: {len(demographics)} with valid gender")

# =============================================================================
# STEP 2: MERGE DATA
# =============================================================================
print("\n--- Merging Data ---")

# Merge: ECR + survey_scored + frequency + demographics
merged_df = ecr_df.merge(survey_scored, on='Prolific_ID', how='left')
merged_df = merged_df.merge(survey_df[['Prolific_ID', 'Frequency_numeric']], on='Prolific_ID', how='left')
merged_df = merged_df.merge(demographics[['Prolific_ID', 'Gender_numeric']], on='Prolific_ID', how='left')

# Use Age from survey_scored
if 'Age' not in merged_df.columns and 'Age_x' in merged_df.columns:
    merged_df['Age'] = merged_df['Age_x']

print(f"Merged: {len(merged_df)} ECR participants")
print(f"  With Age: {merged_df['Age'].notna().sum()}")
print(f"  With Frequency: {merged_df['Frequency_numeric'].notna().sum()}")
print(f"  With Gender: {merged_df['Gender_numeric'].notna().sum()}")

# Check AI attitudes (all 15 items needed for composites)
ai_numeric_cols_all = [f'{var}_numeric' for var in AI_VARS_ALL]
ai_complete = merged_df[ai_numeric_cols_all].notna().all(axis=1).sum()
print(f"  With all 15 AI attitudes: {ai_complete}")

# Composite variables
composite_cols = list(COMPOSITE_VARS.keys())

# =============================================================================
# STEP 3: HANDLE MISSING DATA
# =============================================================================
print("\n--- Handling Missing Data ---")

# Required: ECR + Age + Frequency + composite AI variables
required_cols = ['ECR_Anxiety', 'ECR_Avoidance', 'Age', 'Frequency_numeric'] + composite_cols
complete_mask = merged_df[required_cols].notna().all(axis=1)
analysis_df = merged_df[complete_mask].copy()

# Gender: mean imputation for missing values
n_missing_gender = analysis_df['Gender_numeric'].isna().sum()
gender_mean = analysis_df['Gender_numeric'].mean()
analysis_df['Gender_imputed'] = analysis_df['Gender_numeric'].fillna(gender_mean)
analysis_df['Gender_missing'] = analysis_df['Gender_numeric'].isna().astype(int)  # Flag for sensitivity analysis

N = len(analysis_df)
print(f"Complete cases (ECR + Age + Freq + AI): N = {N}")
print(f"  Gender observed: {N - n_missing_gender}")
print(f"  Gender imputed (mean={gender_mean:.2f}): {n_missing_gender}")

# Standardize continuous predictors
analysis_df['Age_z'] = (analysis_df['Age'] - analysis_df['Age'].mean()) / analysis_df['Age'].std()
analysis_df['Frequency_z'] = (analysis_df['Frequency_numeric'] - analysis_df['Frequency_numeric'].mean()) / analysis_df['Frequency_numeric'].std()
analysis_df['Anxiety_z'] = (analysis_df['ECR_Anxiety'] - analysis_df['ECR_Anxiety'].mean()) / analysis_df['ECR_Anxiety'].std()
analysis_df['Avoidance_z'] = (analysis_df['ECR_Avoidance'] - analysis_df['ECR_Avoidance'].mean()) / analysis_df['ECR_Avoidance'].std()

# =============================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n--- Descriptive Statistics ---")
descriptives = []

for var, label in [('ECR_Anxiety', 'Attachment Anxiety'), ('ECR_Avoidance', 'Attachment Avoidance'),
                   ('Age', 'Age'), ('Gender_imputed', 'Gender (0=M, 1=F)'), ('Frequency_numeric', 'ChatGPT Frequency')]:
    data = analysis_df[var]
    descriptives.append({
        'Variable': label, 'N': len(data), 'Mean': data.mean(), 'SD': data.std(),
        'Median': data.median(), 'Min': data.min(), 'Max': data.max()
    })
    print(f"  {label}: M={data.mean():.2f}, SD={data.std():.2f}")

# Composite and single variables
for var in AI_VARS:
    data = analysis_df[var]
    comp_info = COMPOSITE_VARS[var]
    n_items = len(comp_info['items'])
    descriptives.append({
        'Variable': AI_LABELS[var], 
        'N': len(data), 
        'Mean': data.mean(), 
        'SD': data.std(),
        'Median': data.median(), 
        'Min': data.min(), 
        'Max': data.max(),
        'N_items': n_items,
        'Items': ', '.join(comp_info['items'])
    })
    print(f"  {AI_LABELS[var]}: M={data.mean():.2f}, SD={data.std():.2f}")

pd.DataFrame(descriptives).to_csv(OUTPUT_DIR / "01_descriptive_statistics.csv", index=False)

# =============================================================================
# STEP 5: CORRELATIONS (ZERO-ORDER)
# =============================================================================
print("\n--- Zero-Order Correlations (Composites + Singles) ---")
anxiety_results, avoidance_results = [], []

for var in AI_VARS:
    # Use composite/single variable directly (not _numeric suffix)
    result = compute_correlation_with_ci(analysis_df['ECR_Anxiety'].values, analysis_df[var].values)
    result.update({
        'AI_Attitude': AI_LABELS[var], 
        'Variable': var, 
        'ECR_Dimension': 'Anxiety', 
        'Effect_Size': interpret_r(result['r']),
        'Items': ', '.join(COMPOSITE_VARS[var]['items'])
    })
    anxiety_results.append(result)
    
    result = compute_correlation_with_ci(analysis_df['ECR_Avoidance'].values, analysis_df[var].values)
    result.update({
        'AI_Attitude': AI_LABELS[var], 
        'Variable': var, 
        'ECR_Dimension': 'Avoidance', 
        'Effect_Size': interpret_r(result['r']),
        'Items': ', '.join(COMPOSITE_VARS[var]['items'])
    })
    avoidance_results.append(result)

anxiety_df = pd.DataFrame(anxiety_results)
anxiety_df['p_fdr'] = benjamini_hochberg(anxiety_df['p'].values)
anxiety_df['sig_fdr'] = anxiety_df['p_fdr'].apply(lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
anxiety_df.to_csv(OUTPUT_DIR / "02_anxiety_correlations.csv", index=False)

avoidance_df = pd.DataFrame(avoidance_results)
avoidance_df['p_fdr'] = benjamini_hochberg(avoidance_df['p'].values)
avoidance_df['sig_fdr'] = avoidance_df['p_fdr'].apply(lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
avoidance_df.to_csv(OUTPUT_DIR / "03_avoidance_correlations.csv", index=False)

n_sig_anx = (anxiety_df['sig_fdr'] != 'ns').sum()
n_sig_avo = (avoidance_df['sig_fdr'] != 'ns').sum()
print(f"  Anxiety: {n_sig_anx}/{len(AI_VARS)} significant (FDR corrected)")
print(f"  Avoidance: {n_sig_avo}/{len(AI_VARS)} significant (FDR corrected)")

# =============================================================================
# STEP 6: HIERARCHICAL REGRESSION WITH DIAGNOSTICS
# =============================================================================
print("\n--- Hierarchical Multiple Regression ---")
print("  Step 1: Controls (Age, Gender, ChatGPT Frequency)")
print("  Step 2: + ECR Anxiety, ECR Avoidance")
print(f"  Note: Gender imputed for {n_missing_gender} participants")

# Compute VIF for predictors (multicollinearity check)
X_full = analysis_df[['Age_z', 'Gender_imputed', 'Frequency_z', 'Anxiety_z', 'Avoidance_z']].values
predictor_names = ['Age', 'Gender', 'Frequency', 'Anxiety', 'Avoidance']
vif_values = compute_vif(X_full)

print("\n  VIF (Variance Inflation Factor) for multicollinearity:")
vif_results = []
for name, vif in zip(predictor_names, vif_values):
    interpretation = 'OK' if vif < 5 else 'Moderate' if vif < 10 else 'Severe'
    print(f"    {name}: VIF = {vif:.2f} ({interpretation})")
    vif_results.append({'Predictor': name, 'VIF': vif, 'Interpretation': interpretation})

pd.DataFrame(vif_results).to_csv(OUTPUT_DIR / "07_vif_multicollinearity.csv", index=False)

# Create diagnostics directory
diag_dir = OUTPUT_DIR / "diagnostics"
diag_dir.mkdir(exist_ok=True)

hierarchical_results = []
diagnostic_results = []

for var in AI_VARS:
    # Use composite/single variable directly
    y = analysis_df[var].values
    
    # Step 1: Controls only
    X_step1 = analysis_df[['Age_z', 'Gender_imputed', 'Frequency_z']].values
    result_step1 = run_regression(y, X_step1)
    
    # Step 2: Controls + ECR
    X_step2 = analysis_df[['Age_z', 'Gender_imputed', 'Frequency_z', 'Anxiety_z', 'Avoidance_z']].values
    result_step2 = run_regression(y, X_step2)
    
    if result_step1 and result_step2:
        delta_r2 = result_step2['r_squared'] - result_step1['r_squared']
        df1_change = 2
        df2 = result_step2['df2']
        f_change = (delta_r2 / df1_change) / ((1 - result_step2['r_squared']) / df2) if df2 > 0 else 0
        p_change = 1 - stats.f.cdf(f_change, df1_change, df2) if f_change > 0 else 1
        
        # Compute residuals for diagnostics
        X_const = np.column_stack([np.ones(len(X_step2)), X_step2])
        beta = np.linalg.lstsq(X_const, y, rcond=None)[0]
        y_pred = X_const @ beta
        residuals = y - y_pred
        
        # Shapiro-Wilk test for normality of residuals
        sw_stat, sw_p = shapiro(residuals) if len(residuals) < 5000 else (np.nan, np.nan)
        
        # Plot diagnostics
        plot_residual_diagnostics(y, y_pred, residuals, AI_LABELS[var], diag_dir)
        
        diagnostic_results.append({
            'AI_Attitude': AI_LABELS[var],
            'Shapiro_W': sw_stat,
            'Shapiro_p': sw_p,
            'Normality': 'Yes' if sw_p > 0.05 else 'No',
            'Residual_Mean': residuals.mean(),
            'Residual_SD': residuals.std()
        })
        
        hierarchical_results.append({
            'AI_Attitude': AI_LABELS[var], 
            'Variable': var, 
            'Items': ', '.join(COMPOSITE_VARS[var]['items']),
            'N': result_step2['n'],
            'Step1_R2': result_step1['r_squared'], 'Step1_AdjR2': result_step1['adj_r_squared'],
            'Step1_F': result_step1['f_stat'], 'Step1_F_p': result_step1['f_p'],
            'Beta_Age_Step1': result_step1['betas'][0],
            'Beta_Gender_Step1': result_step1['betas'][1],
            'Beta_Frequency_Step1': result_step1['betas'][2],
            'Step2_R2': result_step2['r_squared'], 'Step2_AdjR2': result_step2['adj_r_squared'],
            'Step2_F': result_step2['f_stat'], 'Step2_F_p': result_step2['f_p'],
            'Delta_R2': delta_r2, 'F_change': f_change, 'p_change': p_change,
            'Beta_Age': result_step2['betas'][0], 't_Age': result_step2['t_stats'][0], 'p_Age': result_step2['p_values'][0],
            'Beta_Gender': result_step2['betas'][1], 't_Gender': result_step2['t_stats'][1], 'p_Gender': result_step2['p_values'][1],
            'Beta_Frequency': result_step2['betas'][2], 't_Frequency': result_step2['t_stats'][2], 'p_Frequency': result_step2['p_values'][2],
            'Beta_Anxiety': result_step2['betas'][3], 't_Anxiety': result_step2['t_stats'][3], 'p_Anxiety': result_step2['p_values'][3],
            'Beta_Avoidance': result_step2['betas'][4], 't_Avoidance': result_step2['t_stats'][4], 'p_Avoidance': result_step2['p_values'][4]
        })

hierarchical_df = pd.DataFrame(hierarchical_results).sort_values('Step2_R2', ascending=False)

# Add FDR correction for p_change, p_Anxiety, p_Avoidance
hierarchical_df['p_change_FDR'] = benjamini_hochberg(hierarchical_df['p_change'].values)
hierarchical_df['sig_change'] = hierarchical_df['p_change_FDR'].apply(lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
hierarchical_df['p_Anxiety_FDR'] = benjamini_hochberg(hierarchical_df['p_Anxiety'].values)
hierarchical_df['p_Avoidance_FDR'] = benjamini_hochberg(hierarchical_df['p_Avoidance'].values)

hierarchical_df.to_csv(OUTPUT_DIR / "04_hierarchical_regression.csv", index=False)

# Save diagnostic results
diagnostic_df = pd.DataFrame(diagnostic_results)
diagnostic_df.to_csv(OUTPUT_DIR / "08_residual_diagnostics.csv", index=False)

n_vars = len(AI_VARS)
n_normal = (diagnostic_df['Normality'] == 'Yes').sum()
print(f"\n  Residual Normality (Shapiro-Wilk p > .05): {n_normal}/{n_vars} models")
print(f"  Diagnostic plots saved to: {diag_dir}")

n_sig_change = (hierarchical_df['sig_change'] != 'ns').sum()
print(f"\n  Models where ECR adds significant ΔR² (FDR corrected): {n_sig_change}/{n_vars}")
print(f"  Max ΔR²: {hierarchical_df['Delta_R2'].max():.3f}")
print(f"  Max total R²: {hierarchical_df['Step2_R2'].max():.3f}")

# =============================================================================
# STEP 7: ECR EFFECTS SUMMARY (CONTROLLING FOR DEMOGRAPHICS) WITH FDR
# =============================================================================
print("\n--- ECR Effects After Controlling for Age, Gender, Frequency ---")
print("  (FDR corrected p-values)")

ecr_controlled = []
for _, row in hierarchical_df.iterrows():
    # Use FDR-corrected p-values for significance
    p_anx_fdr = row['p_Anxiety_FDR']
    p_avo_fdr = row['p_Avoidance_FDR']
    p_change_fdr = row['p_change_FDR']
    
    ecr_controlled.append({
        'AI_Attitude': row['AI_Attitude'],
        'Variable': row['Variable'],
        'Items': row['Items'],
        'Beta_Anxiety': row['Beta_Anxiety'], 
        't_Anxiety': row['t_Anxiety'], 
        'p_Anxiety': row['p_Anxiety'],
        'p_Anxiety_FDR': p_anx_fdr,
        'sig_Anxiety': '***' if p_anx_fdr < 0.001 else '**' if p_anx_fdr < 0.01 else '*' if p_anx_fdr < 0.05 else 'ns',
        'Beta_Avoidance': row['Beta_Avoidance'], 
        't_Avoidance': row['t_Avoidance'], 
        'p_Avoidance': row['p_Avoidance'],
        'p_Avoidance_FDR': p_avo_fdr,
        'sig_Avoidance': '***' if p_avo_fdr < 0.001 else '**' if p_avo_fdr < 0.01 else '*' if p_avo_fdr < 0.05 else 'ns',
        'Delta_R2': row['Delta_R2'], 
        'F_change': row['F_change'], 
        'p_change': row['p_change'],
        'p_change_FDR': p_change_fdr,
        'sig_change': '***' if p_change_fdr < 0.001 else '**' if p_change_fdr < 0.01 else '*' if p_change_fdr < 0.05 else 'ns'
    })

ecr_controlled_df = pd.DataFrame(ecr_controlled)
ecr_controlled_df.to_csv(OUTPUT_DIR / "05_ecr_effects_controlled.csv", index=False)

n_sig_anx_ctrl = (ecr_controlled_df['sig_Anxiety'] != 'ns').sum()
n_sig_avo_ctrl = (ecr_controlled_df['sig_Avoidance'] != 'ns').sum()
print(f"  Anxiety significant (FDR): {n_sig_anx_ctrl}/{n_vars}")
print(f"  Avoidance significant (FDR): {n_sig_avo_ctrl}/{n_vars}")

# =============================================================================
# SAVE SUMMARY
# =============================================================================
summary = {
    'N': int(N),
    'n_variables': n_vars,
    'variables': list(AI_VARS),
    'N_gender_observed': int(N - n_missing_gender),
    'N_gender_imputed': int(n_missing_gender),
    'ECR_Anxiety_M': float(analysis_df['ECR_Anxiety'].mean()),
    'ECR_Anxiety_SD': float(analysis_df['ECR_Anxiety'].std()),
    'ECR_Avoidance_M': float(analysis_df['ECR_Avoidance'].mean()),
    'ECR_Avoidance_SD': float(analysis_df['ECR_Avoidance'].std()),
    'Age_M': float(analysis_df['Age'].mean()),
    'Age_SD': float(analysis_df['Age'].std()),
    'Gender_pct_female': float(analysis_df['Gender_imputed'].mean() * 100),
    'Frequency_M': float(analysis_df['Frequency_numeric'].mean()),
    'Frequency_SD': float(analysis_df['Frequency_numeric'].std()),
    'n_sig_anxiety_zeroorder_FDR': int(n_sig_anx),
    'n_sig_avoidance_zeroorder_FDR': int(n_sig_avo),
    'n_sig_anxiety_controlled_FDR': int(n_sig_anx_ctrl),
    'n_sig_avoidance_controlled_FDR': int(n_sig_avo_ctrl),
    'n_sig_delta_r2_FDR': int(n_sig_change),
    'max_delta_r2': float(hierarchical_df['Delta_R2'].max()),
    'timestamp': datetime.now().isoformat()
}

with open(OUTPUT_DIR / "summary_stats.json", 'w') as f:
    json.dump(summary, f, indent=2)

# =============================================================================
# STEP 8: GROUP ANALYSIS (Cronbach's Alpha & Composite Correlations)
# =============================================================================
print("\n--- AI Attitude Group Analysis ---")
print("  Computing Cronbach's alpha and composite correlations per subscale")

group_results = compute_group_analysis(analysis_df, 'ECR_Anxiety', 'ECR_Avoidance', OUTPUT_DIR)

print("\n  Group Results:")
print(f"  {'Group':<22} {'α':>6} {'r(Anx)':>8} {'r(Avo)':>8}")
print("  " + "-" * 50)
for _, row in group_results.iterrows():
    print(f"  {row['Group']:<22} {row['Cronbach_Alpha']:>6.2f} {row['r_Anxiety']:>7.2f}{row['sig_Anxiety']:<3} {row['r_Avoidance']:>7.2f}{row['sig_Avoidance']:<3}")

# Add group results to summary
summary['group_alphas'] = {row['Group']: round(row['Cronbach_Alpha'], 3) for _, row in group_results.iterrows()}

print(f"\n{'='*70}")
print(f"✓ Results saved to {OUTPUT_DIR}")
print(f"  N = {N} (gender imputed for {n_missing_gender})")
print(f"  Variables analyzed: {n_vars} (3 composites + 2 singles)")
print(f"  All p-values are FDR corrected (Benjamini-Hochberg)")
