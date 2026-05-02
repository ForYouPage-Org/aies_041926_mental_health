#!/usr/bin/env python3
"""
ECR Attachment × LIWC-22 Analysis
=================================

Purpose:
- Analyze correlations between attachment styles (ECR-SF) and LIWC-22 dimensions
- Use official LIWC-22 output from filtered user prompts (< 20 words)

Date: December 29, 2025
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
LIWC_DATA_PATH = DATA_DIR / 'liwc22_user_prompts.csv'
ECR_DATA_PATH = DATA_DIR / 'ecr_scores_validated_clean.csv'
CHAT_IDS_PATH = DATA_DIR / 'chat_participants_list.csv'
SURVEY_SCORED_PATH = DATA_DIR / 'survey_scored.csv'
DEMOGRAPHICS_DIR = DATA_DIR / 'demographics'
GENDER_MAP = {'Male': 0, 'Female': 1}

# Output directory
timestamp = datetime.now().strftime('%y%m%dT%H%M')
OUTPUT_DIR = BASE_DIR / 'results' / f'{timestamp}_LIWC22_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)

# Visualization settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8


def significance_marker(p_value):
    """Return a conventional significance marker for a p-value."""
    if pd.isna(p_value):
        return ''
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''


def zscore(series):
    """Z-score a pandas Series, returning NaN when variance is zero."""
    std = series.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - series.mean()) / std


def run_standardized_ols(y, X):
    """Fit OLS on already standardized variables."""
    X_const = sm.add_constant(X, has_constant='add')
    return sm.OLS(y, X_const).fit()


def compute_model_change(step1_model, step2_model, df_num):
    """Compute the incremental F-test for nested models."""
    delta_r2 = max(step2_model.rsquared - step1_model.rsquared, 0.0)
    df_den = int(step2_model.df_resid)
    if delta_r2 <= 0 or df_den <= 0 or step2_model.rsquared >= 1:
        return delta_r2, np.nan, np.nan
    f_change = (delta_r2 / df_num) / ((1 - step2_model.rsquared) / df_den)
    p_change = stats.f.sf(f_change, df_num, df_den)
    return delta_r2, f_change, p_change

print("=" * 80)
print("LIWC-22 × ATTACHMENT STYLE ANALYSIS")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print()

# ==============================================================================
# STEP 1: Load LIWC-22 data
# ==============================================================================

print("STEP 1: Loading LIWC-22 data...")

liwc_df = pd.read_csv(LIWC_DATA_PATH)
print(f"  Loaded {len(liwc_df):,} rows")
print(f"  Columns: {len(liwc_df.columns)}")

# Key LIWC-22 variables to analyze (grouped by category)
LIWC_VARIABLES = {
    # Summary Variables
    'Summary': [
        ('Analytic', 'Analytical Thinking'),
        ('Clout', 'Social Status/Confidence'),
        ('Authentic', 'Authenticity'),
        ('Tone', 'Emotional Tone'),
    ],
    
    # Pronouns (key for attachment)
    'Pronouns': [
        ('i', 'I-words (self-focus)'),
        ('we', 'We-words'),
        ('you', 'You-words'),
        ('shehe', 'She/He'),
        ('they', 'They'),
    ],
    
    # Affect
    'Affect': [
        ('Affect', 'Total Affect'),
        ('tone_pos', 'Positive Tone'),
        ('tone_neg', 'Negative Tone'),
        ('emo_pos', 'Positive Emotion'),
        ('emo_neg', 'Negative Emotion'),
        ('emo_anx', 'Anxiety'),
        ('emo_anger', 'Anger'),
        ('emo_sad', 'Sadness'),
    ],
    
    # Cognition
    'Cognition': [
        ('Cognition', 'Total Cognition'),
        ('cogproc', 'Cognitive Processes'),
        ('insight', 'Insight'),
        ('cause', 'Causation'),
        ('discrep', 'Discrepancy'),
        ('tentat', 'Tentative'),
        ('certitude', 'Certainty'),
    ],
    
    # Drives
    'Drives': [
        ('Drives', 'Total Drives'),
        ('affiliation', 'Affiliation'),
        ('achieve', 'Achievement'),
        ('power', 'Power'),
    ],
    
    # Social
    'Social': [
        ('Social', 'Total Social'),
        ('socbehav', 'Social Behavior'),
        ('prosocial', 'Prosocial'),
        ('polite', 'Polite'),
        ('conflict', 'Conflict'),
        ('family', 'Family'),
        ('friend', 'Friend'),
    ],
    
    # Perception & Time
    'Perception': [
        ('Perception', 'Total Perception'),
        ('feeling', 'Feeling'),
        ('focuspast', 'Past Focus'),
        ('focuspresent', 'Present Focus'),
        ('focusfuture', 'Future Focus'),
    ],
    
    # Needs & Motives
    'Needs': [
        ('need', 'Need'),
        ('want', 'Want'),
        ('lack', 'Lack'),
        ('risk', 'Risk'),
        ('curiosity', 'Curiosity'),
        ('reward', 'Reward'),
    ],
}

# Flatten for analysis
all_liwc_vars = []
for category, vars_list in LIWC_VARIABLES.items():
    for var, label in vars_list:
        all_liwc_vars.append((var, label, category))

print(f"  Analyzing {len(all_liwc_vars)} LIWC-22 variables across {len(LIWC_VARIABLES)} categories")

# ==============================================================================
# STEP 2: Load and match participants
# ==============================================================================

print("\nSTEP 2: Loading and matching participants...")

ecr_df = pd.read_csv(ECR_DATA_PATH)
ecr_df['Prolific_ID'] = ecr_df['Prolific_ID'].astype(str).str.strip()
print(f"  ECR data: {len(ecr_df)} participants")

chat_ids = pd.read_csv(CHAT_IDS_PATH)
chat_ids['chat_participant_id'] = chat_ids['chat_participant_id'].astype(str).str.strip()
ecr_set = set(ecr_df['Prolific_ID'].astype(str))
chat_set = set(chat_ids['chat_participant_id'].astype(str))
matched_ids = list(ecr_set.intersection(chat_set))

print(f"  Matched participants (ECR + Chat): {len(matched_ids)}")

ecr_matched = ecr_df[ecr_df['Prolific_ID'].isin(matched_ids)].copy()

survey_scored = pd.read_csv(SURVEY_SCORED_PATH)
survey_scored['Prolific_ID'] = survey_scored['Prolific_ID'].astype(str).str.strip()
survey_scored = survey_scored.drop_duplicates(subset='Prolific_ID', keep='first')
print(f"  Survey scored data: {len(survey_scored)} unique participants")

demo_frames = []
for pattern in ('prolific_export_*.csv', 'prolific_demographic_export_*.csv'):
    for demo_path in sorted(Path(DEMOGRAPHICS_DIR).glob(pattern)):
        demo_df = pd.read_csv(demo_path)
        if 'Participant id' in demo_df.columns and 'Sex' in demo_df.columns:
            demo_frames.append(demo_df[['Participant id', 'Sex']].copy())

if demo_frames:
    demographics = pd.concat(demo_frames, ignore_index=True).drop_duplicates(
        subset='Participant id',
        keep='first'
    )
    demographics = demographics.rename(columns={'Participant id': 'Prolific_ID'})
    demographics['Prolific_ID'] = demographics['Prolific_ID'].astype(str).str.strip()
    demographics = demographics[demographics['Sex'].isin(['Male', 'Female'])].copy()
    demographics['Gender_numeric'] = demographics['Sex'].map(GENDER_MAP)
else:
    demographics = pd.DataFrame(columns=['Prolific_ID', 'Gender_numeric'])

print(f"  Demographics with valid gender: {len(demographics)}")

# ==============================================================================
# STEP 3: Aggregate LIWC scores by participant
# ==============================================================================

print("\nSTEP 3: Aggregating LIWC-22 scores by participant...")

# Get only the LIWC variable columns we need
liwc_var_names = [var for var, label, cat in all_liwc_vars]

# Check which columns exist
existing_vars = [v for v in liwc_var_names if v in liwc_df.columns]
missing_vars = [v for v in liwc_var_names if v not in liwc_df.columns]

if missing_vars:
    print(f"  Warning: {len(missing_vars)} variables not found: {missing_vars[:5]}...")

# Aggregate by participant (mean of all messages)
agg_dict = {var: 'mean' for var in existing_vars}
agg_dict['WC'] = 'sum'  # Total word count
agg_dict['message'] = 'count'  # Number of messages

participant_liwc = liwc_df.groupby('participant_id').agg(agg_dict).reset_index()
participant_liwc.rename(columns={'message': 'n_messages', 'WC': 'total_words'}, inplace=True)
participant_liwc['participant_id'] = participant_liwc['participant_id'].astype(str).str.strip()

print(f"  Aggregated LIWC scores for {len(participant_liwc)} participants")

# ==============================================================================
# STEP 4: Merge with ECR attachment scores
# ==============================================================================

print("\nSTEP 4: Merging LIWC data with ECR attachment scores...")

merged_df = ecr_matched.merge(
    participant_liwc,
    left_on='Prolific_ID',
    right_on='participant_id',
    how='inner'
)
merged_df = merged_df.merge(
    survey_scored[['Prolific_ID', 'Age']],
    on='Prolific_ID',
    how='left'
)
merged_df = merged_df.merge(
    demographics[['Prolific_ID', 'Gender_numeric']],
    on='Prolific_ID',
    how='left'
)

merged_df['log_n_messages'] = np.log1p(merged_df['n_messages'])
analysis_df = merged_df.dropna(subset=['ECR_Anxiety', 'ECR_Avoidance', 'Age', 'log_n_messages']).copy()
n_missing_gender = analysis_df['Gender_numeric'].isna().sum()
gender_mean = analysis_df['Gender_numeric'].mean()
if pd.isna(gender_mean):
    gender_mean = 0.5
analysis_df['Gender_imputed'] = analysis_df['Gender_numeric'].fillna(gender_mean)
analysis_df['Gender_missing'] = analysis_df['Gender_numeric'].isna().astype(int)
analysis_df['Age_z'] = zscore(analysis_df['Age'])
analysis_df['MessageVolume_z'] = zscore(analysis_df['log_n_messages'])
analysis_df['Anxiety_z'] = zscore(analysis_df['ECR_Anxiety'])
analysis_df['Avoidance_z'] = zscore(analysis_df['ECR_Avoidance'])

print(f"  Final sample: N = {len(merged_df)} participants")
print(f"  With Age: {merged_df['Age'].notna().sum()}")
print(f"  With Gender: {merged_df['Gender_numeric'].notna().sum()}")
print(f"  Controlled-analysis sample: N = {len(analysis_df)}")
print(f"  Gender observed: {len(analysis_df) - n_missing_gender}")
print(f"  Gender imputed (mean={gender_mean:.2f}): {n_missing_gender}")

# ==============================================================================
# STEP 5: Correlational analysis with FDR correction
# ==============================================================================

print("\nSTEP 5: Correlational analysis (Attachment × LIWC-22)...")

# Correlations with Attachment Anxiety
anxiety_results = []
for var, label, category in all_liwc_vars:
    if var not in merged_df.columns:
        continue
    valid = merged_df[['ECR_Anxiety', var]].dropna()
    if len(valid) > 10:
        r, p = pearsonr(valid['ECR_Anxiety'], valid[var])
        anxiety_results.append({
            'Variable': var,
            'Label': label,
            'Category': category,
            'N': len(valid),
            'r': r,
            'p_uncorrected': p,
            'r_squared': r**2
        })

# Correlations with Attachment Avoidance
avoidance_results = []
for var, label, category in all_liwc_vars:
    if var not in merged_df.columns:
        continue
    valid = merged_df[['ECR_Avoidance', var]].dropna()
    if len(valid) > 10:
        r, p = pearsonr(valid['ECR_Avoidance'], valid[var])
        avoidance_results.append({
            'Variable': var,
            'Label': label,
            'Category': category,
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
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    anxiety_df['sig_fdr'] = anxiety_df['p_fdr'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )

if len(avoidance_df) > 0:
    _, avoidance_df['p_fdr'], _, _ = multipletests(avoidance_df['p_uncorrected'], method='fdr_bh')
    avoidance_df['sig_uncorrected'] = avoidance_df['p_uncorrected'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    avoidance_df['sig_fdr'] = avoidance_df['p_fdr'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )

# Sort by absolute correlation
anxiety_df['abs_r'] = anxiety_df['r'].abs()
avoidance_df['abs_r'] = avoidance_df['r'].abs()
anxiety_df = anxiety_df.sort_values('abs_r', ascending=False)
avoidance_df = avoidance_df.sort_values('abs_r', ascending=False)

# Count significant results
n_sig_anx = len(anxiety_df[anxiety_df['sig_uncorrected'] != ''])
n_sig_avo = len(avoidance_df[avoidance_df['sig_uncorrected'] != ''])
n_sig_anx_fdr = len(anxiety_df[anxiety_df['sig_fdr'] != ''])
n_sig_avo_fdr = len(avoidance_df[avoidance_df['sig_fdr'] != ''])

print(f"\n  === ATTACHMENT ANXIETY × LIWC-22 ===")
print(f"  Significant (uncorrected): {n_sig_anx}/{len(anxiety_df)}")
print(f"  Significant (FDR): {n_sig_anx_fdr}/{len(anxiety_df)}")
print("\n  Top 10 correlations:")
for _, row in anxiety_df.head(10).iterrows():
    print(f"    {row['Label']:<25} r = {row['r']:>7.3f} {row['sig_uncorrected']:<3} (FDR: {row['sig_fdr']})")

print(f"\n  === ATTACHMENT AVOIDANCE × LIWC-22 ===")
print(f"  Significant (uncorrected): {n_sig_avo}/{len(avoidance_df)}")
print(f"  Significant (FDR): {n_sig_avo_fdr}/{len(avoidance_df)}")
print("\n  Top 10 correlations:")
for _, row in avoidance_df.head(10).iterrows():
    print(f"    {row['Label']:<25} r = {row['r']:>7.3f} {row['sig_uncorrected']:<3} (FDR: {row['sig_fdr']})")

# ==============================================================================
# STEP 6: Controlled regression with age, gender, and message-volume covariates
# ==============================================================================

print("\nSTEP 6: Controlled regression (adjusting for age, gender, and message volume)...")
print("  Step 1: Age + Gender + log1p(number of short user messages)")
print("  Step 2: + Attachment Anxiety + Attachment Avoidance")
print(f"  Note: Gender mean-imputed for {n_missing_gender} participants")

controlled_results = []
for var, label, category in all_liwc_vars:
    if var not in merged_df.columns:
        continue

    valid = analysis_df[
        ['Age_z', 'Gender_imputed', 'MessageVolume_z', 'Anxiety_z', 'Avoidance_z', var]
    ].dropna().copy()
    if len(valid) <= 10:
        continue

    valid['y_z'] = zscore(valid[var])
    valid = valid.dropna()

    if len(valid) <= 10:
        continue

    step1_model = run_standardized_ols(
        valid['y_z'],
        valid[['Age_z', 'Gender_imputed', 'MessageVolume_z']]
    )
    step2_model = run_standardized_ols(
        valid['y_z'],
        valid[['Age_z', 'Gender_imputed', 'MessageVolume_z', 'Anxiety_z', 'Avoidance_z']]
    )
    delta_r2, f_change, p_change = compute_model_change(step1_model, step2_model, df_num=2)

    controlled_results.append({
        'Variable': var,
        'Label': label,
        'Category': category,
        'N': len(valid),
        'Step1_R2': step1_model.rsquared,
        'Step2_R2': step2_model.rsquared,
        'Delta_R2': delta_r2,
        'F_change': f_change,
        'p_change': p_change,
        'Step1_F': step1_model.fvalue,
        'Step1_F_p': step1_model.f_pvalue,
        'Step2_F': step2_model.fvalue,
        'Step2_F_p': step2_model.f_pvalue,
        'Beta_Age': step2_model.params['Age_z'],
        't_Age': step2_model.tvalues['Age_z'],
        'p_Age': step2_model.pvalues['Age_z'],
        'Beta_Gender': step2_model.params['Gender_imputed'],
        't_Gender': step2_model.tvalues['Gender_imputed'],
        'p_Gender': step2_model.pvalues['Gender_imputed'],
        'Beta_MessageVolume': step2_model.params['MessageVolume_z'],
        't_MessageVolume': step2_model.tvalues['MessageVolume_z'],
        'p_MessageVolume': step2_model.pvalues['MessageVolume_z'],
        'Beta_Anxiety': step2_model.params['Anxiety_z'],
        't_Anxiety': step2_model.tvalues['Anxiety_z'],
        'p_Anxiety': step2_model.pvalues['Anxiety_z'],
        'Beta_Avoidance': step2_model.params['Avoidance_z'],
        't_Avoidance': step2_model.tvalues['Avoidance_z'],
        'p_Avoidance': step2_model.pvalues['Avoidance_z'],
    })

controlled_df = pd.DataFrame(controlled_results)
anxiety_controlled_df = pd.DataFrame()
avoidance_controlled_df = pd.DataFrame()
n_sig_change_fdr = 0
n_sig_age_fdr = 0
n_sig_gender_fdr = 0
n_sig_volume_fdr = 0
n_sig_anx_ctrl_fdr = 0
n_sig_avo_ctrl_fdr = 0

if len(controlled_df) > 0:
    _, controlled_df['p_Age_fdr'], _, _ = multipletests(
        controlled_df['p_Age'].fillna(1.0), method='fdr_bh'
    )
    _, controlled_df['p_Gender_fdr'], _, _ = multipletests(
        controlled_df['p_Gender'].fillna(1.0), method='fdr_bh'
    )
    _, controlled_df['p_change_fdr'], _, _ = multipletests(
        controlled_df['p_change'].fillna(1.0), method='fdr_bh'
    )
    _, controlled_df['p_MessageVolume_fdr'], _, _ = multipletests(
        controlled_df['p_MessageVolume'].fillna(1.0), method='fdr_bh'
    )
    _, controlled_df['p_Anxiety_fdr'], _, _ = multipletests(
        controlled_df['p_Anxiety'].fillna(1.0), method='fdr_bh'
    )
    _, controlled_df['p_Avoidance_fdr'], _, _ = multipletests(
        controlled_df['p_Avoidance'].fillna(1.0), method='fdr_bh'
    )

    controlled_df['sig_Age'] = controlled_df['p_Age_fdr'].apply(significance_marker)
    controlled_df['sig_Gender'] = controlled_df['p_Gender_fdr'].apply(significance_marker)
    controlled_df['sig_change'] = controlled_df['p_change_fdr'].apply(significance_marker)
    controlled_df['sig_MessageVolume'] = controlled_df['p_MessageVolume_fdr'].apply(significance_marker)
    controlled_df['sig_Anxiety'] = controlled_df['p_Anxiety_fdr'].apply(significance_marker)
    controlled_df['sig_Avoidance'] = controlled_df['p_Avoidance_fdr'].apply(significance_marker)
    controlled_df['abs_beta_anxiety'] = controlled_df['Beta_Anxiety'].abs()
    controlled_df['abs_beta_avoidance'] = controlled_df['Beta_Avoidance'].abs()

    anxiety_controlled_df = controlled_df.sort_values('abs_beta_anxiety', ascending=False).copy()
    avoidance_controlled_df = controlled_df.sort_values('abs_beta_avoidance', ascending=False).copy()

    n_sig_age_fdr = len(controlled_df[controlled_df['sig_Age'] != ''])
    n_sig_gender_fdr = len(controlled_df[controlled_df['sig_Gender'] != ''])
    n_sig_change_fdr = len(controlled_df[controlled_df['sig_change'] != ''])
    n_sig_volume_fdr = len(controlled_df[controlled_df['sig_MessageVolume'] != ''])
    n_sig_anx_ctrl_fdr = len(controlled_df[controlled_df['sig_Anxiety'] != ''])
    n_sig_avo_ctrl_fdr = len(controlled_df[controlled_df['sig_Avoidance'] != ''])

print(f"  Age effects across LIWC features (FDR): {n_sig_age_fdr}/{len(controlled_df)}")
print(f"  Gender effects across LIWC features (FDR): {n_sig_gender_fdr}/{len(controlled_df)}")
print(f"  Message-volume effects across LIWC features (FDR): {n_sig_volume_fdr}/{len(controlled_df)}")
print(f"  Attachment adds significant Delta R^2 (FDR): {n_sig_change_fdr}/{len(controlled_df)}")
print(f"  Anxiety effects controlling for age, gender, and message volume (FDR): {n_sig_anx_ctrl_fdr}/{len(controlled_df)}")
print(f"  Avoidance effects controlling for age, gender, and message volume (FDR): {n_sig_avo_ctrl_fdr}/{len(controlled_df)}")

if len(anxiety_controlled_df) > 0:
    print("\n  Top 10 controlled anxiety effects:")
    for _, row in anxiety_controlled_df.head(10).iterrows():
        print(
            f"    {row['Label']:<25} beta = {row['Beta_Anxiety']:>7.3f} "
            f"{row['sig_Anxiety']:<3} (FDR p = {row['p_Anxiety_fdr']:.4f})"
        )

if len(avoidance_controlled_df) > 0:
    print("\n  Top 10 controlled avoidance effects:")
    for _, row in avoidance_controlled_df.head(10).iterrows():
        print(
            f"    {row['Label']:<25} beta = {row['Beta_Avoidance']:>7.3f} "
            f"{row['sig_Avoidance']:<3} (FDR p = {row['p_Avoidance_fdr']:.4f})"
        )

# ==============================================================================
# STEP 7: Save key data tables as TXT
# ==============================================================================

print("\nSTEP 7: Saving key data tables...")

# Table 1: Anxiety correlations (sorted by |r|)
table1_path = f'{OUTPUT_DIR}/TABLE1_anxiety_liwc22_correlations.txt'
with open(table1_path, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("TABLE 1: Attachment Anxiety × LIWC-22 Dimensions\n")
    f.write("=" * 100 + "\n")
    f.write(f"N = {anxiety_df['N'].iloc[0]} participants\n")
    f.write(f"LIWC-22 (Boyd et al., 2022)\n")
    f.write(f"FDR correction applied (Benjamini-Hochberg)\n")
    f.write(f"Significant (uncorrected p < .05): {n_sig_anx}/{len(anxiety_df)}\n")
    f.write(f"Significant (FDR q < .05): {n_sig_anx_fdr}/{len(anxiety_df)}\n")
    f.write("-" * 100 + "\n\n")
    f.write(f"{'Category':<15} {'LIWC Variable':<25} {'r':>8} {'p':>10} {'Sig':>5} {'p_FDR':>10} {'FDR':>5} {'R²':>8}\n")
    f.write("-" * 100 + "\n")
    for _, row in anxiety_df.iterrows():
        f.write(f"{row['Category']:<15} {row['Label']:<25} {row['r']:>8.3f} {row['p_uncorrected']:>10.4f} {row['sig_uncorrected']:>5} {row['p_fdr']:>10.4f} {row['sig_fdr']:>5} {row['r_squared']:>8.3f}\n")
    f.write("-" * 100 + "\n")
    f.write("\nNote: *** p<.001, ** p<.01, * p<.05\n")
    f.write("      Sorted by absolute correlation strength\n")

print(f"  ✓ Saved: {table1_path}")

# Table 2: Avoidance correlations (sorted by |r|)
table2_path = f'{OUTPUT_DIR}/TABLE2_avoidance_liwc22_correlations.txt'
with open(table2_path, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("TABLE 2: Attachment Avoidance × LIWC-22 Dimensions\n")
    f.write("=" * 100 + "\n")
    f.write(f"N = {avoidance_df['N'].iloc[0]} participants\n")
    f.write(f"LIWC-22 (Boyd et al., 2022)\n")
    f.write(f"FDR correction applied (Benjamini-Hochberg)\n")
    f.write(f"Significant (uncorrected p < .05): {n_sig_avo}/{len(avoidance_df)}\n")
    f.write(f"Significant (FDR q < .05): {n_sig_avo_fdr}/{len(avoidance_df)}\n")
    f.write("-" * 100 + "\n\n")
    f.write(f"{'Category':<15} {'LIWC Variable':<25} {'r':>8} {'p':>10} {'Sig':>5} {'p_FDR':>10} {'FDR':>5} {'R²':>8}\n")
    f.write("-" * 100 + "\n")
    for _, row in avoidance_df.iterrows():
        f.write(f"{row['Category']:<15} {row['Label']:<25} {row['r']:>8.3f} {row['p_uncorrected']:>10.4f} {row['sig_uncorrected']:>5} {row['p_fdr']:>10.4f} {row['sig_fdr']:>5} {row['r_squared']:>8.3f}\n")
    f.write("-" * 100 + "\n")
    f.write("\nNote: *** p<.001, ** p<.01, * p<.05\n")
    f.write("      Sorted by absolute correlation strength\n")

print(f"  ✓ Saved: {table2_path}")

# ==============================================================================
# Table 3: Anxiety controlled effects
table3_path = f'{OUTPUT_DIR}/TABLE3_anxiety_liwc22_controlled.txt'
with open(table3_path, 'w') as f:
    f.write("=" * 160 + "\n")
    f.write("TABLE 3: Attachment Anxiety × LIWC-22 Dimensions (Controlling for Age, Gender, and Message Volume)\n")
    f.write("=" * 160 + "\n")
    if len(anxiety_controlled_df) > 0:
        f.write(f"N = {anxiety_controlled_df['N'].iloc[0]} participants\n")
    f.write("Model: LIWC_z ~ Age_z + Gender_imputed + log1p(n_messages)_z + Anxiety_z + Avoidance_z\n")
    f.write(f"Gender missing values mean-imputed: {n_missing_gender}\n")
    f.write("FDR correction applied to Anxiety coefficients (Benjamini-Hochberg)\n")
    f.write(f"Significant Anxiety effects (FDR q < .05): {n_sig_anx_ctrl_fdr}/{len(anxiety_controlled_df)}\n")
    f.write(f"Significant model change (FDR q < .05): {n_sig_change_fdr}/{len(anxiety_controlled_df)}\n")
    f.write("-" * 160 + "\n\n")
    f.write(
        f"{'Category':<15} {'LIWC Variable':<25} {'Beta':>8} {'t':>8} {'p':>10} "
        f"{'p_FDR':>10} {'FDR':>5} {'Delta_R2':>10} {'beta_age':>10} "
        f"{'beta_gender':>12} {'beta_vol':>10}\n"
    )
    f.write("-" * 160 + "\n")
    for _, row in anxiety_controlled_df.iterrows():
        f.write(
            f"{row['Category']:<15} {row['Label']:<25} {row['Beta_Anxiety']:>8.3f} "
            f"{row['t_Anxiety']:>8.3f} {row['p_Anxiety']:>10.4f} {row['p_Anxiety_fdr']:>10.4f} "
            f"{row['sig_Anxiety']:>5} {row['Delta_R2']:>10.3f} {row['Beta_Age']:>10.3f} "
            f"{row['Beta_Gender']:>12.3f} {row['Beta_MessageVolume']:>10.3f}\n"
        )
    f.write("-" * 160 + "\n")
    f.write("\nNote: *** p<.001, ** p<.01, * p<.05\n")
    f.write("      Beta values are standardized regression coefficients\n")
    f.write("      Gender coded 0=male, 1=female; missing values mean-imputed\n")
    f.write("      Sorted by absolute Anxiety beta\n")

print(f"  ✓ Saved: {table3_path}")

# Table 4: Avoidance controlled effects
table4_path = f'{OUTPUT_DIR}/TABLE4_avoidance_liwc22_controlled.txt'
with open(table4_path, 'w') as f:
    f.write("=" * 160 + "\n")
    f.write("TABLE 4: Attachment Avoidance × LIWC-22 Dimensions (Controlling for Age, Gender, and Message Volume)\n")
    f.write("=" * 160 + "\n")
    if len(avoidance_controlled_df) > 0:
        f.write(f"N = {avoidance_controlled_df['N'].iloc[0]} participants\n")
    f.write("Model: LIWC_z ~ Age_z + Gender_imputed + log1p(n_messages)_z + Anxiety_z + Avoidance_z\n")
    f.write(f"Gender missing values mean-imputed: {n_missing_gender}\n")
    f.write("FDR correction applied to Avoidance coefficients (Benjamini-Hochberg)\n")
    f.write(f"Significant Avoidance effects (FDR q < .05): {n_sig_avo_ctrl_fdr}/{len(avoidance_controlled_df)}\n")
    f.write(f"Significant model change (FDR q < .05): {n_sig_change_fdr}/{len(avoidance_controlled_df)}\n")
    f.write("-" * 160 + "\n\n")
    f.write(
        f"{'Category':<15} {'LIWC Variable':<25} {'Beta':>8} {'t':>8} {'p':>10} "
        f"{'p_FDR':>10} {'FDR':>5} {'Delta_R2':>10} {'beta_age':>10} "
        f"{'beta_gender':>12} {'beta_vol':>10}\n"
    )
    f.write("-" * 160 + "\n")
    for _, row in avoidance_controlled_df.iterrows():
        f.write(
            f"{row['Category']:<15} {row['Label']:<25} {row['Beta_Avoidance']:>8.3f} "
            f"{row['t_Avoidance']:>8.3f} {row['p_Avoidance']:>10.4f} {row['p_Avoidance_fdr']:>10.4f} "
            f"{row['sig_Avoidance']:>5} {row['Delta_R2']:>10.3f} {row['Beta_Age']:>10.3f} "
            f"{row['Beta_Gender']:>12.3f} {row['Beta_MessageVolume']:>10.3f}\n"
        )
    f.write("-" * 160 + "\n")
    f.write("\nNote: *** p<.001, ** p<.01, * p<.05\n")
    f.write("      Beta values are standardized regression coefficients\n")
    f.write("      Gender coded 0=male, 1=female; missing values mean-imputed\n")
    f.write("      Sorted by absolute Avoidance beta\n")

print(f"  ✓ Saved: {table4_path}")

# ==============================================================================
# STEP 8: Create key visualizations
# ==============================================================================

print("\nSTEP 8: Creating visualizations...")

# Color scheme
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

# === FIGURE 1: Top 15 correlations for each attachment dimension ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Panel A: Anxiety - Top 15
top_anx = anxiety_df.head(15)
y_pos = np.arange(len(top_anx))
colors_anx = [get_color(p) for p in top_anx['p_uncorrected']]

ax1.barh(y_pos, top_anx['r'], color=colors_anx, edgecolor='white', linewidth=0.8, height=0.7)
ax1.axvline(0, color='#333333', linewidth=1)
ax1.axvline(0.3, color='#cccccc', linewidth=0.8, linestyle='--', alpha=0.6)
ax1.axvline(-0.3, color='#cccccc', linewidth=0.8, linestyle='--', alpha=0.6)

ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{row['Label']} ({row['Category'][:4]})" for _, row in top_anx.iterrows()], fontsize=9)
ax1.set_xlabel('Correlation coefficient (r)', fontsize=11, fontweight='bold')
ax1.set_title('A. Attachment Anxiety × LIWC-22 (Top 15)', fontsize=12, fontweight='bold', loc='left')
ax1.set_xlim(-0.4, 0.4)
ax1.invert_yaxis()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

# Add correlation values
for i, (_, row) in enumerate(top_anx.iterrows()):
    x_pos = row['r'] + 0.02 if row['r'] >= 0 else row['r'] - 0.02
    ha = 'left' if row['r'] >= 0 else 'right'
    ax1.text(x_pos, i, f"r={row['r']:.2f}{row['sig_uncorrected']}", 
             ha=ha, va='center', fontsize=8, fontweight='medium')

# Panel B: Avoidance - Top 15
top_avo = avoidance_df.head(15)
y_pos = np.arange(len(top_avo))
colors_avo = [get_color(p) for p in top_avo['p_uncorrected']]

ax2.barh(y_pos, top_avo['r'], color=colors_avo, edgecolor='white', linewidth=0.8, height=0.7)
ax2.axvline(0, color='#333333', linewidth=1)
ax2.axvline(0.3, color='#cccccc', linewidth=0.8, linestyle='--', alpha=0.6)
ax2.axvline(-0.3, color='#cccccc', linewidth=0.8, linestyle='--', alpha=0.6)

ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"{row['Label']} ({row['Category'][:4]})" for _, row in top_avo.iterrows()], fontsize=9)
ax2.set_xlabel('Correlation coefficient (r)', fontsize=11, fontweight='bold')
ax2.set_title('B. Attachment Avoidance × LIWC-22 (Top 15)', fontsize=12, fontweight='bold', loc='left')
ax2.set_xlim(-0.4, 0.4)
ax2.invert_yaxis()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

# Add correlation values
for i, (_, row) in enumerate(top_avo.iterrows()):
    x_pos = row['r'] + 0.02 if row['r'] >= 0 else row['r'] - 0.02
    ha = 'left' if row['r'] >= 0 else 'right'
    ax2.text(x_pos, i, f"r={row['r']:.2f}{row['sig_uncorrected']}", 
             ha=ha, va='center', fontsize=8, fontweight='medium')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS['sig_001'], label='p < .001'),
    Patch(facecolor=COLORS['sig_01'], label='p < .01'),
    Patch(facecolor=COLORS['sig_05'], label='p < .05'),
    Patch(facecolor=COLORS['ns'], label='n.s.'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
           frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Attachment Dimensions × LIWC-22 Linguistic Features', 
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
fig1_path = f'{OUTPUT_DIR}/figures/FIG1_attachment_liwc22_top15.png'
fig.savefig(fig1_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✓ Saved: {fig1_path}")

# === FIGURE 2: Scatterplots for top significant findings ===
# Combine and find top 4 significant results
all_sig = []
for _, row in anxiety_df[anxiety_df['sig_uncorrected'] != ''].head(4).iterrows():
    all_sig.append({**row.to_dict(), 'attachment': 'Anxiety', 'ecr_var': 'ECR_Anxiety'})
for _, row in avoidance_df[avoidance_df['sig_uncorrected'] != ''].head(4).iterrows():
    all_sig.append({**row.to_dict(), 'attachment': 'Avoidance', 'ecr_var': 'ECR_Avoidance'})

if len(all_sig) >= 4:
    all_sig_df = pd.DataFrame(all_sig)
    top_sig = all_sig_df.nsmallest(4, 'p_uncorrected')
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    
    for idx, (_, result) in enumerate(top_sig.iterrows()):
        ax = axes[idx]
        ecr_var = result['ecr_var']
        liwc_var = result['Variable']
        
        valid = merged_df[[ecr_var, liwc_var]].dropna()
        
        # Scatter plot
        ax.scatter(valid[ecr_var], valid[liwc_var], alpha=0.5, s=40, 
                   color='#1f77b4', edgecolor='white', linewidth=0.5)
        
        # Regression line
        z = np.polyfit(valid[ecr_var], valid[liwc_var], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[ecr_var].min(), valid[ecr_var].max(), 100)
        ax.plot(x_line, p(x_line), color='#c44e52', linewidth=2.5, alpha=0.9)
        
        # Stats box
        stats_text = f"r = {result['r']:.3f} {result['sig_uncorrected']}\nN = {result['N']}\np = {result['p_uncorrected']:.4f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
        
        ax.set_xlabel(f"Attachment {result['attachment']}", fontsize=10, fontweight='bold')
        ax.set_ylabel(f"{result['Label']} (%)", fontsize=10, fontweight='bold')
        ax.set_title(f"{'ABCD'[idx]}. {result['attachment']} × {result['Label']}", 
                     fontsize=11, fontweight='bold', loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)
    
    fig.suptitle('Top Attachment × LIWC-22 Associations', 
                 fontsize=13, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    fig2_path = f'{OUTPUT_DIR}/figures/FIG2_attachment_liwc22_scatterplots.png'
    fig.savefig(fig2_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {fig2_path}")

# ==============================================================================
# STEP 9: Save data
# ==============================================================================

print("\nSTEP 9: Saving data files...")

merged_df.to_csv(f'{OUTPUT_DIR}/merged_ecr_liwc22_data.csv', index=False)
anxiety_df.to_csv(f'{OUTPUT_DIR}/anxiety_liwc22_correlations.csv', index=False)
avoidance_df.to_csv(f'{OUTPUT_DIR}/avoidance_liwc22_correlations.csv', index=False)
controlled_df.to_csv(f'{OUTPUT_DIR}/liwc22_controlled_regression.csv', index=False)

print(f"  ✓ Saved merged data and correlation tables")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "=" * 80)
print("LIWC-22 ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nSample:")
print(f"  • N = {len(merged_df)} participants")
print(f"  • {len(all_liwc_vars)} LIWC-22 variables analyzed")

print(f"\nAttachment Anxiety - Significant correlations (p < .05):")
if n_sig_anx > 0:
    for _, row in anxiety_df[anxiety_df['sig_uncorrected'] != ''].head(5).iterrows():
        print(f"  • {row['Label']}: r = {row['r']:.3f} {row['sig_uncorrected']}")
else:
    print("  • None")

print(f"\nAttachment Avoidance - Significant correlations (p < .05):")
if n_sig_avo > 0:
    for _, row in avoidance_df[avoidance_df['sig_uncorrected'] != ''].head(5).iterrows():
        print(f"  • {row['Label']}: r = {row['r']:.3f} {row['sig_uncorrected']}")
else:
    print("  • None")

print(f"\nAttachment Anxiety - Significant controlled effects (FDR q < .05):")
if n_sig_anx_ctrl_fdr > 0:
    for _, row in anxiety_controlled_df[anxiety_controlled_df['sig_Anxiety'] != ''].head(5).iterrows():
        print(f"  • {row['Label']}: beta = {row['Beta_Anxiety']:.3f} {row['sig_Anxiety']}")
else:
    print("  • None")

print(f"\nAttachment Avoidance - Significant controlled effects (FDR q < .05):")
if n_sig_avo_ctrl_fdr > 0:
    for _, row in avoidance_controlled_df[avoidance_controlled_df['sig_Avoidance'] != ''].head(5).iterrows():
        print(f"  • {row['Label']}: beta = {row['Beta_Avoidance']:.3f} {row['sig_Avoidance']}")
else:
    print("  • None")

print(f"\nOutput files:")
print(f"  • {OUTPUT_DIR}/")
print(f"    ├── TABLE1_anxiety_liwc22_correlations.txt")
print(f"    ├── TABLE2_avoidance_liwc22_correlations.txt")
print(f"    ├── TABLE3_anxiety_liwc22_controlled.txt")
print(f"    ├── TABLE4_avoidance_liwc22_controlled.txt")
print(f"    ├── figures/FIG1_attachment_liwc22_top15.png")
print(f"    ├── figures/FIG2_attachment_liwc22_scatterplots.png")
print(f"    ├── liwc22_controlled_regression.csv")
print(f"    └── merged_ecr_liwc22_data.csv")

print("\n" + "=" * 80)
print("Citation: Boyd, R.L., et al. (2022). LIWC-22. UT Austin.")
print("=" * 80)

