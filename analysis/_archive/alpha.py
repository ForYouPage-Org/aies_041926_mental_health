import pandas as pd

# Load data
df = pd.read_csv('/Users/marxw/Sirius/data/survey_scored.csv')

# Mapping
mapping = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neither Agree nor Disagree': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

# Apply mapping
ai_cols = [f'AI_Q{i}' for i in range(1, 16)]
for col in ai_cols:
    if col in df.columns:
        df[col] = df[col].map(mapping)

# 1. Create Composites (Mean of items)
# usage of min_periods=1 ensures we get a score even if 1 item is missing, 
# but you can remove it if you require all answers.
df['Emotional_Engagement_Comp'] = df[['AI_Q4', 'AI_Q5', 'AI_Q6']].mean(axis=1)
df['Self_Efficacy_Comp']        = df[['AI_Q7', 'AI_Q8', 'AI_Q9']].mean(axis=1)
df['Behavior_Change_Comp']      = df[['AI_Q10', 'AI_Q11', 'AI_Q12']].mean(axis=1)

# 2. Select Single Items
df['Trust_Single']      = df['AI_Q1']   # "I trust the answers..."
df['Dependency_Single'] = df['AI_Q13']  # "I worry I am relying on it too much..."

# View the final 5 variables
final_cols = ['Emotional_Engagement_Comp', 'Self_Efficacy_Comp', 'Behavior_Change_Comp', 
              'Trust_Single', 'Dependency_Single']

print(df[final_cols].head())

# Correlation between these final 5 to see how they relate
print("\nCorrelation of Final 5 Variables:")
print(df[final_cols].corr().round(2))