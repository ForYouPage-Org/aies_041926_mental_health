# ECR-SF Quick Reference Guide

**Updated:** October 20, 2025

---

## 🎯 Which File Should I Use?

### For Analysis

**→ Use: `ecr_scores_validated_clean.csv`**

This is your **primary analysis file**:

- ✅ 197 unique participants
- ✅ Duplicates removed (kept first occurrence)
- ✅ Validated scoring methodology
- ✅ Anxiety and Avoidance scores (1-7 scale)

---

## 📊 Quick Stats

| Metric               | Value       |
| -------------------- | ----------- |
| **N**                | 197         |
| **Anxiety M (SD)**   | 4.10 (1.10) |
| **Avoidance M (SD)** | 3.20 (1.00) |
| **Correlation**      | r = 0.40    |

### Attachment Styles

- **Secure:** 27.4%
- **Preoccupied:** 22.3%
- **Dismissive:** 16.8%
- **Fearful:** 33.5%

---

## 📁 File Directory

| File                                  | Purpose                                  | Rows |
| ------------------------------------- | ---------------------------------------- | ---- |
| **`ecr_scores_validated_clean.csv`**  | ⭐ **USE THIS** - Final validated scores | 197  |
| `ecr_scores_detailed.csv`             | Item-level detail for auditing           | 205  |
| `ecr_validation_comparison_clean.csv` | Comparison with prior calculation        | 197  |
| `ECR_VALIDATION_REPORT.md`            | 📄 Full technical documentation          | -    |
| `ECR_VALIDATION_SUMMARY.md`           | 📄 Executive summary                     | -    |
| `ECR_COMPARISON_TABLE.md`             | 📄 Prior vs. NEW comparison              | -    |

---

## 🔍 How to Read the Scores

### Anxiety Dimension (Range: 1-7)

- **Low (1-3):** Comfortable with autonomy, not worried about abandonment
- **Moderate (3-5):** Some concern about partner availability
- **High (5-7):** Strong fear of rejection, need for reassurance

### Avoidance Dimension (Range: 1-7)

- **Low (1-3):** Comfortable with closeness and interdependence
- **Moderate (3-5):** Some discomfort with intimacy
- **High (5-7):** Strong preference for independence, avoid closeness

---

## 📏 Classification Thresholds (Median Split)

Using sample medians:

- **Anxiety threshold:** 4.00
- **Avoidance threshold:** 3.50

| Style           | Anxiety | Avoidance |
| --------------- | ------- | --------- |
| **Secure**      | <4.00   | <3.50     |
| **Preoccupied** | ≥4.00   | <3.50     |
| **Dismissive**  | <4.00   | ≥3.50     |
| **Fearful**     | ≥4.00   | ≥3.50     |

---

## ⚠️ Important Notes

### 1. Sample Characteristics

This sample has:

- **Higher anxiety** than norms (4.10 vs. ~3.50)
- **Lower avoidance** than norms (3.20 vs. ~3.50)
- **Elevated fearful attachment** (33.5% vs. ~15-20%)

### 2. Comparison with Prior Calculation

- **87 participants overlap** with prior calculation
- **54-62% exact agreement** (difference < 0.01)
- **Discrepancies mainly due to** duplicate handling

### 3. Missing Data

- Only **40.9%** of survey participants completed ECR-SF
- Likely reflects relationship status requirement
- No evidence of systematic bias

---

## 🔢 CSV Structure

### ecr_scores_validated_clean.csv

```csv
Prolific_ID,ECR_Anxiety,ECR_Avoidance
10293847,4.5,3.0
42069690,3.667,4.0
...
```

**Columns:**

- `Prolific_ID`: Unique participant identifier (string)
- `ECR_Anxiety`: Anxiety score, 1-7 scale (float)
- `ECR_Avoidance`: Avoidance score, 1-7 scale (float)

---

## 📖 How the Scores Were Calculated

### 1. Item Mapping

- **Anxiety:** Items 2, 4, 6, 8 (reversed), 10, 12
- **Avoidance:** Items 1 (reversed), 3, 5 (reversed), 7, 9 (reversed), 11
- **Attention check:** Excluded from scoring

### 2. Reverse Scoring

Formula: `Reversed Score = 8 - Original Score`

Applied to:

- Item 1 (Avoidance)
- Item 5 (Avoidance)
- Item 8 (Anxiety)
- Item 9 (Avoidance)

### 3. Subscale Calculation

```
Score = Mean of subscale items
Required: ≥3 of 6 items (50% threshold)
```

### 4. Duplicate Handling

- 8 participants had duplicate entries
- Kept **first occurrence** for each Prolific_ID

---

## 💡 Usage Examples

### Python

```python
import pandas as pd

# Load data
df = pd.read_csv('ecr_scores_validated_clean.csv')

# Basic statistics
print(df[['ECR_Anxiety', 'ECR_Avoidance']].describe())

# Create attachment styles
df['Attachment_Style'] = 'Unknown'
df.loc[(df['ECR_Anxiety'] < 4.0) & (df['ECR_Avoidance'] < 3.5), 'Attachment_Style'] = 'Secure'
df.loc[(df['ECR_Anxiety'] >= 4.0) & (df['ECR_Avoidance'] < 3.5), 'Attachment_Style'] = 'Preoccupied'
df.loc[(df['ECR_Anxiety'] < 4.0) & (df['ECR_Avoidance'] >= 3.5), 'Attachment_Style'] = 'Dismissive'
df.loc[(df['ECR_Anxiety'] >= 4.0) & (df['ECR_Avoidance'] >= 3.5), 'Attachment_Style'] = 'Fearful'

# Count by style
print(df['Attachment_Style'].value_counts())
```

### R

```r
# Load data
df <- read.csv('ecr_scores_validated_clean.csv')

# Basic statistics
summary(df[, c('ECR_Anxiety', 'ECR_Avoidance')])

# Create attachment styles
df$Attachment_Style <- ifelse(df$ECR_Anxiety < 4.0 & df$ECR_Avoidance < 3.5, 'Secure',
                       ifelse(df$ECR_Anxiety >= 4.0 & df$ECR_Avoidance < 3.5, 'Preoccupied',
                       ifelse(df$ECR_Anxiety < 4.0 & df$ECR_Avoidance >= 3.5, 'Dismissive',
                       'Fearful')))

# Count by style
table(df$Attachment_Style)
```

---

## ✅ Validation Status

| Check                      | Status  |
| -------------------------- | ------- |
| Attention check excluded   | ✅ Pass |
| Reverse scoring applied    | ✅ Pass |
| Valid score range (1-7)    | ✅ Pass |
| Adequate variance (SD>1.0) | ✅ Pass |
| Expected correlation       | ✅ Pass |
| Duplicates handled         | ✅ Pass |
| Missing data threshold     | ✅ Pass |

---

## 🔗 Related Files

- **Calculation script:** `calculate_validate_ecr_scores.py`
- **Full report:** `ECR_VALIDATION_REPORT.md`
- **Comparison:** `ECR_COMPARISON_TABLE.md`
- **Prior calculation:** `results/251019T1313_ecr_engagement_FULL/ecr_engagement_data_FULL.csv`

---

## 📞 Questions?

For methodology details, see: **ECR_VALIDATION_REPORT.md**  
For comparison with prior calculation, see: **ECR_COMPARISON_TABLE.md**  
For executive summary, see: **ECR_VALIDATION_SUMMARY.md**

---

**Last updated:** October 20, 2025  
**Institution:** University of Washington
