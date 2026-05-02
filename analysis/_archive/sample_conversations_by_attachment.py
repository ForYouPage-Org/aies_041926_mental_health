#!/usr/bin/env python3
"""
Sample Conversations by Attachment Style
========================================

Samples conversations from individuals with high attachment anxiety and high avoidance.

Usage:
    python sample_conversations_by_attachment.py                    # Default: top quartile
    python sample_conversations_by_attachment.py --threshold 5.0    # Custom threshold
    python sample_conversations_by_attachment.py --n_conversations 50  # Sample 50 per group
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "data"

ECR_PATH = BASE_DIR / "data" / "ecr_scores_validated_clean.csv"
CONVERSATIONS_PATH = DATA_DIR / "conversations_full_20251007.csv"

TIMESTAMP = datetime.now().strftime("%y%m%dT%H%M")
OUTPUT_DIR = BASE_DIR / "results" / f"{TIMESTAMP}_attachment_conversations"

# =============================================================================
# Argument Parsing
# =============================================================================

parser = argparse.ArgumentParser(description='Sample conversations by attachment style')
parser.add_argument('--threshold', type=float, default=None, 
                    help='ECR score threshold for "high" (default: top quartile)')
parser.add_argument('--n_conversations', type=int, default=100,
                    help='Number of conversations to sample per group (default: 100)')
parser.add_argument('--n_messages', type=int, default=None,
                    help='Max messages per conversation (default: all)')
parser.add_argument('--user_only', action='store_true',
                    help='Only include user messages (exclude AI responses)')
parser.add_argument('--max_words', type=int, default=None,
                    help='Filter messages to those with <= max_words (default: no filter)')
parser.add_argument('--min_words', type=int, default=None,
                    help='Filter messages to those with >= min_words (default: no filter)')
args = parser.parse_args()

# =============================================================================
# Load Data
# =============================================================================

print("=" * 70)
print("SAMPLE CONVERSATIONS BY ATTACHMENT STYLE")
print("=" * 70)

print("\n--- Loading Data ---")

# ECR Scores
ecr_df = pd.read_csv(ECR_PATH)
ecr_df['Prolific_ID'] = ecr_df['Prolific_ID'].astype(str).str.strip()
print(f"ECR scores: {len(ecr_df)} participants")

# Score distribution
print(f"\nScore Distributions:")
print(f"  Anxiety:   M={ecr_df['ECR_Anxiety'].mean():.2f}, SD={ecr_df['ECR_Anxiety'].std():.2f}, "
      f"Range=[{ecr_df['ECR_Anxiety'].min():.2f}, {ecr_df['ECR_Anxiety'].max():.2f}]")
print(f"  Avoidance: M={ecr_df['ECR_Avoidance'].mean():.2f}, SD={ecr_df['ECR_Avoidance'].std():.2f}, "
      f"Range=[{ecr_df['ECR_Avoidance'].min():.2f}, {ecr_df['ECR_Avoidance'].max():.2f}]")

# Determine thresholds
if args.threshold is not None:
    anxiety_threshold = args.threshold
    avoidance_threshold = args.threshold
    print(f"\nUsing custom threshold: {args.threshold}")
else:
    # Use top quartile (75th percentile)
    anxiety_threshold = ecr_df['ECR_Anxiety'].quantile(0.75)
    avoidance_threshold = ecr_df['ECR_Avoidance'].quantile(0.75)
    print(f"\nUsing top quartile thresholds:")
    print(f"  High Anxiety: > {anxiety_threshold:.2f} (75th percentile)")
    print(f"  High Avoidance: > {avoidance_threshold:.2f} (75th percentile)")

# Identify high attachment individuals
high_anxiety_ids = set(ecr_df[ecr_df['ECR_Anxiety'] > anxiety_threshold]['Prolific_ID'])
high_avoidance_ids = set(ecr_df[ecr_df['ECR_Avoidance'] > avoidance_threshold]['Prolific_ID'])

print(f"\nHigh Anxiety participants (> {anxiety_threshold:.2f}): {len(high_anxiety_ids)}")
print(f"High Avoidance participants (> {avoidance_threshold:.2f}): {len(high_avoidance_ids)}")
print(f"Overlap (both high): {len(high_anxiety_ids & high_avoidance_ids)}")

# =============================================================================
# Load Conversations
# =============================================================================

print("\n--- Loading Conversations ---")
print("(This may take a moment for large files...)")

# Load in chunks for memory efficiency
chunk_size = 100000
conversations_chunks = []

# First pass: identify which participant IDs are in the conversation data
target_ids = high_anxiety_ids | high_avoidance_ids

for chunk in pd.read_csv(CONVERSATIONS_PATH, chunksize=chunk_size, low_memory=False):
    chunk['participant_id'] = chunk['participant_id'].astype(str).str.strip()
    # Filter to only target participants
    filtered = chunk[chunk['participant_id'].isin(target_ids)]
    if len(filtered) > 0:
        conversations_chunks.append(filtered)

if not conversations_chunks:
    print("ERROR: No conversations found for high attachment participants!")
    exit(1)

conversations_df = pd.concat(conversations_chunks, ignore_index=True)
print(f"Loaded {len(conversations_df)} messages from {conversations_df['participant_id'].nunique()} participants")

# Filter user messages only (always apply - we want user messages)
conversations_df = conversations_df[conversations_df['user_or_ai'] == 'user'].copy()
print(f"Filtered to user messages only: {len(conversations_df)} messages")

# Compute word count if not present
if 'word_count' not in conversations_df.columns:
    conversations_df['word_count'] = conversations_df['message'].fillna('').astype(str).str.split().str.len()

# Apply word count filters
if args.max_words is not None:
    before = len(conversations_df)
    conversations_df = conversations_df[conversations_df['word_count'] <= args.max_words]
    print(f"Filtered to messages <= {args.max_words} words: {len(conversations_df)} messages (removed {before - len(conversations_df)})")

if args.min_words is not None:
    before = len(conversations_df)
    conversations_df = conversations_df[conversations_df['word_count'] >= args.min_words]
    print(f"Filtered to messages >= {args.min_words} words: {len(conversations_df)} messages (removed {before - len(conversations_df)})")

# =============================================================================
# Sample Conversations
# =============================================================================

print("\n--- Sampling Conversations ---")

def sample_conversations(df, participant_ids, n_target, group_name):
    """Sample n conversations from participants with specified IDs."""
    # Filter to target participants
    group_df = df[df['participant_id'].isin(participant_ids)].copy()
    
    if len(group_df) == 0:
        print(f"  Warning: No conversations found for {group_name} group")
        return pd.DataFrame()
    
    # Get unique conversations
    unique_convs = group_df['conversation_id'].unique()
    n_available = len(unique_convs)
    
    print(f"  {group_name}: {n_available} conversations available from {group_df['participant_id'].nunique()} participants")
    
    # Sample
    n_sample = min(n_target, n_available)
    sampled_convs = np.random.choice(unique_convs, size=n_sample, replace=False)
    
    # Get all messages from sampled conversations
    sampled_df = group_df[group_df['conversation_id'].isin(sampled_convs)].copy()
    sampled_df['attachment_group'] = group_name
    
    print(f"  Sampled: {n_sample} conversations, {len(sampled_df)} messages")
    
    return sampled_df

np.random.seed(42)  # Reproducibility

# Sample from high anxiety (excluding overlap to avoid double counting)
anxiety_only_ids = high_anxiety_ids - high_avoidance_ids
anxiety_sample = sample_conversations(
    conversations_df, 
    high_anxiety_ids,  # Include all high anxiety
    args.n_conversations, 
    "High_Anxiety"
)

# Sample from high avoidance (excluding overlap)
avoidance_only_ids = high_avoidance_ids - high_anxiety_ids
avoidance_sample = sample_conversations(
    conversations_df,
    high_avoidance_ids,  # Include all high avoidance
    args.n_conversations,
    "High_Avoidance"
)

# Merge ECR scores for analysis
anxiety_sample = anxiety_sample.merge(
    ecr_df[['Prolific_ID', 'ECR_Anxiety', 'ECR_Avoidance']], 
    left_on='participant_id', right_on='Prolific_ID', how='left'
)
avoidance_sample = avoidance_sample.merge(
    ecr_df[['Prolific_ID', 'ECR_Anxiety', 'ECR_Avoidance']], 
    left_on='participant_id', right_on='Prolific_ID', how='left'
)

# =============================================================================
# Save Results
# =============================================================================

print("\n--- Saving Results ---")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save separate files
anxiety_sample.to_csv(OUTPUT_DIR / "high_anxiety_conversations.csv", index=False)
avoidance_sample.to_csv(OUTPUT_DIR / "high_avoidance_conversations.csv", index=False)

# Save combined
combined = pd.concat([anxiety_sample, avoidance_sample], ignore_index=True)
combined.to_csv(OUTPUT_DIR / "all_sampled_conversations.csv", index=False)

# Create summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'anxiety_threshold': anxiety_threshold,
    'avoidance_threshold': avoidance_threshold,
    'n_high_anxiety_participants': len(high_anxiety_ids),
    'n_high_avoidance_participants': len(high_avoidance_ids),
    'anxiety_conversations_sampled': anxiety_sample['conversation_id'].nunique(),
    'anxiety_messages_total': len(anxiety_sample),
    'avoidance_conversations_sampled': avoidance_sample['conversation_id'].nunique(),
    'avoidance_messages_total': len(avoidance_sample),
}

# Save summary
import json
with open(OUTPUT_DIR / "sampling_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

# Create participant-level summary
anxiety_participants = anxiety_sample.groupby('participant_id').agg({
    'conversation_id': 'nunique',
    'message': 'count',
    'ECR_Anxiety': 'first',
    'ECR_Avoidance': 'first'
}).reset_index()
anxiety_participants.columns = ['participant_id', 'n_conversations', 'n_messages', 'ECR_Anxiety', 'ECR_Avoidance']
anxiety_participants['group'] = 'High_Anxiety'

avoidance_participants = avoidance_sample.groupby('participant_id').agg({
    'conversation_id': 'nunique',
    'message': 'count',
    'ECR_Anxiety': 'first',
    'ECR_Avoidance': 'first'
}).reset_index()
avoidance_participants.columns = ['participant_id', 'n_conversations', 'n_messages', 'ECR_Anxiety', 'ECR_Avoidance']
avoidance_participants['group'] = 'High_Avoidance'

participant_summary = pd.concat([anxiety_participants, avoidance_participants])
participant_summary.to_csv(OUTPUT_DIR / "participant_summary.csv", index=False)

# =============================================================================
# Print Summary
# =============================================================================

print(f"\n{'='*70}")
print("SAMPLING COMPLETE")
print(f"{'='*70}")
print(f"\nThresholds Used:")
print(f"  Anxiety > {anxiety_threshold:.2f}")
print(f"  Avoidance > {avoidance_threshold:.2f}")

print(f"\nHigh Anxiety Sample:")
print(f"  Conversations: {anxiety_sample['conversation_id'].nunique()}")
print(f"  Messages: {len(anxiety_sample)}")
print(f"  Participants: {anxiety_sample['participant_id'].nunique()}")
if 'ECR_Anxiety' in anxiety_sample.columns:
    print(f"  Mean Anxiety Score: {anxiety_sample['ECR_Anxiety'].mean():.2f}")

print(f"\nHigh Avoidance Sample:")
print(f"  Conversations: {avoidance_sample['conversation_id'].nunique()}")
print(f"  Messages: {len(avoidance_sample)}")
print(f"  Participants: {avoidance_sample['participant_id'].nunique()}")
if 'ECR_Avoidance' in avoidance_sample.columns:
    print(f"  Mean Avoidance Score: {avoidance_sample['ECR_Avoidance'].mean():.2f}")

print(f"\nOutput saved to: {OUTPUT_DIR}")
print(f"  - high_anxiety_conversations.csv")
print(f"  - high_avoidance_conversations.csv")
print(f"  - all_sampled_conversations.csv")
print(f"  - participant_summary.csv")
print(f"  - sampling_summary.json")

