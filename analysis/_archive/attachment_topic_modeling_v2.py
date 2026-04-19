#!/usr/bin/env python3
"""
Topic Modeling by Attachment Style v2 - With Academic/Homework Filtering
=========================================================================

Revision: Separates homework/academic use from personal/emotional use
to better understand attachment-related patterns.

Date: December 29, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import requests
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

FILTERED_PROMPTS_PATH = '/Users/marxw/Sirius/FaccT_122925/results/251229T1212_NRC_VAD_analysis/filtered_user_prompts_lt20words.csv'
ECR_DATA_PATH = '/Users/marxw/Sirius/analysis_102025/ecr_scores_validated_clean.csv'

# Output directory
timestamp = datetime.now().strftime('%y%m%dT%H%M')
OUTPUT_DIR = f'/Users/marxw/Sirius/FaccT_122925/results/{timestamp}_topic_modeling_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

# Sampling configuration
MAX_MESSAGES_PER_GROUP = 8000
MIN_CLUSTER_SIZE = 15

print("=" * 80)
print("TOPIC MODELING v2 - WITH ACADEMIC/HOMEWORK FILTERING")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print()

# ==============================================================================
# Academic/Homework Detection Patterns
# ==============================================================================

# Keywords strongly indicating academic/homework use
ACADEMIC_KEYWORDS = [
    # Direct academic terms
    r'\b(homework|assignment|essay|thesis|dissertation|coursework)\b',
    r'\b(exam|quiz|test|midterm|final)\b',
    r'\b(professor|teacher|instructor|class|lecture|course)\b',
    r'\b(grade|grading|rubric|deadline|due date)\b',
    r'\b(apa|mla|chicago|citation|cite|reference)\b',
    r'\b(paragraph|introduction|conclusion|body paragraph)\b',
    
    # Writing assistance patterns
    r'\b(rewrite|rephrase|paraphrase|summarize|shorten)\b',
    r'\b(make it shorter|make it longer|word count)\b',
    r'\b(proofread|grammar|spelling|punctuation)\b',
    r'\b(formal|informal|academic writing|tone)\b',
    
    # Code/programming homework
    r'\b(code|function|algorithm|debug|error|syntax)\b',
    r'\b(python|java|javascript|c\+\+|sql|html|css)\b',
    r'\b(compile|runtime|output|variable|loop)\b',
    
    # Math/Science homework
    r'\b(solve|calculate|equation|formula|derivative|integral)\b',
    r'\b(proof|theorem|hypothesis|experiment)\b',
    r'\b(convert|units|measurement)\b',
    
    # Translation/Language learning
    r'\b(translate|translation|spanish|french|german|chinese|japanese)\b',
    r'\b(vocabulary|grammar|conjugate|tense)\b',
    
    # Generic task requests
    r'\b(explain this|what is|define|describe|compare)\b',
    r'\b(list|outline|bullet points|steps)\b',
    r'\b(example|examples|sample)\b',
]

# Keywords indicating personal/emotional use
PERSONAL_KEYWORDS = [
    # Emotional terms
    r'\b(feel|feeling|felt|feelings)\b',
    r'\b(sad|happy|angry|anxious|depressed|lonely|scared)\b',
    r'\b(love|hate|miss|hurt|pain|cry|crying)\b',
    r'\b(stress|stressed|overwhelmed|exhausted)\b',
    
    # Relationship terms
    r'\b(boyfriend|girlfriend|husband|wife|partner|ex)\b',
    r'\b(friend|friendship|relationship|dating)\b',
    r'\b(family|mom|dad|mother|father|brother|sister)\b',
    r'\b(breakup|divorce|argument|fight)\b',
    
    # Personal advice
    r'\b(advice|help me|what should i|how do i deal)\b',
    r'\b(situation|problem|issue|struggle)\b',
    r'\b(life|future|career|decision)\b',
    
    # Roleplay/fantasy
    r'\b(roleplay|rp|character|story|scenario)\b',
    r'\b(pretend|imagine|act as|you are)\b',
    r'\b(master|servant|dom|sub)\b',
    
    # Mental health
    r'\b(therapy|therapist|counseling|mental health)\b',
    r'\b(trauma|abuse|addiction|recovery)\b',
    r'\b(suicide|self-harm|crisis)\b',
    
    # Self-reflection
    r'\b(myself|my life|my feelings|i think|i feel)\b',
    r'\b(understand me|listen|talk to|vent)\b',
]

def classify_message(text):
    """
    Classify message as 'academic', 'personal', or 'ambiguous'.
    Returns tuple: (category, academic_score, personal_score)
    """
    if pd.isna(text):
        return 'ambiguous', 0, 0
    
    text_lower = str(text).lower()
    
    academic_score = 0
    personal_score = 0
    
    # Count academic keyword matches
    for pattern in ACADEMIC_KEYWORDS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            academic_score += 1
    
    # Count personal keyword matches
    for pattern in PERSONAL_KEYWORDS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            personal_score += 1
    
    # Classification logic
    if academic_score >= 2 and academic_score > personal_score * 2:
        return 'academic', academic_score, personal_score
    elif personal_score >= 2 and personal_score > academic_score:
        return 'personal', academic_score, personal_score
    elif academic_score >= 1 and personal_score == 0:
        return 'academic', academic_score, personal_score
    elif personal_score >= 1 and academic_score == 0:
        return 'personal', academic_score, personal_score
    else:
        return 'ambiguous', academic_score, personal_score


# ==============================================================================
# Load and Prepare Data
# ==============================================================================

print("STEP 1: Loading data...")

prompts_df = pd.read_csv(FILTERED_PROMPTS_PATH)
print(f"  Loaded {len(prompts_df):,} filtered user prompts")

ecr_df = pd.read_csv(ECR_DATA_PATH)
print(f"  Loaded {len(ecr_df)} ECR records")

# Merge
prompts_df['participant_id'] = prompts_df['participant_id'].astype(str)
ecr_df['Prolific_ID'] = ecr_df['Prolific_ID'].astype(str)

merged_df = prompts_df.merge(
    ecr_df[['Prolific_ID', 'ECR_Anxiety', 'ECR_Avoidance']],
    left_on='participant_id',
    right_on='Prolific_ID',
    how='inner'
)

merged_df = merged_df[merged_df['message'].notna() & (merged_df['message'].str.len() > 10)]
print(f"  Valid messages: {len(merged_df):,}")

# ==============================================================================
# Classify All Messages
# ==============================================================================

print("\nSTEP 2: Classifying messages (academic vs personal vs ambiguous)...")

# Apply classification
classifications = merged_df['message'].apply(classify_message)
merged_df['category'] = [c[0] for c in classifications]
merged_df['academic_score'] = [c[1] for c in classifications]
merged_df['personal_score'] = [c[2] for c in classifications]

# Overall statistics
total = len(merged_df)
academic_count = (merged_df['category'] == 'academic').sum()
personal_count = (merged_df['category'] == 'personal').sum()
ambiguous_count = (merged_df['category'] == 'ambiguous').sum()

print(f"\n  OVERALL MESSAGE CLASSIFICATION:")
print(f"  ────────────────────────────────")
print(f"  Academic/Homework:  {academic_count:>6,} ({academic_count/total*100:>5.1f}%)")
print(f"  Personal/Emotional: {personal_count:>6,} ({personal_count/total*100:>5.1f}%)")
print(f"  Ambiguous/Other:    {ambiguous_count:>6,} ({ambiguous_count/total*100:>5.1f}%)")

# ==============================================================================
# Create Attachment Groups
# ==============================================================================

print("\nSTEP 3: Creating attachment style groups...")

anxiety_median = merged_df['ECR_Anxiety'].median()
avoidance_median = merged_df['ECR_Avoidance'].median()

groups = {
    'high_anxiety': merged_df[merged_df['ECR_Anxiety'] >= anxiety_median],
    'low_anxiety': merged_df[merged_df['ECR_Anxiety'] < anxiety_median],
    'high_avoidance': merged_df[merged_df['ECR_Avoidance'] >= avoidance_median],
    'low_avoidance': merged_df[merged_df['ECR_Avoidance'] < avoidance_median],
}

# Classification breakdown by group
print(f"\n  CLASSIFICATION BY ATTACHMENT GROUP:")
print(f"  {'Group':<20} {'Academic':>12} {'Personal':>12} {'Ambiguous':>12}")
print(f"  " + "─" * 58)

group_stats = {}
for group_name, group_df in groups.items():
    n_total = len(group_df)
    n_academic = (group_df['category'] == 'academic').sum()
    n_personal = (group_df['category'] == 'personal').sum()
    n_ambiguous = (group_df['category'] == 'ambiguous').sum()
    
    group_stats[group_name] = {
        'total': n_total,
        'academic': n_academic,
        'academic_pct': n_academic/n_total*100,
        'personal': n_personal,
        'personal_pct': n_personal/n_total*100,
        'ambiguous': n_ambiguous,
        'ambiguous_pct': n_ambiguous/n_total*100,
    }
    
    print(f"  {group_name:<20} {n_academic/n_total*100:>11.1f}% {n_personal/n_total*100:>11.1f}% {n_ambiguous/n_total*100:>11.1f}%")

# ==============================================================================
# Load Embedding Model
# ==============================================================================

print("\nSTEP 4: Loading sentence embedding model...")

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-mpnet-base-v2')
print("  ✓ Loaded all-mpnet-base-v2")

# ==============================================================================
# Topic Modeling Functions
# ==============================================================================

def call_ollama(prompt, model=OLLAMA_MODEL, max_tokens=30):
    """Call local Ollama API for topic labeling."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.3}
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get('response', '').strip()
        return ""
    except:
        return ""


def extract_tfidf_keywords(texts, n_keywords=8):
    """Extract top TF-IDF keywords."""
    if len(texts) < 3:
        return []
    try:
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[-n_keywords:][::-1]
        return [(feature_names[i], mean_scores[i]) for i in top_indices]
    except:
        return []


def get_representative_samples(texts, embeddings, centroid, n_samples=5):
    """Get samples closest to cluster centroid."""
    distances = cosine_similarity([centroid], embeddings)[0]
    top_indices = distances.argsort()[-n_samples:][::-1]
    return [texts[i] for i in top_indices]


def run_topic_modeling(messages_df, group_name, category_filter=None, max_messages=5000):
    """Run topic modeling on a subset of messages."""
    
    if category_filter:
        filtered_df = messages_df[messages_df['category'] == category_filter]
        filter_label = f" [{category_filter} only]"
    else:
        filtered_df = messages_df
        filter_label = ""
    
    if len(filtered_df) < 50:
        print(f"    Skipping {group_name}{filter_label}: only {len(filtered_df)} messages")
        return None
    
    print(f"\n  Processing {group_name}{filter_label}...")
    
    # Sample if too large
    if len(filtered_df) > max_messages:
        filtered_df = filtered_df.sample(max_messages, random_state=42)
        print(f"    Sampled to {max_messages:,} messages")
    
    texts = filtered_df['message'].tolist()
    
    # Embed
    print(f"    Embedding {len(texts):,} messages...")
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)
    
    # Cluster
    print("    Clustering with HDBSCAN...")
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    
    unique_clusters = [c for c in set(cluster_labels) if c != -1]
    print(f"    Found {len(unique_clusters)} clusters")
    
    cluster_results = []
    for cluster_id in sorted(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
        cluster_embeddings = embeddings[cluster_mask]
        
        if len(cluster_texts) < 10:
            continue
        
        centroid = cluster_embeddings.mean(axis=0)
        keywords = extract_tfidf_keywords(cluster_texts, n_keywords=8)
        keyword_str = ", ".join([kw for kw, _ in keywords[:8]])
        rep_samples = get_representative_samples(cluster_texts, cluster_embeddings, centroid, n_samples=5)
        
        # LLM labeling
        llm_prompt = f"""Based on these keywords and examples from ChatGPT user messages, provide a SHORT topic label (3-5 words).

Keywords: {keyword_str}

Examples:
1. "{rep_samples[0][:120]}"
2. "{rep_samples[1][:120]}"

Return ONLY the label. Example: "Relationship Advice Seeking" or "Creative Writing Help"
"""
        topic_label = call_ollama(llm_prompt)
        topic_label = topic_label.replace('"', '').replace('\n', ' ').strip()[:50]
        
        if not topic_label or len(topic_label) < 3:
            top_kws = [kw for kw, _ in keywords[:3]]
            topic_label = f"[{', '.join(top_kws)}]"
        
        cluster_results.append({
            'cluster_id': cluster_id,
            'n_messages': len(cluster_texts),
            'pct_of_total': len(cluster_texts) / len(texts) * 100,
            'keywords': keyword_str,
            'topic_label': topic_label,
            'representative_samples': rep_samples[:3]
        })
        
        print(f"      • {topic_label}: {len(cluster_texts)} msgs ({len(cluster_texts)/len(texts)*100:.1f}%)")
    
    cluster_results.sort(key=lambda x: x['n_messages'], reverse=True)
    
    return {
        'group_name': group_name,
        'category_filter': category_filter,
        'total_messages': len(texts),
        'n_clusters': len(cluster_results),
        'noise_pct': (cluster_labels == -1).sum() / len(texts) * 100,
        'clusters': cluster_results
    }


# ==============================================================================
# Run Topic Modeling - Focus on Personal/Emotional Messages
# ==============================================================================

print("\nSTEP 5: Topic modeling on PERSONAL/EMOTIONAL messages only...")

personal_results = {}
for group_name, group_df in groups.items():
    result = run_topic_modeling(group_df, group_name, category_filter='personal', max_messages=3000)
    if result:
        personal_results[group_name] = result

# ==============================================================================
# Run Topic Modeling - Ambiguous messages (may contain interesting patterns)
# ==============================================================================

print("\n\nSTEP 6: Topic modeling on AMBIGUOUS messages...")

ambiguous_results = {}
for group_name, group_df in groups.items():
    result = run_topic_modeling(group_df, group_name, category_filter='ambiguous', max_messages=3000)
    if result:
        ambiguous_results[group_name] = result

# ==============================================================================
# Save Results
# ==============================================================================

print("\n\nSTEP 7: Saving results...")

# Save comprehensive JSON
all_results = {
    'classification_summary': {
        'total_messages': total,
        'academic_count': int(academic_count),
        'academic_pct': float(academic_count/total*100),
        'personal_count': int(personal_count),
        'personal_pct': float(personal_count/total*100),
        'ambiguous_count': int(ambiguous_count),
        'ambiguous_pct': float(ambiguous_count/total*100),
    },
    'group_stats': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                        for kk, vv in v.items()} for k, v in group_stats.items()},
    'personal_topics': personal_results,
    'ambiguous_topics': ambiguous_results,
}

with open(f'{OUTPUT_DIR}/topic_modeling_v2_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# Create formatted report
report_path = f'{OUTPUT_DIR}/TOPIC_MODELING_V2_REPORT.txt'
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("TOPIC MODELING v2 - ACADEMIC vs PERSONAL ANALYSIS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Method: Keyword classification + MPNet + HDBSCAN + Ollama\n")
    f.write("-" * 80 + "\n\n")
    
    # Overall classification
    f.write("OVERALL MESSAGE CLASSIFICATION\n")
    f.write("=" * 40 + "\n")
    f.write(f"Total messages analyzed: {total:,}\n\n")
    f.write(f"  Academic/Homework:  {academic_count:>6,} ({academic_count/total*100:>5.1f}%)\n")
    f.write(f"  Personal/Emotional: {personal_count:>6,} ({personal_count/total*100:>5.1f}%)\n")
    f.write(f"  Ambiguous/Other:    {ambiguous_count:>6,} ({ambiguous_count/total*100:>5.1f}%)\n\n")
    
    # By attachment group
    f.write("\nCLASSIFICATION BY ATTACHMENT GROUP\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"{'Group':<20} {'Academic':>10} {'Personal':>10} {'Ambiguous':>10}\n")
    f.write("-" * 52 + "\n")
    for group_name, stats in group_stats.items():
        f.write(f"{group_name:<20} {stats['academic_pct']:>9.1f}% {stats['personal_pct']:>9.1f}% {stats['ambiguous_pct']:>9.1f}%\n")
    
    # Personal topics
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("TOPICS IN PERSONAL/EMOTIONAL MESSAGES\n")
    f.write("=" * 80 + "\n")
    
    for group_name, result in personal_results.items():
        f.write(f"\n{'─'*40}\n")
        f.write(f"{group_name.upper().replace('_', ' ')}\n")
        f.write(f"{'─'*40}\n")
        f.write(f"Messages: {result['total_messages']:,} | Clusters: {result['n_clusters']} | Noise: {result['noise_pct']:.1f}%\n\n")
        
        for i, cluster in enumerate(result['clusters'][:8], 1):
            f.write(f"{i}. {cluster['topic_label']}\n")
            f.write(f"   Count: {cluster['n_messages']} ({cluster['pct_of_total']:.1f}%)\n")
            f.write(f"   Keywords: {cluster['keywords']}\n")
            f.write(f"   Example: \"{cluster['representative_samples'][0][:80]}...\"\n\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")

print(f"  ✓ Saved: {report_path}")

# ==============================================================================
# Create Visualization
# ==============================================================================

print("\nSTEP 8: Creating visualizations...")

import matplotlib.pyplot as plt

# Figure 1: Classification pie charts by group
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

colors = ['#4878d0', '#ee854a', '#6acc64']  # Academic, Personal, Ambiguous

for idx, (group_name, stats) in enumerate(group_stats.items()):
    ax = axes[idx]
    sizes = [stats['academic_pct'], stats['personal_pct'], stats['ambiguous_pct']]
    labels = [f"Academic\n{stats['academic_pct']:.1f}%", 
              f"Personal\n{stats['personal_pct']:.1f}%", 
              f"Ambiguous\n{stats['ambiguous_pct']:.1f}%"]
    
    wedges, texts = ax.pie(sizes, colors=colors, startangle=90)
    ax.set_title(group_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    
    # Add legend for first subplot only
    if idx == 0:
        ax.legend(wedges, ['Academic/Homework', 'Personal/Emotional', 'Ambiguous'],
                  loc='lower left', fontsize=9)

fig.suptitle('Message Classification by Attachment Style', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/FIG1_classification_by_group.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/FIG1_classification_by_group.png")

# Figure 2: Personal topics comparison
if personal_results:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    
    for idx, group_name in enumerate(['high_anxiety', 'low_anxiety', 'high_avoidance', 'low_avoidance']):
        ax = axes[idx]
        
        if group_name in personal_results:
            result = personal_results[group_name]
            clusters = result['clusters'][:6]
            
            if clusters:
                labels = [c['topic_label'][:35] for c in clusters]
                sizes = [c['pct_of_total'] for c in clusters]
                
                colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(clusters)))
                bars = ax.barh(range(len(labels)), sizes, color=colors_bar)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_xlabel('% of Personal Messages', fontsize=10)
                ax.invert_yaxis()
                
                for bar, size in zip(bars, sizes):
                    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                           f'{size:.1f}%', va='center', fontsize=8)
        
        ax.set_title(group_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlim(0, max([c['pct_of_total'] for r in personal_results.values() for c in r['clusters'][:6]]) * 1.3 if personal_results else 10)
    
    fig.suptitle('Topics in Personal/Emotional Messages by Attachment Style', 
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/FIG2_personal_topics_by_group.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/FIG2_personal_topics_by_group.png")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"\n📊 KEY FINDING: {academic_count/total*100:.1f}% of messages are Academic/Homework")
print(f"   Only {personal_count/total*100:.1f}% are Personal/Emotional content\n")

print("Personal message topics by attachment style:")
for group_name, result in personal_results.items():
    print(f"\n  {group_name.upper().replace('_', ' ')}:")
    for cluster in result['clusters'][:3]:
        print(f"    • {cluster['topic_label']} ({cluster['pct_of_total']:.1f}%)")

print(f"\nOutput files:")
print(f"  • {OUTPUT_DIR}/")
print(f"    ├── TOPIC_MODELING_V2_REPORT.txt")
print(f"    ├── topic_modeling_v2_results.json")
print(f"    ├── FIG1_classification_by_group.png")
print(f"    └── FIG2_personal_topics_by_group.png")

print("\n" + "=" * 80)

