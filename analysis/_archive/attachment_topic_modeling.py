#!/usr/bin/env python3
"""
Topic Modeling by Attachment Style using MPNet + Clustering + LLM Labeling
==========================================================================

Method (adapted from research methodology):
1. Embed texts using MPNet (sentence-transformers)
2. Cluster embeddings with HDBSCAN (nonparametric, infers cluster count)
3. Extract representative posts and TF-IDF keywords for each cluster
4. Use local Ollama (qwen3:8b) to generate topic labels
5. Compare topics across attachment style groups

Date: December 29, 2025
Institution: University of Washington
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
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

FILTERED_PROMPTS_PATH = '/Users/marxw/Sirius/FaccT_122925/results/251229T1212_NRC_VAD_analysis/filtered_user_prompts_lt20words.csv'
ECR_DATA_PATH = '/Users/marxw/Sirius/analysis_102025/ecr_scores_validated_clean.csv'

# Output directory
timestamp = datetime.now().strftime('%y%m%dT%H%M')
OUTPUT_DIR = f'/Users/marxw/Sirius/FaccT_122925/results/{timestamp}_topic_modeling'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"  # Changed from qwen3:8b which returns empty

# Sampling configuration (for speed)
MAX_MESSAGES_PER_GROUP = 5000  # Sample this many messages per attachment group
MIN_CLUSTER_SIZE = 20  # Minimum cluster size for HDBSCAN (lowered for more granularity)
SAMPLE_SIZE_FOR_EMBEDDING = 10000  # Max messages to embed per group

print("=" * 80)
print("TOPIC MODELING BY ATTACHMENT STYLE")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Using: MPNet embeddings + HDBSCAN clustering + Ollama ({OLLAMA_MODEL})")
print()

# ==============================================================================
# STEP 1: Load and prepare data
# ==============================================================================

print("STEP 1: Loading data...")

# Load filtered prompts
prompts_df = pd.read_csv(FILTERED_PROMPTS_PATH)
print(f"  Loaded {len(prompts_df):,} filtered user prompts")

# Load ECR data
ecr_df = pd.read_csv(ECR_DATA_PATH)
print(f"  Loaded {len(ecr_df)} ECR records")

# Merge to get attachment scores for each message
prompts_df['participant_id'] = prompts_df['participant_id'].astype(str)
ecr_df['Prolific_ID'] = ecr_df['Prolific_ID'].astype(str)

merged_df = prompts_df.merge(
    ecr_df[['Prolific_ID', 'ECR_Anxiety', 'ECR_Avoidance']],
    left_on='participant_id',
    right_on='Prolific_ID',
    how='inner'
)

print(f"  Merged: {len(merged_df):,} messages with attachment data")

# Filter to valid messages
merged_df = merged_df[merged_df['message'].notna() & (merged_df['message'].str.len() > 10)]
print(f"  After filtering short/empty: {len(merged_df):,} messages")

# ==============================================================================
# STEP 2: Create attachment groups
# ==============================================================================

print("\nSTEP 2: Creating attachment style groups...")

# Use median split for groups
anxiety_median = merged_df['ECR_Anxiety'].median()
avoidance_median = merged_df['ECR_Avoidance'].median()

print(f"  Anxiety median: {anxiety_median:.2f}")
print(f"  Avoidance median: {avoidance_median:.2f}")

# Create groups
groups = {
    'high_anxiety': merged_df[merged_df['ECR_Anxiety'] >= anxiety_median],
    'low_anxiety': merged_df[merged_df['ECR_Anxiety'] < anxiety_median],
    'high_avoidance': merged_df[merged_df['ECR_Avoidance'] >= avoidance_median],
    'low_avoidance': merged_df[merged_df['ECR_Avoidance'] < avoidance_median],
}

for name, df in groups.items():
    print(f"  {name}: {len(df):,} messages from {df['participant_id'].nunique()} participants")

# ==============================================================================
# STEP 3: Load embedding model
# ==============================================================================

print("\nSTEP 3: Loading sentence embedding model...")

try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-mpnet-base-v2')
    print("  ✓ Loaded all-mpnet-base-v2")
except ImportError:
    print("  ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
    exit(1)

# ==============================================================================
# STEP 4: Define topic modeling pipeline
# ==============================================================================

def call_ollama(prompt, model=OLLAMA_MODEL, max_tokens=200):
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
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def extract_tfidf_keywords(texts, n_keywords=10):
    """Extract top TF-IDF keywords from a set of texts."""
    if len(texts) < 3:
        return []
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get mean TF-IDF scores
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


def run_topic_modeling(messages, group_name, max_messages=5000):
    """
    Run topic modeling pipeline for a group of messages.
    
    Returns dict with cluster info, keywords, and LLM-generated labels.
    """
    print(f"\n  Processing {group_name}...")
    
    # Sample if too large
    if len(messages) > max_messages:
        messages = messages.sample(max_messages, random_state=42)
        print(f"    Sampled to {max_messages:,} messages")
    
    texts = messages['message'].tolist()
    
    # Embed messages
    print(f"    Embedding {len(texts):,} messages...")
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)
    
    # Cluster with HDBSCAN
    print("    Clustering with HDBSCAN...")
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=10,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(embeddings)
    except ImportError:
        # Fallback to KMeans if HDBSCAN not available
        print("    (HDBSCAN not available, using KMeans with k=10)")
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=10, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(embeddings)
    
    # Get cluster statistics
    unique_clusters = [c for c in set(cluster_labels) if c != -1]
    print(f"    Found {len(unique_clusters)} clusters (excluding noise)")
    
    # Analyze each cluster
    cluster_results = []
    
    for cluster_id in sorted(unique_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
        cluster_embeddings = embeddings[cluster_mask]
        
        if len(cluster_texts) < 10:
            continue
        
        # Get centroid
        centroid = cluster_embeddings.mean(axis=0)
        
        # Get TF-IDF keywords
        keywords = extract_tfidf_keywords(cluster_texts, n_keywords=8)
        keyword_str = ", ".join([kw for kw, _ in keywords[:8]])
        
        # Get representative samples
        rep_samples = get_representative_samples(cluster_texts, cluster_embeddings, centroid, n_samples=5)
        
        # Generate LLM topic label
        llm_prompt = f"""You are analyzing user messages sent to ChatGPT. Based on these keywords and examples, provide a SHORT topic label (3-5 words).

Keywords: {keyword_str}

Example messages:
1. "{rep_samples[0][:150]}"
2. "{rep_samples[1][:150]}"

Return ONLY the topic label. Example: "Academic Writing Help" or "Emotional Support Seeking"
"""
        
        topic_label = call_ollama(llm_prompt, max_tokens=30)
        # Clean up label
        topic_label = topic_label.replace('"', '').replace('\n', ' ').strip()
        if len(topic_label) > 50:
            topic_label = topic_label[:50] + "..."
        
        # Fallback: use top keywords if LLM fails
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
        
        print(f"      Cluster {cluster_id}: {len(cluster_texts)} msgs - {topic_label}")
    
    # Sort by size
    cluster_results.sort(key=lambda x: x['n_messages'], reverse=True)
    
    return {
        'group_name': group_name,
        'total_messages': len(texts),
        'n_clusters': len(cluster_results),
        'noise_pct': (cluster_labels == -1).sum() / len(texts) * 100,
        'clusters': cluster_results
    }


# ==============================================================================
# STEP 5: Run topic modeling for each attachment group
# ==============================================================================

print("\nSTEP 4: Running topic modeling for each attachment group...")

results = {}

# Process each group
for group_name, group_df in groups.items():
    results[group_name] = run_topic_modeling(
        group_df, 
        group_name, 
        max_messages=MAX_MESSAGES_PER_GROUP
    )

# ==============================================================================
# STEP 6: Save results
# ==============================================================================

print("\n\nSTEP 5: Saving results...")

# Save detailed JSON
json_path = f'{OUTPUT_DIR}/topic_modeling_results.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"  ✓ Saved: {json_path}")

# Create summary tables
summary_rows = []
for group_name, group_results in results.items():
    for cluster in group_results['clusters'][:10]:  # Top 10 clusters
        summary_rows.append({
            'Group': group_name,
            'Topic_Label': cluster['topic_label'],
            'N_Messages': cluster['n_messages'],
            'Pct_of_Group': f"{cluster['pct_of_total']:.1f}%",
            'Keywords': cluster['keywords']
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = f'{OUTPUT_DIR}/topic_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"  ✓ Saved: {summary_path}")

# Create formatted report
report_path = f'{OUTPUT_DIR}/TOPIC_MODELING_REPORT.txt'
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("TOPIC MODELING BY ATTACHMENT STYLE\n")
    f.write("=" * 80 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Method: MPNet embeddings + HDBSCAN clustering + Ollama ({OLLAMA_MODEL})\n")
    f.write("-" * 80 + "\n\n")
    
    for group_name, group_results in results.items():
        f.write(f"\n{'='*40}\n")
        f.write(f"{group_name.upper().replace('_', ' ')}\n")
        f.write(f"{'='*40}\n")
        f.write(f"Total messages: {group_results['total_messages']:,}\n")
        f.write(f"Clusters found: {group_results['n_clusters']}\n")
        f.write(f"Noise (unclustered): {group_results['noise_pct']:.1f}%\n\n")
        
        f.write("TOP TOPICS:\n")
        f.write("-" * 40 + "\n")
        
        for i, cluster in enumerate(group_results['clusters'][:10], 1):
            f.write(f"\n{i}. {cluster['topic_label']}\n")
            f.write(f"   Messages: {cluster['n_messages']} ({cluster['pct_of_total']:.1f}%)\n")
            f.write(f"   Keywords: {cluster['keywords']}\n")
            f.write(f"   Example: \"{cluster['representative_samples'][0][:100]}...\"\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")

print(f"  ✓ Saved: {report_path}")

# ==============================================================================
# STEP 7: Create comparison visualization
# ==============================================================================

print("\nSTEP 6: Creating comparison visualization...")

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    group_order = ['high_anxiety', 'low_anxiety', 'high_avoidance', 'low_avoidance']
    titles = ['High Attachment Anxiety', 'Low Attachment Anxiety', 
              'High Attachment Avoidance', 'Low Attachment Avoidance']
    
    for idx, (group_name, title) in enumerate(zip(group_order, titles)):
        ax = axes[idx // 2, idx % 2]
        
        if group_name in results:
            group_data = results[group_name]
            clusters = group_data['clusters'][:8]  # Top 8
            
            if clusters:
                labels = [c['topic_label'][:30] for c in clusters]
                sizes = [c['pct_of_total'] for c in clusters]
                
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(clusters)))
                
                bars = ax.barh(range(len(labels)), sizes, color=colors)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_xlabel('% of Messages', fontsize=10)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.invert_yaxis()
                
                # Add percentage labels
                for bar, size in zip(bars, sizes):
                    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                           f'{size:.1f}%', va='center', fontsize=8)
    
    plt.suptitle('Topic Distribution by Attachment Style', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig_path = f'{OUTPUT_DIR}/FIG_topic_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {fig_path}")

except Exception as e:
    print(f"  Warning: Could not create visualization: {e}")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "=" * 80)
print("TOPIC MODELING COMPLETE")
print("=" * 80)

print("\nKey findings by group:")
for group_name, group_results in results.items():
    print(f"\n{group_name.upper().replace('_', ' ')}:")
    for cluster in group_results['clusters'][:3]:
        print(f"  • {cluster['topic_label']} ({cluster['pct_of_total']:.1f}%)")

print(f"\nOutput files:")
print(f"  • {OUTPUT_DIR}/")
print(f"    ├── TOPIC_MODELING_REPORT.txt")
print(f"    ├── topic_summary.csv")
print(f"    ├── topic_modeling_results.json")
print(f"    └── FIG_topic_comparison.png")

print("\n" + "=" * 80)

