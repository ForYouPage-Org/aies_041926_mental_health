#!/usr/bin/env python3
"""
LLM-based Conversation Annotation using Local Ollama
=====================================================

Annotates sampled conversations into predefined topic categories.

Categories (from taxonomy):
- Writing: Edit/Critique, Personal Writing, Translation, Summary, Fiction
- Practical Guidance: How-To, Tutoring, Creative Ideation, Health/Self-Care
- Technical Help: Math, Data Analysis, Programming
- Multimedia: Create Image, Analyze Image, Other Media
- Seeking Information: Specific Info, Products, Cooking/Recipes
- Self-Expression: Chitchat, Relationships/Reflection, Games/Roleplay
- Other/Unknown: About Model, Other, Unclear

Date: December 29, 2025
"""

import pandas as pd
import numpy as np
import requests
import json
from tqdm import tqdm
import time

# ==============================================================================
# Configuration
# ==============================================================================

INPUT_PATH = '/Users/marxw/Sirius/FaccT_122925/results/251229T1251_topic_modeling_v2/sampled_100_conversations.csv'
OUTPUT_PATH = '/Users/marxw/Sirius/FaccT_122925/results/251229T1251_topic_modeling_v2/annotated_100_conversations.csv'

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "ministral-3:8b"

# Category taxonomy
CATEGORIES = {
    "Writing": [
        "Edit or Critique Provided Text",
        "Personal Writing or Communication", 
        "Translation",
        "Argument or Summary Generation",
        "Write Fiction"
    ],
    "Practical Guidance": [
        "How-To Advice",
        "Tutoring or Teaching",
        "Creative Ideation",
        "Health, Fitness, Beauty, or Self-Care"
    ],
    "Technical Help": [
        "Mathematical Calculation",
        "Data Analysis",
        "Computer Programming"
    ],
    "Multimedia": [
        "Create an Image",
        "Analyze an Image",
        "Generate or Retrieve Other Media"
    ],
    "Seeking Information": [
        "Specific Info",
        "Purchasable Products",
        "Cooking and Recipes"
    ],
    "Self-Expression": [
        "Greetings and Chitchat",
        "Relationships and Personal Reflection",
        "Games and Role Play"
    ],
    "Other/Unknown": [
        "Asking About the Model",
        "Other",
        "Unclear"
    ]
}

# Flatten categories for the prompt
ALL_SUBCATEGORIES = []
for topic, subcats in CATEGORIES.items():
    for subcat in subcats:
        ALL_SUBCATEGORIES.append(f"{topic}: {subcat}")

CATEGORY_LIST_STR = "\n".join([f"- {cat}" for cat in ALL_SUBCATEGORIES])

print("=" * 80)
print("LLM CONVERSATION ANNOTATION")
print("=" * 80)
print(f"Model: {OLLAMA_MODEL}")
print(f"Categories: {len(ALL_SUBCATEGORIES)} subcategories across {len(CATEGORIES)} topics")
print()

# ==============================================================================
# Load Data
# ==============================================================================

print("Loading sampled conversations...")
df = pd.read_csv(INPUT_PATH)
print(f"  Loaded {len(df)} messages from {df['conversation_id'].nunique()} conversations")

# Group by conversation
conversations = df.groupby('conversation_id').agg({
    'message': lambda x: ' ||| '.join(x.astype(str)),
    'participant_id': 'first',
    'conversation_title': 'first',
    'timestamp': 'first'
}).reset_index()

conversations['message_count'] = df.groupby('conversation_id').size().values

print(f"  Grouped into {len(conversations)} conversations")

# ==============================================================================
# Annotation Function
# ==============================================================================

def annotate_conversation(messages_text, title="", max_chars=1500):
    """
    Use Ollama to classify a conversation into one of the predefined categories.
    """
    # Truncate if too long
    if len(messages_text) > max_chars:
        messages_text = messages_text[:max_chars] + "..."
    
    prompt = f"""You are classifying ChatGPT conversations into categories. 

CONVERSATION TITLE: {title if title else "N/A"}

USER MESSAGES (separated by |||):
{messages_text}

CATEGORIES (choose exactly ONE):
{CATEGORY_LIST_STR}

Based on the user messages above, classify this conversation into exactly ONE category from the list.

Respond with ONLY the category in this exact format:
TOPIC: Subcategory

Example responses:
- "Writing: Translation"
- "Self-Expression: Relationships and Personal Reflection"
- "Technical Help: Computer Programming"

Your classification:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 50,
                    "temperature": 0.1
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            
            # Parse the response
            result_clean = result.split('\n')[0].strip()
            
            # Try to extract topic and subcategory
            if ':' in result_clean:
                parts = result_clean.split(':', 1)
                topic = parts[0].strip()
                subcategory = parts[1].strip() if len(parts) > 1 else ""
                
                # Validate topic
                valid_topics = list(CATEGORIES.keys())
                if topic not in valid_topics:
                    # Try to match
                    for vt in valid_topics:
                        if vt.lower() in topic.lower() or topic.lower() in vt.lower():
                            topic = vt
                            break
                
                return topic, subcategory, result_clean
            else:
                return "Other/Unknown", "Unclear", result_clean
        else:
            return "Other/Unknown", "Error", f"HTTP {response.status_code}"
            
    except Exception as e:
        return "Other/Unknown", "Error", str(e)


# ==============================================================================
# Annotate All Conversations
# ==============================================================================

print("\nAnnotating conversations with Ollama...")

results = []
for idx, row in tqdm(conversations.iterrows(), total=len(conversations), desc="Annotating"):
    topic, subcategory, raw_response = annotate_conversation(
        row['message'], 
        title=row.get('conversation_title', '')
    )
    
    results.append({
        'conversation_id': row['conversation_id'],
        'participant_id': row['participant_id'],
        'conversation_title': row['conversation_title'],
        'message_count': row['message_count'],
        'topic': topic,
        'subcategory': subcategory,
        'llm_raw_response': raw_response,
        'messages_preview': row['message'][:500] + "..." if len(row['message']) > 500 else row['message']
    })
    
    # Small delay to avoid overwhelming Ollama
    time.sleep(0.1)

# Create results DataFrame
results_df = pd.DataFrame(results)

# ==============================================================================
# Save Results
# ==============================================================================

print("\nSaving annotated conversations...")
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"  ✓ Saved: {OUTPUT_PATH}")

# ==============================================================================
# Summary Statistics
# ==============================================================================

print("\n" + "=" * 80)
print("ANNOTATION SUMMARY")
print("=" * 80)

# Topic distribution
print("\nTOPIC DISTRIBUTION:")
print("-" * 40)
topic_counts = results_df['topic'].value_counts()
for topic, count in topic_counts.items():
    pct = count / len(results_df) * 100
    print(f"  {topic:<25} {count:>3} ({pct:>5.1f}%)")

# Subcategory distribution
print("\n\nSUBCATEGORY DISTRIBUTION (Top 15):")
print("-" * 50)
subcat_counts = results_df['subcategory'].value_counts().head(15)
for subcat, count in subcat_counts.items():
    pct = count / len(results_df) * 100
    print(f"  {subcat:<40} {count:>3} ({pct:>5.1f}%)")

# Save summary
summary_path = OUTPUT_PATH.replace('.csv', '_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("CONVERSATION ANNOTATION SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total conversations annotated: {len(results_df)}\n")
    f.write(f"Model used: {OLLAMA_MODEL}\n\n")
    
    f.write("TOPIC DISTRIBUTION:\n")
    f.write("-" * 40 + "\n")
    for topic, count in topic_counts.items():
        pct = count / len(results_df) * 100
        f.write(f"  {topic:<25} {count:>3} ({pct:>5.1f}%)\n")
    
    f.write("\n\nSUBCATEGORY DISTRIBUTION:\n")
    f.write("-" * 50 + "\n")
    for subcat, count in results_df['subcategory'].value_counts().items():
        pct = count / len(results_df) * 100
        f.write(f"  {subcat:<40} {count:>3} ({pct:>5.1f}%)\n")

print(f"\n  ✓ Saved summary: {summary_path}")

print("\n" + "=" * 80)
print("ANNOTATION COMPLETE")
print("=" * 80)

