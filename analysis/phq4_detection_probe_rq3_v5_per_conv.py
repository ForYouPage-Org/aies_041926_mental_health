#!/usr/bin/env python3
"""RQ3 Detection probe v5: per-conversation scoring + multi-strategy aggregation.

Addresses the v4 conversation-vs-trait gap: instead of bundling all of a
participant's clinically-flagged emotional-disclosure conversations into one
LLM prompt, score each conversation INDEPENDENTLY, then test multiple
participant-level aggregation strategies.

For each of the 29 clinically-flagged conversations (Clinical Expert Case
Review #2; xlsx tabs + Unamed CSV):
  - Run the per-item PHQ-2/GAD-2 cascade on the conversation alone
  - Get per-conversation PHQ2_pred (0-6) and GAD2_pred (0-6)

Then aggregate at participant level using five strategies:
  - mean   (trait = average per-event severity)
  - max    (trait = most severe event)
  - sum    (trait = accumulated symptomatic evidence)
  - count  (trait = # conversations with PHQ/GAD-pred >= 3, screen-positive)
  - prop   (trait = fraction of conversations with screen-positive prediction)

For each aggregation, correlate with self-report PHQ-4
(Pearson + Spearman + group-mean by binary screen).
"""

from __future__ import annotations

import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.stats as ss

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
PHQ4_FRAME = BASE_DIR / "results" / "260419T1358_phq4_ai_controlled_regression" / "00_merged_analysis_data.csv"
XLSX = BASE_DIR / "Clinical Expert Case Review [#2]-STUDY00021493.xlsx"
CSV = BASE_DIR / "Clinical Expert Case Review [#2]-STUDY00021493 - Unamed.csv"
MAPPING = Path("/tmp/clinical_review_mappings.csv")

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:8b"
MAX_CHARS = 8000

TS = datetime.now().strftime("%y%m%dT%H%M")
OUT = BASE_DIR / "results" / f"{TS}_phq4_detection_probe_v5_per_conv"
OUT.mkdir(parents=True, exist_ok=True)


# Same items as v2/v3/v4
ITEMS = {
    "phq2_q1": {
        "item_text": "Little interest or pleasure in doing things",
        "construct": "Anhedonia",
        "what_to_look_for": "References to losing interest in hobbies or activities; expressions of boredom, 'nothing is fun', flat-affect framing; reduced enthusiasm; lack of forward-looking enjoyment.",
        "exemplars": [
            ("0", "- whats a good restaurant\n- write me a vacation itinerary i'm so excited", "Engaged."),
            ("1", "- ugh i feel meh today\n- everything feels boring lately", "Mild boredom."),
            ("2", "- i used to love painting but i don't feel like it anymore\n- nothing seems exciting these days", "Loss of pleasure."),
            ("3", "- i don't enjoy anything anymore\n- forced myself to go out and felt nothing", "Pervasive anhedonia."),
        ],
    },
    "phq2_q2": {
        "item_text": "Feeling down, depressed, or hopeless",
        "construct": "Depressed mood",
        "what_to_look_for": "Self-references to feeling down, sad, hopeless, worthless, stuck; pessimistic framing; rumination on losses.",
        "exemplars": [
            ("0", "- write me a thank you note\n- whats the best way to learn javascript", "No affective load."),
            ("1", "- i had a rough week, comfort recipe\n- i'm feeling kinda blue today", "Brief mood mentions."),
            ("2", "- i feel sad most days\n- i don't see things getting better", "Sustained low mood."),
            ("3", "- i feel hopeless about my future\n- whats the point of trying anymore", "Pervasive hopelessness."),
        ],
    },
    "gad2_q1": {
        "item_text": "Feeling nervous, anxious, or on edge",
        "construct": "Nervousness",
        "what_to_look_for": "Self-references to nervous, anxious, on edge, jittery; somatic anxiety; anticipation of panic.",
        "exemplars": [
            ("0", "- best practices for sql\n- summarize this article", "Calm, transactional."),
            ("1", "- i'm nervous about my presentation tomorrow\n- give me tips for staying calm", "Situational nerves."),
            ("2", "- i feel on edge all the time\n- my heart races whenever someone messages me", "Frequent anxious arousal."),
            ("3", "- i can't calm down, my chest feels tight every day\n- i wake up feeling panicked", "Pervasive anxiety."),
        ],
    },
    "gad2_q2": {
        "item_text": "Not being able to stop or control worrying",
        "construct": "Uncontrollable worry",
        "what_to_look_for": "Hypothetical 'what if' framing; reassurance-seeking; rumination on bad outcomes.",
        "exemplars": [
            ("0", "- what time does the costco close\n- write a haiku about autumn", "No worry framing."),
            ("1", "- what if my cat doesn't like the new food\n- am i overreacting", "Mild what-if framing."),
            ("2", "- i keep imagining worst-case scenarios about my health\n- my mind keeps spinning", "Frequent intrusive worry."),
            ("3", "- i can't stop worrying about everything every day\n- everything feels catastrophic", "Pervasive uncontrollable worry."),
        ],
    },
}


def build_item_prompt(item_key: str, messages_block: str) -> str:
    item = ITEMS[item_key]
    exemplars_text = "\n\n".join(
        f"### Score {sp}\nExample:\n{ex}\nWhy: {note}"
        for sp, ex, note in item["exemplars"]
    )
    return f"""You are a research assistant for an IRB-approved psycholinguistic study. NOT a clinical diagnosis.

PHQ-4 ITEM: "{item['item_text']}"
CONSTRUCT: {item['construct']}

WHAT TO LOOK FOR: {item['what_to_look_for']}

ANCHORED EXEMPLARS:
{exemplars_text}

PARTICIPANT'S MESSAGES (one conversation):
=====
{messages_block}
=====

Step 1: Identify any cues briefly (or 'no relevant cues').
Step 2: Choose closest scale point 0/1/2/3.

Respond ONLY: {{"cues": "...", "score": <int 0|1|2|3>}}"""


def call_ollama(prompt: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                      "think": False, "keep_alive": "30m",
                      "options": {"temperature": 0, "num_ctx": 8192}},
                timeout=180,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            print(f"    fail {attempt+1}: {e}")
            time.sleep(min(8 * (attempt + 1), 30))
    return None


def parse_item(raw: str) -> int | None:
    if not raw:
        return None
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    d = json.loads(raw[start:i+1])
                    s = int(d.get("score", -1))
                    if s in (0, 1, 2, 3):
                        return s
                except Exception:
                    return None
    return None


def build_block(msgs: list[str], max_chars: int) -> str:
    out, used = [], 0
    for m in msgs:
        line = f"- {m.strip().replace(chr(10), ' ')[:600]}"
        if used + len(line) + 2 > max_chars:
            break
        out.append(line)
        used += len(line) + 2
    return "\n".join(out)


# =============================================================
# Build per-conversation list (29 conversations across 16 pids)
# =============================================================
xls = pd.ExcelFile(XLSX)
conversations: list[dict] = []  # {label, participant_id, msgs}
for sn in xls.sheet_names:
    if sn == "Unamed":
        continue
    df = pd.read_excel(XLSX, sheet_name=sn)
    if "user_message" in df.columns:
        msgs = df["user_message"].dropna().astype(str).tolist()
        conversations.append({"label": sn, "msgs": msgs})

csv = pd.read_csv(CSV, header=None, low_memory=False)
header_text = csv.columns[0]
conversations.append({
    "label": "Unamed_0",
    "msgs": [m.strip() for m in str(header_text).split("\n---\n") if m.strip()],
})
for i in range(len(csv)):
    text = str(csv.iloc[i, 0])
    msgs = [m.strip() for m in text.split("\n---\n") if m.strip()]
    conversations.append({"label": f"Unamed_{i+1}", "msgs": msgs})

# Attach participant_id from mapping
mapping = pd.read_csv(MAPPING)
mp = dict(zip(mapping["pseudonym"], mapping["participant_id"]))
for c in conversations:
    c["participant_id"] = mp.get(c["label"])

print(f"Total conversations: {len(conversations)}")
print(f"With matched pid: {sum(1 for c in conversations if pd.notna(c['participant_id']))}")


# =============================================================
# Score each conversation independently
# =============================================================
print(f"\n=== Per-conversation cascade ({OLLAMA_MODEL}) ===")
rows = []
t0 = time.time()
for i, c in enumerate(conversations):
    if not c["msgs"]:
        continue
    block = build_block(c["msgs"], MAX_CHARS)
    item_scores = {}
    for ik in ITEMS:
        prompt = build_item_prompt(ik, block)
        raw = call_ollama(prompt)
        s = parse_item(raw or "")
        item_scores[ik] = s if s is not None else np.nan
    rec = {
        "conv_label": c["label"],
        "participant_id": c["participant_id"],
        "n_msgs": len(c["msgs"]),
        **item_scores,
    }
    rec["PHQ2_conv_pred"] = (
        rec["phq2_q1"] + rec["phq2_q2"] if not pd.isna(rec["phq2_q1"]) and not pd.isna(rec["phq2_q2"]) else np.nan
    )
    rec["GAD2_conv_pred"] = (
        rec["gad2_q1"] + rec["gad2_q2"] if not pd.isna(rec["gad2_q1"]) and not pd.isna(rec["gad2_q2"]) else np.nan
    )
    rows.append(rec)
    elapsed = (time.time() - t0) / (i + 1)
    print(f"  [{i+1}/{len(conversations)}] {c['label']} ({len(c['msgs'])}m) -> "
          f"phq={rec['PHQ2_conv_pred']}, gad={rec['GAD2_conv_pred']} "
          f"({elapsed:.1f}s/conv)")

per_conv = pd.DataFrame(rows)
per_conv.to_csv(OUT / "01_per_conversation_scores.csv", index=False)
print(f"Saved per-conversation scores: {OUT / '01_per_conversation_scores.csv'}")


# =============================================================
# Aggregate to participant level (5 strategies)
# =============================================================
print(f"\n=== Per-participant aggregation strategies ===")
by_pid = per_conv.dropna(subset=["participant_id"]).groupby("participant_id")
agg_records = []
for pid, group in by_pid:
    g = group.dropna(subset=["PHQ2_conv_pred", "GAD2_conv_pred"])
    if len(g) == 0:
        continue
    rec = {"participant_id": pid, "n_conv": len(g)}
    for col, prefix in [("PHQ2_conv_pred", "PHQ2"), ("GAD2_conv_pred", "GAD2")]:
        scores = g[col].values
        rec[f"{prefix}_mean"] = float(np.mean(scores))
        rec[f"{prefix}_max"]  = float(np.max(scores))
        rec[f"{prefix}_sum"]  = float(np.sum(scores))
        rec[f"{prefix}_count_pos"] = int((scores >= 3).sum())  # # convs at screen+
        rec[f"{prefix}_prop_pos"] = float((scores >= 3).mean())  # fraction at screen+
    agg_records.append(rec)

agg = pd.DataFrame(agg_records)
print(f"  {len(agg)} participants with aggregated scores")


# =============================================================
# Merge with self-report PHQ-4
# =============================================================
phq = pd.read_csv(PHQ4_FRAME)
phq["Prolific_ID"] = phq["Prolific_ID"].astype(str)
phq["PHQ2_Dep_Pos"] = (phq["PHQ4_Depression"] >= 3).astype(int)
phq["GAD2_Anx_Pos"] = (phq["PHQ4_Anxiety"] >= 3).astype(int)
agg["participant_id"] = agg["participant_id"].astype(str)
merged = agg.merge(
    phq[["Prolific_ID", "PHQ4_Depression", "PHQ4_Anxiety", "PHQ2_Dep_Pos", "GAD2_Anx_Pos"]],
    left_on="participant_id", right_on="Prolific_ID", how="left",
)
merged.to_csv(OUT / "02_aggregated_with_self_report.csv", index=False)
print(f"  {merged['PHQ4_Depression'].notna().sum()} of {len(merged)} have PHQ-4 self-report")


# =============================================================
# Test each aggregation against self-report
# =============================================================
print(f"\n=== Aggregation strategy x self-report correlation ===")
results = []
sub = merged.dropna(subset=["PHQ4_Depression"])
print(f"  N (with PHQ-4) = {len(sub)}")
for prefix, sr_col, sr_pos_col in [("PHQ2", "PHQ4_Depression", "PHQ2_Dep_Pos"),
                                    ("GAD2", "PHQ4_Anxiety",    "GAD2_Anx_Pos")]:
    print(f"\n  --- {prefix} aggregation -> {sr_col} ---")
    for agg_name in ["mean", "max", "sum", "count_pos", "prop_pos"]:
        col = f"{prefix}_{agg_name}"
        s = sub.dropna(subset=[col])
        if len(s) < 4:
            continue
        rp, pp = ss.pearsonr(s[col], s[sr_col])
        rs, ps = ss.spearmanr(s[col], s[sr_col])
        # Group-mean test (binary screen)
        a = s.loc[s[sr_pos_col] == 1, col]
        b = s.loc[s[sr_pos_col] == 0, col]
        if len(a) >= 2 and len(b) >= 2:
            u, p_mwu = ss.mannwhitneyu(a, b, alternative="greater")
            mwu_str = f"MWU+={p_mwu:.3f} (M+={a.mean():.2f},M-={b.mean():.2f})"
        else:
            mwu_str = "MWU n/a"
        marker = " *" if (pp < .05 or ps < .05 or (len(a)>=2 and len(b)>=2 and p_mwu < .05)) else ""
        print(f"    {agg_name:10s}  pearson r={rp:+.3f} (p={pp:.3f}), spearman r={rs:+.3f} (p={ps:.3f}); {mwu_str}{marker}")
        results.append({
            "construct": prefix, "aggregation": agg_name, "n": len(s),
            "pearson_r": rp, "pearson_p": pp,
            "spearman_r": rs, "spearman_p": ps,
            "M_pos": float(a.mean()) if len(a) else np.nan,
            "M_neg": float(b.mean()) if len(b) else np.nan,
            "mwu_p": p_mwu if (len(a)>=2 and len(b)>=2) else np.nan,
        })
res_df = pd.DataFrame(results)
res_df.to_csv(OUT / "03_aggregation_correlations.csv", index=False)


# =============================================================
# Per-participant table for interpretation
# =============================================================
display_cols = ["participant_id", "n_conv",
                "PHQ4_Depression", "PHQ2_Dep_Pos", "PHQ2_mean", "PHQ2_max", "PHQ2_sum", "PHQ2_count_pos",
                "PHQ4_Anxiety",    "GAD2_Anx_Pos", "GAD2_mean", "GAD2_max", "GAD2_sum", "GAD2_count_pos"]
print(f"\n=== Per-participant aggregated scores vs self-report ===")
disp = merged[display_cols].sort_values("PHQ4_Depression", ascending=False)
print(disp.to_string(index=False))

print(f"\nDone. Output: {OUT}")
