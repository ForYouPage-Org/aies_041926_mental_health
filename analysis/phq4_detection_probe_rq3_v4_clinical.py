#!/usr/bin/env python3
"""RQ3 Detection probe v4: per-item PHQ-2 / GAD-2 cascade on CLINICALLY-CURATED prompts.

The most stringent input curation possible: each participant's input is the
union of user_messages from conversations that a clinical expert (psychotherapist)
flagged as emotional disclosure (Clinical Expert Case Review #2). This pushes
input distribution as close to the literature's clinical-interview / mental-
health-subreddit text as our corpus permits.

For each participant who has at least one clinically-flagged conversation:
  - Concatenate all their flagged user_messages
  - Run the same per-item PHQ-2/GAD-2 cascade as v2/v3 (Qwen3:8B, anchored
    exemplars, CoT)
  - Compare predicted PHQ-2/GAD-2 sum against actual self-report

Sample is small (16 participants, ~9 with self-report PHQ-4) so we report
descriptive correlations and per-participant predictions rather than CV-AUC.
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
BUNDLES = Path("/tmp/clinical_review_bundles.json")
PHQ4_FRAME = BASE_DIR / "results" / "260419T1358_phq4_ai_controlled_regression" / "00_merged_analysis_data.csv"

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:8b"
MAX_CHARS = 10000

TS = datetime.now().strftime("%y%m%dT%H%M")
OUT = BASE_DIR / "results" / f"{TS}_phq4_detection_probe_v4_clinical"
OUT.mkdir(parents=True, exist_ok=True)

# Same items as v2/v3
ITEMS = {
    "phq2_q1": {
        "item_text": "Little interest or pleasure in doing things",
        "construct": "Anhedonia (loss of interest/pleasure)",
        "what_to_look_for": "References to losing interest in hobbies or activities; expressions of boredom, 'nothing is fun', flat-affect framing; reduced enthusiasm; lack of forward-looking enjoyment.",
        "exemplars": [
            ("0 (Not at all)", "- whats a good restaurant\n- write me a vacation itinerary i'm so excited\n- help me brainstorm fun bday party ideas", "Engaged, anticipating enjoyment."),
            ("1 (Several days)", "- ugh i feel meh today\n- everything feels boring lately\n- whats a fun hobby to try", "Mild boredom but still seeking activities."),
            ("2 (More than half)", "- i used to love painting but i don't feel like it anymore\n- nothing seems exciting these days\n- whats wrong with me i can't feel happy", "Loss of pleasure references."),
            ("3 (Nearly every day)", "- i don't enjoy anything anymore\n- i forced myself to go out and felt nothing\n- everything feels pointless", "Pervasive anhedonia."),
        ],
    },
    "phq2_q2": {
        "item_text": "Feeling down, depressed, or hopeless",
        "construct": "Depressed mood / hopelessness",
        "what_to_look_for": "Self-references to feeling down, sad, hopeless, worthless, stuck; pessimistic future framing; rumination on losses; statements implying things won't get better.",
        "exemplars": [
            ("0", "- write me a thank you note\n- whats the best way to learn javascript\n- explain quantum entanglement", "No affective load."),
            ("1", "- i had a rough week, comfort recipe\n- write a funny pep talk i need a laugh\n- i'm feeling kinda blue today", "Brief mood mentions but seeking lift."),
            ("2", "- i feel sad most days\n- i don't see things getting better\n- everything feels heavy lately", "Sustained low mood."),
            ("3", "- i feel hopeless about my future\n- whats the point of trying anymore\n- i'm worthless to the people around me", "Pervasive hopelessness."),
        ],
    },
    "gad2_q1": {
        "item_text": "Feeling nervous, anxious, or on edge",
        "construct": "Nervousness / anxious arousal",
        "what_to_look_for": "Self-references to nervous, anxious, on edge, jittery, restless; somatic anxiety (racing heart, stomach in knots); anticipation of panic.",
        "exemplars": [
            ("0", "- best practices for sql\n- summarize this article\n- recipe for chicken parmesan", "Calm, transactional."),
            ("1", "- i'm nervous about my presentation tomorrow\n- give me tips for staying calm\n- stomach feels weird before flights", "Situational nerves."),
            ("2", "- i feel on edge all the time\n- my heart races whenever someone messages me\n- i can't sit still lately", "Frequent anxious arousal."),
            ("3", "- i can't calm down, my chest feels tight every day\n- i wake up feeling panicked\n- everything feels overwhelming", "Pervasive anxiety."),
        ],
    },
    "gad2_q2": {
        "item_text": "Not being able to stop or control worrying",
        "construct": "Uncontrollable worry",
        "what_to_look_for": "Hypothetical 'what if' framing; reassurance-seeking; rumination on bad outcomes; difficulty stopping worry; mind-racing references.",
        "exemplars": [
            ("0", "- what time does the costco close\n- write a haiku about autumn\n- explain pep8", "No worry framing."),
            ("1", "- what if my cat doesn't like the new food\n- am i overreacting to feeling tired\n- how do i stop overthinking my last text", "Mild what-if framing."),
            ("2", "- i keep imagining worst-case scenarios about my health\n- i can't stop replaying conversations from last week\n- my mind keeps spinning", "Frequent intrusive worry."),
            ("3", "- i can't stop worrying about everything every day\n- my mind never shuts off\n- everything feels catastrophic", "Pervasive uncontrollable worry."),
        ],
    },
}


def build_item_prompt(item_key: str, messages_block: str) -> str:
    item = ITEMS[item_key]
    exemplars_text = "\n\n".join(
        f"### Score {sp}\nExample messages:\n{ex}\nWhy this score: {note}"
        for sp, ex, note in item["exemplars"]
    )
    return f"""You are a research assistant for an IRB-approved psycholinguistic study. You are NOT making a clinical diagnosis. Your task is to score linguistic indicators of one PHQ-4 screening item against a defined rubric.

PHQ-4 ITEM: "{item['item_text']}"
CONSTRUCT: {item['construct']}

WHAT TO LOOK FOR in the participant's prompts:
{item['what_to_look_for']}

SCORING RUBRIC (4 anchored exemplars, one per scale point):

{exemplars_text}

PARTICIPANT'S MESSAGES (clinically-flagged emotional-disclosure conversations):
=====
{messages_block}
=====

Step 1: Identify any cues in the messages above relevant to "{item['item_text']}". List them briefly. If none, say "no relevant cues".
Step 2: Compare the cues to the rubric exemplars and choose the closest scale point (0, 1, 2, or 3). It's expected that most participants in a young-adult sample will fall in the 0-2 range; only the most clear cases hit 3. If there are NO cues at all, score 0.

Respond ONLY with a single JSON object, no markdown:
{{"cues": "<short list of cues found, or 'no relevant cues'>", "score": <int 0|1|2|3>}}"""


def call_ollama(prompt: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                      "think": False, "keep_alive": "30m",
                      "options": {"temperature": 0, "num_ctx": 12288}},
                timeout=180,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            print(f"    fail attempt {attempt+1}/{retries}: {e}")
            time.sleep(min(8 * (attempt + 1), 30))
    return None


def parse_item(raw: str) -> dict | None:
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
                    d = json.loads(raw[start : i + 1])
                    s = int(d.get("score", -1))
                    if s in (0, 1, 2, 3):
                        return {"cues": str(d.get("cues", ""))[:300], "score": s}
                except Exception:
                    return None
    return None


def build_block(msgs: list[str], max_chars: int) -> str:
    out, used = [], 0
    for m in msgs:
        line = f"- {m.strip().replace(chr(10),' ')[:500]}"
        if used + len(line) + 2 > max_chars:
            break
        out.append(line)
        used += len(line) + 2
    return "\n".join(out)


print(f"Reading bundles: {BUNDLES}")
with open(BUNDLES) as f:
    bundles = json.load(f)
print(f"  {len(bundles)} clinical-review participants")

print(f"Reading PHQ-4 frame: {PHQ4_FRAME}")
phq = pd.read_csv(PHQ4_FRAME)
phq["Prolific_ID"] = phq["Prolific_ID"].astype(str)
phq["PHQ2_Dep_Pos"] = (phq["PHQ4_Depression"] >= 3).astype(int)
phq["GAD2_Anx_Pos"] = (phq["PHQ4_Anxiety"] >= 3).astype(int)


print("\n=== Scoring cascade on clinical-review bundles ===")
rows = []
t0 = time.time()
for i, (pid, msgs) in enumerate(bundles.items()):
    block = build_block(msgs, MAX_CHARS)
    item_scores = {}
    item_cues = {}
    for ik in ITEMS.keys():
        prompt = build_item_prompt(ik, block)
        raw = call_ollama(prompt)
        parsed = parse_item(raw or "")
        if parsed is None:
            item_scores[ik] = np.nan
            item_cues[f"{ik}_cues"] = "PARSE_FAIL"
        else:
            item_scores[ik] = parsed["score"]
            item_cues[f"{ik}_cues"] = parsed["cues"]
    row = {"participant_id": pid, "n_msgs": len(msgs), **item_scores, **item_cues}
    rows.append(row)
    print(f"  [{i+1}/{len(bundles)}] {pid} ({len(msgs)} msgs): "
          f"phq2={item_scores.get('phq2_q1')}+{item_scores.get('phq2_q2')}; "
          f"gad2={item_scores.get('gad2_q1')}+{item_scores.get('gad2_q2')} "
          f"({(time.time()-t0)/(i+1):.1f}s/p)")

results = pd.DataFrame(rows)
results["PHQ2_pred"] = results[["phq2_q1", "phq2_q2"]].sum(axis=1, min_count=1)
results["GAD2_pred"] = results[["gad2_q1", "gad2_q2"]].sum(axis=1, min_count=1)
results.to_csv(OUT / "01_clinical_cascade_scores.csv", index=False)


# Merge with self-report PHQ-4
merged = results.merge(
    phq[["Prolific_ID", "PHQ4_Depression", "PHQ4_Anxiety", "PHQ2_Dep_Pos", "GAD2_Anx_Pos"]],
    left_on="participant_id", right_on="Prolific_ID", how="left",
)
merged.to_csv(OUT / "02_merged_with_self_report.csv", index=False)
n_with_phq = merged["PHQ4_Depression"].notna().sum()
print(f"\n{n_with_phq}/{len(merged)} clinical-review participants have self-report PHQ-4")

print("\n=== Per-participant predicted vs actual ===")
print(merged[["participant_id", "n_msgs", "PHQ2_pred", "PHQ4_Depression", "PHQ2_Dep_Pos",
              "GAD2_pred", "PHQ4_Anxiety", "GAD2_Anx_Pos"]].to_string(index=False))


# Correlations and group test
print("\n=== Correlations (clinical-review-input cascade vs self-report) ===")
sub = merged.dropna(subset=["PHQ2_pred", "PHQ4_Depression"])
if len(sub) >= 4:
    rp, pp = ss.pearsonr(sub["PHQ2_pred"], sub["PHQ4_Depression"])
    rs, ps = ss.spearmanr(sub["PHQ2_pred"], sub["PHQ4_Depression"])
    print(f"  PHQ2_pred vs PHQ-2 dep sum: pearson r={rp:.3f} (p={pp:.4f}), spearman r={rs:.3f} (p={ps:.4f}), N={len(sub)}")
sub = merged.dropna(subset=["GAD2_pred", "PHQ4_Anxiety"])
if len(sub) >= 4:
    rp, pp = ss.pearsonr(sub["GAD2_pred"], sub["PHQ4_Anxiety"])
    rs, ps = ss.spearmanr(sub["GAD2_pred"], sub["PHQ4_Anxiety"])
    print(f"  GAD2_pred vs GAD-2 anx sum: pearson r={rp:.3f} (p={pp:.4f}), spearman r={rs:.3f} (p={ps:.4f}), N={len(sub)}")

# Mean cascade score split by binary screen
sub = merged.dropna(subset=["PHQ2_pred", "PHQ2_Dep_Pos"])
if len(sub) >= 4:
    a = sub.loc[sub["PHQ2_Dep_Pos"]==1, "PHQ2_pred"]
    b = sub.loc[sub["PHQ2_Dep_Pos"]==0, "PHQ2_pred"]
    print(f"  PHQ2_pred: dep+ M={a.mean():.2f} (n={len(a)}), dep- M={b.mean():.2f} (n={len(b)})")
    if len(a)>=2 and len(b)>=2:
        u, p = ss.mannwhitneyu(a, b, alternative='greater')
        print(f"    Mann-Whitney (greater): U={u:.0f}, p={p:.4f}")

sub = merged.dropna(subset=["GAD2_pred", "GAD2_Anx_Pos"])
if len(sub) >= 4:
    a = sub.loc[sub["GAD2_Anx_Pos"]==1, "GAD2_pred"]
    b = sub.loc[sub["GAD2_Anx_Pos"]==0, "GAD2_pred"]
    print(f"  GAD2_pred: anx+ M={a.mean():.2f} (n={len(a)}), anx- M={b.mean():.2f} (n={len(b)})")
    if len(a)>=2 and len(b)>=2:
        u, p = ss.mannwhitneyu(a, b, alternative='greater')
        print(f"    Mann-Whitney (greater): U={u:.0f}, p={p:.4f}")


# Floor effect
print("\n=== Floor effect under clinical curation ===")
for ik in ITEMS.keys():
    n_zero = (results[ik] == 0).sum()
    n_total = results[ik].notna().sum()
    print(f"  {ik}: {n_zero}/{n_total} score 0  ({n_zero/max(n_total,1):.0%})")


print(f"\nDone. Output: {OUT}")
