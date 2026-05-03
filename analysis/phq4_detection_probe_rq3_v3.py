#!/usr/bin/env python3
"""RQ3 Detection probe v3: per-item PHQ-2 / GAD-2 cascade on AFFECT-CURATED prompts.

KEY CHANGE FROM V2: input subset of messages per participant is no longer a
random sample. We select the top-K personally-relevant messages per
participant, defined as messages that:
  - have at least 7 words (substantive),
  - contain at least one first-person pronoun (about the writer),
  - contain at least one psychological-content word (Affect | cogproc | feeling),
ranked by `psych_score = WC * (Affect + cogproc + feeling) / 100`
(absolute count of psychological-content words, weighted by message length).

Rationale: prior LLM-prompting work on depression detection (LlaMADRS,
Mental-LLM, MentaLLaMA, Cognitive-Mental-LLM, eRisk@CLEF) operates on input
that is itself self-selected for affect/personal content -- clinical interview
transcripts (LlaMADRS) or posts written in mental-health subreddits (Reddit
Dreaddit / DepSeverity / SDCNL / r/depression / r/SuicideWatch / r/anxiety).
Our v2 fed the model 60 random messages drawn from naturalistic ChatGPT use,
which is ~60% homework/code/recipes and only a small fraction affect-laden.
v3 curates the input to make the LLM detection task input-distribution-
comparable to the literature.

Privacy interpretation: this matches a more concerning threat model -- an
attacker with full chat-history access who SELECTS the most affect-laden
messages before scoring -- rather than the user-friendly random-sample
threat model in v2. v2 + v3 together bracket the privacy-threat space.

Cascade per participant (4 LLM calls): unchanged from v2 (per-item
PHQ-2/GAD-2 with researcher-persona + 4 anchored exemplars + CoT;
aggregation = deterministic sum).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

BASE_DIR = Path(__file__).parent.parent
LIWC_MERGED = BASE_DIR / "results" / "260419T1358_phq4_liwc_analysis" / "00_merged_mh_liwc_data.csv"
PROMPTS_FILE = BASE_DIR / "results" / "251229T1212_NRC_VAD_analysis" / "filtered_user_prompts_lt20words.csv"
PER_MSG_LIWC = BASE_DIR / "results" / "251229T1212_NRC_VAD_analysis" / "LIWC-22 Results - filtered_user_prompts_lt20word___ - LIWC Analysis.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--liwc-merged", default=str(LIWC_MERGED))
ap.add_argument("--prompts", default=str(PROMPTS_FILE))
ap.add_argument("--per-msg-liwc", default=str(PER_MSG_LIWC),
                help="Per-message LIWC scores used to rank affect-relevance.")
ap.add_argument("--ollama-model", default="qwen3:8b")
ap.add_argument("--ollama-url", default="http://localhost:11434")
ap.add_argument("--n-msgs-per-participant", type=int, default=20,
                help="Top-K personally-relevant messages per participant.")
ap.add_argument("--min-words-per-msg", type=int, default=7,
                help="Minimum words for a candidate message (substantive).")
ap.add_argument("--max-chars-prompt", type=int, default=10000)
ap.add_argument("--cv-folds", type=int, default=5)
ap.add_argument("--cv-repeats", type=int, default=5)
ap.add_argument("--n-boot-auc", type=int, default=2000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--cached-llm", default="")
ap.add_argument("--out-suffix", default="phq4_detection_probe_v3_curated")
args = ap.parse_args()

TS = datetime.now().strftime("%y%m%dT%H%M")
OUT = BASE_DIR / "results" / f"{TS}_{args.out_suffix}"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "cache").mkdir(parents=True, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)


# ============================================================
# PHQ-2 / GAD-2 item-level prompts
# ============================================================
# Each item gets its own prompt with: persona + item text + scale anchors +
# 4 synthetic exemplars + CoT instruction.

ITEMS = {
    "phq2_q1": {
        "item_text": "Little interest or pleasure in doing things",
        "construct": "Anhedonia (loss of interest/pleasure in usual activities)",
        "what_to_look_for": "References to losing interest in hobbies, social activities, or things they used to enjoy; expressions of boredom or 'nothing is fun'; reduced enthusiasm; flat-affect framing; lack of forward-looking enjoyment language.",
        "exemplars": [
            ("0 (Not at all)",
             "- whats a good restaurant near union square\n- can you write me a vacation itinerary for portugal i'm so excited\n- help me brainstorm fun bday party ideas\n- recommend a fantasy book i'd love",
             "Engaged, future-positive, anticipating enjoyment."),
            ("1 (Several days)",
             "- ugh i feel meh today, what should i do this weekend\n- everything feels boring lately can you suggest something\n- help me get motivated to start this project\n- whats a fun hobby to try",
             "Mild boredom/low energy mentions but still seeking activities."),
            ("2 (More than half the days)",
             "- i used to love painting but i don't feel like it anymore\n- nothing seems exciting these days\n- i can't get myself to do anything fun\n- what should i do when nothing makes me happy",
             "Explicit references to loss of pleasure and difficulty engaging."),
            ("3 (Nearly every day)",
             "- i don't enjoy anything anymore why is that\n- i used to like music but now its just noise\n- i forced myself to go out and felt nothing the whole time\n- whats wrong with me i can't feel happy doing things i used to love",
             "Pervasive anhedonia, loss of pleasure in formerly valued activities."),
        ],
    },
    "phq2_q2": {
        "item_text": "Feeling down, depressed, or hopeless",
        "construct": "Depressed mood / hopelessness",
        "what_to_look_for": "Self-references to feeling down, sad, hopeless, worthless, or stuck; pessimistic framing of the future; rumination on relational losses; statements implying things won't get better.",
        "exemplars": [
            ("0 (Not at all)",
             "- write me a thank you note to my professor\n- whats the best way to learn javascript\n- can you help debug this python error\n- explain quantum entanglement simply",
             "Task-focused, no affective load."),
            ("1 (Several days)",
             "- i had a rough week, can you suggest a comfort recipe\n- write a funny pep talk i need a laugh\n- i'm feeling kinda blue today help me cheer up\n- give me a pep talk for my interview",
             "Brief mood mentions but seeking lift."),
            ("2 (More than half the days)",
             "- i feel sad most days, what could help\n- i don't see things getting better\n- everything feels heavy lately\n- why am i always like this",
             "Sustained low mood, mild hopelessness."),
            ("3 (Nearly every day)",
             "- i feel hopeless about my future\n- whats the point of trying anymore\n- i'm worthless to the people around me\n- i don't know if things will ever get better",
             "Pervasive hopelessness, worthlessness."),
        ],
    },
    "gad2_q1": {
        "item_text": "Feeling nervous, anxious, or on edge",
        "construct": "Nervousness / anxious arousal",
        "what_to_look_for": "Self-references to feeling nervous, anxious, on edge, jittery, jumpy, or restless; somatic anxiety mentions (racing heart, stomach in knots, can't sit still); anticipation of panic.",
        "exemplars": [
            ("0 (Not at all)",
             "- best practices for sql optimization\n- summarize this article for me\n- how does compound interest work\n- recipe for chicken parmesan",
             "Calm, transactional."),
            ("1 (Several days)",
             "- i'm a little nervous about my presentation tomorrow help me practice\n- give me tips for staying calm during a job interview\n- stomach feels weird before flights any tips\n- how to relax before a big test",
             "Situational nerves about specific events."),
            ("2 (More than half the days)",
             "- i feel on edge all the time, what helps\n- my heart races whenever someone messages me\n- i can't sit still lately\n- why do i feel jittery for no reason",
             "Frequent anxious arousal, often somatic."),
            ("3 (Nearly every day)",
             "- i can't calm down, my chest feels tight every day\n- i wake up feeling panicked\n- i'm constantly on edge and exhausted from it\n- everything feels overwhelming and i can't relax",
             "Pervasive anxiety, somatic distress."),
        ],
    },
    "gad2_q2": {
        "item_text": "Not being able to stop or control worrying",
        "construct": "Uncontrollable worry / intrusive worry",
        "what_to_look_for": "Hypothetical 'what if' framing; reassurance-seeking on the same topic repeatedly; rumination on possible bad outcomes; difficulty stopping worry; mind-racing references; repetitive checking-style questions.",
        "exemplars": [
            ("0 (Not at all)",
             "- what time does the costco close\n- write a haiku about autumn\n- whats the capital of mongolia\n- explain pep8 to me",
             "No worry framing."),
            ("1 (Several days)",
             "- what if my cat doesn't like the new food, should i be worried\n- is it a problem if i forgot to reply to my friend yesterday\n- how do i stop overthinking my last text\n- am i overreacting to feeling tired today",
             "Mild what-if framing on specific topics."),
            ("2 (More than half the days)",
             "- i keep imagining worst-case scenarios about my health\n- i can't stop replaying conversations from last week\n- what if my partner is mad at me, how do i tell\n- my mind keeps spinning, how do i stop",
             "Frequent intrusive worry, reassurance-seeking patterns."),
            ("3 (Nearly every day)",
             "- i can't stop worrying about everything, every day\n- my mind never shuts off i worry constantly\n- everything feels catastrophic, what if x what if y\n- i checked the same thing 10 times today and still feel uneasy",
             "Pervasive uncontrollable worry."),
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

PHQ-4 ITEM: "{item["item_text"]}"
CONSTRUCT: {item["construct"]}

WHAT TO LOOK FOR in the participant's everyday ChatGPT prompts:
{item["what_to_look_for"]}

SCORING RUBRIC (4 anchored exemplars, one per scale point):

{exemplars_text}

PARTICIPANT'S SAMPLED MESSAGES (across many ChatGPT conversations):
=====
{messages_block}
=====

Step 1: Identify any cues in the messages above relevant to "{item["item_text"]}". List them briefly. If none, say "no relevant cues".
Step 2: Compare the cues to the rubric exemplars and choose the closest scale point (0, 1, 2, or 3). It's expected that most participants in a young-adult sample will fall in the 0-2 range; only the most clear cases hit 3. If there are NO cues at all, score 0.

Respond ONLY with a single JSON object, no markdown:
{{"cues": "<short list of cues found, or 'no relevant cues'>", "score": <int 0|1|2|3>}}"""


# ============================================================
# Load data
# ============================================================
print(f"Loading LIWC: {args.liwc_merged}")
liwc = pd.read_csv(args.liwc_merged)
liwc["PHQ2_Dep_Sum"] = liwc["PHQ4_Depression"]   # PHQ-2 dep raw 0-6
liwc["GAD2_Anx_Sum"] = liwc["PHQ4_Anxiety"]      # GAD-2 anx raw 0-6
liwc["PHQ2_Dep_Pos"] = (liwc["PHQ2_Dep_Sum"] >= 3).astype(int)
liwc["GAD2_Anx_Pos"] = (liwc["GAD2_Anx_Sum"] >= 3).astype(int)
print(f"  N={len(liwc)}, Dep+ N={liwc['PHQ2_Dep_Pos'].sum()}, Anx+ N={liwc['GAD2_Anx_Pos'].sum()}")

print(f"Loading per-message LIWC for affect-curation: {args.per_msg_liwc}")
per_msg = pd.read_csv(args.per_msg_liwc, low_memory=False)
per_msg = per_msg[per_msg["user_or_ai"] == "user"].copy()
per_msg["message"] = per_msg["message"].astype(str)
# Personally-relevant content selector:
# - WC >= min_words (substantive)
# - >=1 first-person pronoun (about the writer)
# - >=1 word in Affect | cogproc | feeling (psychological content)
# - rank by psych_score = WC * (Affect + cogproc + feeling) / 100
for col in ["Affect", "cogproc", "feeling", "ppron", "WC"]:
    per_msg[col] = pd.to_numeric(per_msg[col], errors="coerce").fillna(0)
per_msg["psych_score"] = per_msg["WC"] * (per_msg["Affect"] + per_msg["cogproc"] + per_msg["feeling"]) / 100.0
candidates = per_msg[
    (per_msg["WC"] >= args.min_words_per_msg)
    & (per_msg["ppron"] > 0)
    & ((per_msg["Affect"] > 0) | (per_msg["cogproc"] > 0) | (per_msg["feeling"] > 0))
].copy()
print(f"  Curation: {len(candidates)} personally-relevant messages "
      f"(WC>={args.min_words_per_msg}, has ppron, has psych content) from {candidates['participant_id'].nunique()} participants")
per_pp = candidates.groupby("participant_id").size()
print(f"  Per-participant candidate counts: median={int(per_pp.median())}, "
      f">=10: {(per_pp>=10).sum()}, >=20: {(per_pp>=20).sum()}, >=30: {(per_pp>=30).sum()}")
# Build per-participant top-K message lists, ordered by psych_score desc
prompts_curated: dict[str, list[str]] = {}
for pid, group in candidates.groupby("participant_id"):
    top = group.nlargest(args.n_msgs_per_participant, "psych_score")
    prompts_curated[str(pid)] = top["message"].tolist()


# ============================================================
# LLM call wrapper
# ============================================================
def call_ollama(prompt: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{args.ollama_url}/api/generate",
                json={
                    "model": args.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "keep_alive": "30m",
                    "options": {"temperature": 0, "num_ctx": 12288},
                },
                timeout=180,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            print(f"    Ollama call failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(min(8 * (attempt + 1), 30))
    return None


def parse_item_response(raw: str) -> dict | None:
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
                    if s not in (0, 1, 2, 3):
                        return None
                    return {"cues": str(d.get("cues", ""))[:300], "score": s}
                except Exception:
                    return None
    return None


def build_messages_block(msgs: list[str], max_chars: int) -> str:
    out, used = [], 0
    for m in msgs:
        line = f"- {m.strip()}"
        if used + len(line) + 2 > max_chars:
            break
        out.append(line)
        used += len(line) + 2
    return "\n".join(out)


# ============================================================
# Per-participant cascade scoring
# ============================================================
def get_cascade_scores(liwc, prompts_curated, args, cache_file: Path) -> pd.DataFrame:
    """v3: prompts_curated is a dict[participant_id -> list of top-K
    personally-relevant messages, ordered by psych_score desc]."""
    if args.cached_llm:
        return pd.read_csv(args.cached_llm)
    if cache_file.exists():
        cached = pd.read_csv(cache_file)
        done_ids = set(cached["participant_id"].astype(str))
        print(f"  Resuming from cache: {len(cached)} done")
    else:
        cached = pd.DataFrame()
        done_ids = set()

    rows = []
    pids = liwc["participant_id"].astype(str).tolist()
    t0 = time.time()
    for i, pid in enumerate(pids):
        if pid in done_ids:
            continue
        sub = prompts_curated.get(pid, [])
        if len(sub) == 0:
            row = {"participant_id": pid, "n_msgs_used": 0,
                   **{k: np.nan for k in ITEMS.keys()},
                   **{f"{k}_cues": "" for k in ITEMS.keys()}}
            rows.append(row)
            print(f"  [{i+1}/{len(pids)}] {pid}: 0 candidate messages, scored NaN")
            continue
        # Already top-K by psych_score; do not re-sample
        sample = sub
        n = len(sample)
        block = build_messages_block(sample, args.max_chars_prompt)

        item_scores: dict[str, float] = {}
        item_cues: dict[str, str] = {}
        for item_key in ITEMS.keys():
            prompt = build_item_prompt(item_key, block)
            raw = call_ollama(prompt)
            parsed = parse_item_response(raw or "")
            if parsed is None:
                item_scores[item_key] = np.nan
                item_cues[f"{item_key}_cues"] = "PARSE_FAIL"
            else:
                item_scores[item_key] = parsed["score"]
                item_cues[f"{item_key}_cues"] = parsed["cues"]

        row = {"participant_id": pid, "n_msgs_used": n, **item_scores, **item_cues}
        rows.append(row)
        if (i + 1) % 5 == 0 or i == 0:
            avg = (time.time() - t0) / (i + 1)
            eta = avg * (len(pids) - i - 1) / 60
            print(f"  [{i+1}/{len(pids)}] {pid}: phq2_q1={item_scores.get('phq2_q1')}, q2={item_scores.get('phq2_q2')}, "
                  f"gad2_q1={item_scores.get('gad2_q1')}, q2={item_scores.get('gad2_q2')} (avg {avg:.1f}s/p, eta {eta:.1f}min)")
        if (i + 1) % 10 == 0:
            tmp = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
            tmp.to_csv(cache_file, index=False)

    full = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
    full.to_csv(cache_file, index=False)
    return full


# ============================================================
# Pipeline
# ============================================================
print(f"\n=== Step 1: per-item cascade ({args.ollama_model}) ===")
cache_file = OUT / "cache" / "llm_cascade_scores.csv"
llm_df = get_cascade_scores(liwc, prompts_curated, args, cache_file)

# Compute summed PHQ-2 and GAD-2 predictions
llm_df["PHQ2_pred"] = llm_df[["phq2_q1", "phq2_q2"]].sum(axis=1, min_count=1)
llm_df["GAD2_pred"] = llm_df[["gad2_q1", "gad2_q2"]].sum(axis=1, min_count=1)
llm_df["PHQ2_pred_pos"] = (llm_df["PHQ2_pred"] >= 3).astype(int)
llm_df["GAD2_pred_pos"] = (llm_df["GAD2_pred"] >= 3).astype(int)
llm_df.to_csv(OUT / "01_llm_cascade_scores.csv", index=False)

# Distribution check
print("\nLLM cascade output distribution:")
print(llm_df[["phq2_q1", "phq2_q2", "gad2_q1", "gad2_q2", "PHQ2_pred", "GAD2_pred"]].describe().round(2))


# Merge
liwc["participant_id"] = liwc["participant_id"].astype(str)
llm_df["participant_id"] = llm_df["participant_id"].astype(str)
merged = liwc.merge(
    llm_df[["participant_id", "phq2_q1", "phq2_q2", "gad2_q1", "gad2_q2",
            "PHQ2_pred", "GAD2_pred", "PHQ2_pred_pos", "GAD2_pred_pos"]],
    on="participant_id", how="left"
)
merged.to_csv(OUT / "00_merged_with_llm.csv", index=False)


# Step 2: raw correlations
print("\n=== Step 2: LLM-cascade correlations with PHQ-4 self-report ===")
corr_rows = []
for llm_col, phq_col, label in [
    ("phq2_q1", "PHQ4_Depression", "phq2_q1 (anhedonia) -> PHQ-2 dep sum"),
    ("phq2_q2", "PHQ4_Depression", "phq2_q2 (low mood) -> PHQ-2 dep sum"),
    ("PHQ2_pred", "PHQ4_Depression", "**PHQ2_pred -> PHQ-2 dep sum**"),
    ("PHQ2_pred", "PHQ4_Anxiety",    "PHQ2_pred -> GAD-2 anx sum (cross)"),
    ("gad2_q1", "PHQ4_Anxiety", "gad2_q1 (nervous) -> GAD-2 anx sum"),
    ("gad2_q2", "PHQ4_Anxiety", "gad2_q2 (worry) -> GAD-2 anx sum"),
    ("GAD2_pred", "PHQ4_Anxiety",    "**GAD2_pred -> GAD-2 anx sum**"),
    ("GAD2_pred", "PHQ4_Depression", "GAD2_pred -> PHQ-2 dep sum (cross)"),
]:
    sub = merged.dropna(subset=[llm_col, phq_col])
    if len(sub) < 5:
        continue
    rp, pp = stats.pearsonr(sub[llm_col], sub[phq_col])
    rs, ps = stats.spearmanr(sub[llm_col], sub[phq_col])
    print(f"  {label}: pearson r={rp:.3f} (p={pp:.4f}), spearman r={rs:.3f} (p={ps:.4f}), N={len(sub)}")
    corr_rows.append({"Comparison": label, "N": len(sub),
                      "Pearson_r": rp, "Pearson_p": pp,
                      "Spearman_r": rs, "Spearman_p": ps})
pd.DataFrame(corr_rows).to_csv(OUT / "02_llm_cascade_correlations.csv", index=False)


# Step 3: Classifiers (LIWC vs LLM-cascade vs combined)
LIWC_FEATS = [c for c in liwc.columns if c not in [
    "Prolific_ID", "PHQ4_Anxiety", "PHQ4_Depression", "PROMIS_Anxiety",
    "participant_id", "PHQ4_Total", "PHQ2_Dep_Pos", "GAD2_Anx_Pos",
    "PHQ2_Dep_Sum", "GAD2_Anx_Sum",
    "Age", "Gender_numeric", "total_words", "n_messages", "log_n_messages",
]]


def cv_predict(X, y, model_factory, n_folds=5, n_repeats=5, seed=42):
    probs_acc = np.zeros((len(y), n_repeats))
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        probs = np.full(len(y), np.nan)
        for tr, te in skf.split(X, y):
            mdl = model_factory()
            mdl.fit(X[tr], y[tr])
            probs[te] = mdl.predict_proba(X[te])[:, 1]
        probs_acc[:, rep] = probs
    return probs_acc.mean(axis=1)


def auc_with_bootstrap_ci(y, probs, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    auc = roc_auc_score(y, probs)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], probs[idx]))
    return auc, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def topk_recall(y, probs, k_frac=0.30):
    k = max(int(np.ceil(len(y) * k_frac)), 1)
    order = np.argsort(probs)[::-1]
    return y[order[:k]].sum() / y.sum() if y.sum() > 0 else np.nan


print(f"\n=== Step 3: Cross-validated classifiers ===")
results = []
for outcome_col, outcome_label, llm_score_col in [
    ("PHQ2_Dep_Pos", "Depression", "PHQ2_pred"),
    ("GAD2_Anx_Pos", "Anxiety", "GAD2_pred"),
]:
    sub = merged.copy()
    y = sub[outcome_col].values.astype(int)
    n_pos = int(y.sum())
    n = len(y)
    print(f"\n  --- {outcome_label}: positives={n_pos}/{n} (base rate {n_pos/n:.2f}) ---")

    # (A) LIWC L2
    X_liwc = StandardScaler().fit_transform(sub[LIWC_FEATS].fillna(0).values)
    probs_liwc = cv_predict(X_liwc, y,
        lambda: LogisticRegressionCV(Cs=10, cv=3, penalty="l2", solver="lbfgs",
                                     class_weight="balanced", max_iter=2000),
        n_folds=args.cv_folds, n_repeats=args.cv_repeats, seed=args.seed)
    auc_a, lo_a, hi_a = auc_with_bootstrap_ci(y, probs_liwc, n_boot=args.n_boot_auc)
    rec_a = topk_recall(y, probs_liwc, 0.30)
    print(f"    (A) LIWC L2:        AUC={auc_a:.3f} [{lo_a:.3f}, {hi_a:.3f}], top30%-recall={rec_a:.3f}")
    results.append({"Outcome": outcome_label, "Model": "(A) LIWC L2",
                    "AUC": auc_a, "AUC_CI_low": lo_a, "AUC_CI_high": hi_a,
                    "Top30_recall": rec_a, "n": n, "n_pos": n_pos})

    # (B) LLM cascade alone (sum of 2 items as continuous score)
    llm_x = sub[llm_score_col].values.astype(float)
    valid = ~np.isnan(llm_x)
    if valid.sum() >= 20:
        # min-max scale
        x = llm_x[valid]
        probs_b = (x - x.min()) / max(1e-9, x.max() - x.min())
        y_v = y[valid]
        auc_b, lo_b, hi_b = auc_with_bootstrap_ci(y_v, probs_b, n_boot=args.n_boot_auc)
        rec_b = topk_recall(y_v, probs_b, 0.30)
        print(f"    (B) LLM cascade:    AUC={auc_b:.3f} [{lo_b:.3f}, {hi_b:.3f}], top30%-recall={rec_b:.3f}, N={int(valid.sum())}")
        results.append({"Outcome": outcome_label, "Model": "(B) LLM cascade",
                        "AUC": auc_b, "AUC_CI_low": lo_b, "AUC_CI_high": hi_b,
                        "Top30_recall": rec_b, "n": int(valid.sum()), "n_pos": int(y_v.sum())})

        # (C) Combined LIWC + LLM
        X_combined = np.column_stack([X_liwc[valid], llm_x[valid].reshape(-1, 1)])
        X_combined = StandardScaler().fit_transform(X_combined)
        probs_c = cv_predict(X_combined, y_v,
            lambda: LogisticRegressionCV(Cs=10, cv=3, penalty="l2", solver="lbfgs",
                                         class_weight="balanced", max_iter=2000),
            n_folds=args.cv_folds, n_repeats=args.cv_repeats, seed=args.seed)
        auc_c, lo_c, hi_c = auc_with_bootstrap_ci(y_v, probs_c, n_boot=args.n_boot_auc)
        rec_c = topk_recall(y_v, probs_c, 0.30)
        print(f"    (C) LIWC + LLM:     AUC={auc_c:.3f} [{lo_c:.3f}, {hi_c:.3f}], top30%-recall={rec_c:.3f}")
        results.append({"Outcome": outcome_label, "Model": "(C) LIWC + LLM",
                        "AUC": auc_c, "AUC_CI_low": lo_c, "AUC_CI_high": hi_c,
                        "Top30_recall": rec_c, "n": int(valid.sum()), "n_pos": int(y_v.sum())})

        # ROC plot
        fig, ax = plt.subplots(figsize=(6, 6))
        for tag, yy, pp, auc_ in [("(A) LIWC L2", y, probs_liwc, auc_a),
                                  ("(B) LLM cascade", y_v, probs_b, auc_b),
                                  ("(C) Combined", y_v, probs_c, auc_c)]:
            fpr, tpr, _ = roc_curve(yy, pp)
            ax.plot(fpr, tpr, label=f"{tag}  AUC={auc_:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC: predicting {outcome_label}+ ({outcome_col})\nN={n}, positives={n_pos}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT / "figures" / f"roc_{outcome_label.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close()

pd.DataFrame(results).to_csv(OUT / "03_classifier_aucs.csv", index=False)


# Manifest
manifest = {
    "timestamp": TS,
    "args": vars(args),
    "method": "per-item PHQ-2/GAD-2 anchored cascade (LlaMADRS-style)",
    "cited": ["Galatzer-Levy et al. 2025 LlaMADRS",
              "Sadeghi et al. 2025 Cognitive-Mental-LLM",
              "Yang et al. 2024 MentaLLaMA",
              "Xu et al. 2024 Mental-LLM",
              "Parapar et al. 2024-2025 eRisk@CLEF"],
    "N_total": int(len(merged)),
    "Dep_pos_N": int(merged["PHQ2_Dep_Pos"].sum()),
    "Anx_pos_N": int(merged["GAD2_Anx_Pos"].sum()),
    "files": [
        "00_merged_with_llm.csv",
        "01_llm_cascade_scores.csv",
        "02_llm_cascade_correlations.csv",
        "03_classifier_aucs.csv",
        "figures/roc_depression.png",
        "figures/roc_anxiety.png",
        "cache/llm_cascade_scores.csv",
    ],
}
with open(OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)

print(f"\nDone. Output: {OUT}")
