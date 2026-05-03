#!/usr/bin/env python3
"""RQ3 Detection probe: how detectable is PHQ-4 depression / anxiety status
from a participant's everyday ChatGPT prompts?

Three classifiers compared (5-fold CV, repeated 5x for stability):
  (A) LIWC baseline: L2-logistic on 46 LIWC features
  (B) LLM zero-shot judge: Qwen3:8B reads up to N sampled messages per
      participant and rates depression/anxiety on 0-100 each. Score is
      a single continuous predictor.
  (C) Combined: LIWC + LLM stacked.

Outcomes:
  - PHQ-2 Depression+ (>=3)
  - GAD-2 Anxiety+ (>=3)
  - Continuous PHQ-4 Depression / Anxiety (Spearman r between predicted prob
    and raw score; for LLM judge: direct correlation between rating and score)

Reports AUC (with bootstrap 95% CI), calibration, and class-balanced top-k recall.
Frames as a LOWER BOUND on the implicit-disclosure channel.

Cost: N=104 LLM calls at ~10-30s each => ~30-50 min on local Qwen3:8B.
"""

from __future__ import annotations

import argparse
import json
import os
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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

BASE_DIR = Path(__file__).parent.parent
LIWC_MERGED = BASE_DIR / "results" / "260419T1358_phq4_liwc_analysis" / "00_merged_mh_liwc_data.csv"
PROMPTS_FILE = BASE_DIR / "results" / "251229T1212_NRC_VAD_analysis" / "filtered_user_prompts_lt20words.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--liwc-merged", default=str(LIWC_MERGED))
ap.add_argument("--prompts", default=str(PROMPTS_FILE))
ap.add_argument("--ollama-model", default="qwen3:8b")
ap.add_argument("--ollama-url", default="http://localhost:11434")
ap.add_argument("--n-msgs-per-participant", type=int, default=50)
ap.add_argument("--min-words-per-msg", type=int, default=5)
ap.add_argument("--max-chars-prompt", type=int, default=12000)
ap.add_argument("--cv-folds", type=int, default=5)
ap.add_argument("--cv-repeats", type=int, default=5)
ap.add_argument("--n-boot-auc", type=int, default=2000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--skip-llm", action="store_true",
                help="Skip LLM judge (run baseline only).")
ap.add_argument("--cached-llm", default="",
                help="Path to cached LLM scores CSV; skip LLM call if provided.")
ap.add_argument("--out-suffix", default="phq4_detection_probe")
args = ap.parse_args()

TS = datetime.now().strftime("%y%m%dT%H%M")
OUT = BASE_DIR / "results" / f"{TS}_{args.out_suffix}"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "cache").mkdir(parents=True, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)


# ============================================================
# 1. Load data
# ============================================================
print(f"Loading LIWC: {args.liwc_merged}")
liwc = pd.read_csv(args.liwc_merged)
liwc["PHQ4_Total"] = liwc["PHQ4_Anxiety"] + liwc["PHQ4_Depression"]
liwc["PHQ2_Dep_Pos"] = (liwc["PHQ4_Depression"] >= 3).astype(int)
liwc["GAD2_Anx_Pos"] = (liwc["PHQ4_Anxiety"] >= 3).astype(int)
print(f"  N participants = {len(liwc)}")
print(f"  Dep+ N = {liwc['PHQ2_Dep_Pos'].sum()}, Anx+ N = {liwc['GAD2_Anx_Pos'].sum()}")

print(f"Loading prompts: {args.prompts}")
prompts = pd.read_csv(args.prompts, low_memory=False)
prompts = prompts[prompts["user_or_ai"] == "user"].copy()
prompts["message"] = prompts["message"].astype(str)
prompts = prompts[prompts["word_count"] >= args.min_words_per_msg]
print(f"  Filtered prompts (>= {args.min_words_per_msg} words): {len(prompts)}")
print(f"  participants represented: {prompts['participant_id'].nunique()}")


# ============================================================
# 2. LIWC features for baseline
# ============================================================
LIWC_FEATS = [c for c in liwc.columns if c not in [
    "Prolific_ID", "PHQ4_Anxiety", "PHQ4_Depression", "PROMIS_Anxiety",
    "participant_id", "PHQ4_Total", "PHQ2_Dep_Pos", "GAD2_Anx_Pos",
    "Age", "Gender_numeric", "total_words", "n_messages", "log_n_messages",
]]
print(f"\nLIWC features used as predictors: {len(LIWC_FEATS)}")
print(f"  (excluded ID/label/control columns)")
print(f"  feature list: {LIWC_FEATS}")


# ============================================================
# 3. LLM judge via Ollama
# ============================================================

JUDGE_PROMPT = """You are a psycholinguistic researcher studying everyday ChatGPT prompts.

This participant is one of 158 young adults aged 18-25. Across the whole sample, ~38% screen positive on PHQ-2 depression and ~49% on GAD-2 anxiety -- mental-health symptoms are common in this population, not rare.

Your task: estimate this specific participant's relative risk *compared to the sample base rate*, on a 0-100 scale where:
- 50 = roughly average for THIS sample (still moderate baseline distress)
- 0-25 = clearly less symptomatic than typical (very task-focused, no relational/affective signal)
- 25-50 = below or at average
- 50-75 = above average, some cues
- 75-100 = clearly more symptomatic than typical, multiple coherent cues

Use SUBTLE psycholinguistic cues, not just explicit symptom mentions:
- DEPRESSION: relational rumination, social comparison, hopeless or pessimistic future framing, low-energy references, anhedonia ("nothing is fun"), self/we/they references in distressed contexts, processing loss/loneliness.
- ANXIETY: anticipatory uncertainty, hypothetical worst-cases, somatic anxiety ("nervous", "stress", "panic", "racing heart"), reassurance-seeking, "what if" framing.

CRITICAL: distribute scores across the 0-100 range. DO NOT cluster at 0 or 50. If prompts seem completely neutral and task-focused (homework, code, recipes), score 25-40. If clearly elevated, score 65-90. Only score 0-15 if the prompts are *unusually* task-focused and emotionally flat compared to typical young adults.

Messages from one participant (sampled across many conversations):
=====
{messages_block}
=====

Respond ONLY with a single JSON object, no prose, no markdown:
{"depression_score": <int 0-100>, "anxiety_score": <int 0-100>, "evidence": "<one short sentence describing the strongest cues observed, or 'mostly task-focused'>"}"""


def call_ollama(prompt: str, model: str, url: str, timeout=300) -> str | None:
    """Call Ollama generate API; return raw response string or None on failure."""
    try:
        r = requests.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": False,                # disable Qwen3 thinking trace
                "options": {"temperature": 0, "num_ctx": 16384},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        print(f"    Ollama call failed: {e}")
        return None


def parse_judge(raw: str) -> dict | None:
    """Extract first JSON object from raw LLM output; tolerate extra text."""
    if not raw:
        return None
    raw = raw.strip()
    # Strip <think>...</think> blocks if any leaked through
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # Find first {...} block (greedy not needed; use balance)
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i in range(start, len(raw)):
        if raw[i] == "{": depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        d = json.loads(raw[start:end])
        return {
            "depression_score": int(d.get("depression_score", -1)),
            "anxiety_score": int(d.get("anxiety_score", -1)),
            "evidence": str(d.get("evidence", ""))[:300],
        }
    except Exception:
        return None


def build_messages_block(msgs: list[str], max_chars: int) -> str:
    """Concatenate sampled messages with separators, truncate to max_chars."""
    out = []
    used = 0
    for m in msgs:
        line = f"- {m.strip()}"
        if used + len(line) + 2 > max_chars:
            break
        out.append(line)
        used += len(line) + 2
    return "\n".join(out)


def get_llm_scores(liwc, prompts, args, cache_file: Path) -> pd.DataFrame:
    """Per-participant Qwen scoring; results cached to CSV (resumable)."""
    if args.cached_llm:
        print(f"  Loading cached LLM scores from {args.cached_llm}")
        return pd.read_csv(args.cached_llm)

    if cache_file.exists():
        cached = pd.read_csv(cache_file)
        print(f"  Resuming from cache ({len(cached)} done)")
        done_ids = set(cached["participant_id"].astype(str))
    else:
        cached = pd.DataFrame()
        done_ids = set()

    rows = []
    pids = liwc["participant_id"].astype(str).tolist()
    rng = random.Random(args.seed)
    t0 = time.time()
    for i, pid in enumerate(pids):
        if pid in done_ids:
            continue
        sub = prompts[prompts["participant_id"].astype(str) == pid]["message"].tolist()
        if len(sub) == 0:
            print(f"  [{i+1}/{len(pids)}] {pid}: no messages, skipping")
            continue
        n = min(len(sub), args.n_msgs_per_participant)
        sample = rng.sample(sub, n)
        block = build_messages_block(sample, args.max_chars_prompt)
        prompt = JUDGE_PROMPT.replace("{messages_block}", block)
        t_start = time.time()
        raw = call_ollama(prompt, args.ollama_model, args.ollama_url)
        elapsed = time.time() - t_start
        parsed = parse_judge(raw or "")
        if parsed is None:
            print(f"  [{i+1}/{len(pids)}] {pid}: parse failed (raw len={len(raw or '')})")
            parsed = {"depression_score": np.nan, "anxiety_score": np.nan, "evidence": "PARSE_FAIL"}
        row = {
            "participant_id": pid,
            "n_msgs_used": n,
            "elapsed_sec": round(elapsed, 1),
            **parsed,
        }
        rows.append(row)
        if (i + 1) % 5 == 0 or i == 0:
            avg = (time.time() - t0) / (i + 1)
            eta = avg * (len(pids) - i - 1) / 60
            print(f"  [{i+1}/{len(pids)}] {pid}: dep={parsed['depression_score']}, anx={parsed['anxiety_score']}, ev={parsed['evidence'][:50]} ({elapsed:.1f}s, eta {eta:.1f}min)")
        # incremental save
        if (i + 1) % 10 == 0:
            tmp = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
            tmp.to_csv(cache_file, index=False)
    full = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
    full.to_csv(cache_file, index=False)
    return full


# ============================================================
# 4. Classifiers
# ============================================================

def cv_predict(X, y, model_factory, n_folds=5, n_repeats=5, seed=42):
    """Repeated stratified k-fold predicted probabilities (mean over repeats)."""
    rng = np.random.RandomState(seed)
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
    n = len(y)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], probs[idx]))
    boots = np.array(boots)
    return auc, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def youden_threshold(y, probs):
    fpr, tpr, thr = roc_curve(y, probs)
    j = tpr - fpr
    k = np.argmax(j)
    return thr[k], tpr[k], fpr[k]


def topk_recall(y, probs, k_frac=0.30):
    n = len(y)
    k = max(int(np.ceil(n * k_frac)), 1)
    order = np.argsort(probs)[::-1]
    top = order[:k]
    return y[top].sum() / y.sum() if y.sum() > 0 else np.nan


# ============================================================
# Main pipeline
# ============================================================

# Step 1: LLM judge (or load cache)
llm_cache = OUT / "cache" / "llm_judge_scores.csv"
existing_cache = sorted(BASE_DIR.glob("results/*_phq4_detection_probe/cache/llm_judge_scores.csv"))
if existing_cache and not args.cached_llm:
    print(f"\nFound prior LLM cache: {existing_cache[-1]}")

if args.skip_llm:
    print("\n[Skipping LLM judge per --skip-llm]")
    llm_df = pd.DataFrame({"participant_id": liwc["participant_id"].astype(str),
                           "depression_score": np.nan,
                           "anxiety_score": np.nan,
                           "evidence": ""})
else:
    print(f"\n=== Step 1: LLM judge ({args.ollama_model}) ===")
    llm_df = get_llm_scores(liwc, prompts, args, llm_cache)

# Merge LLM scores onto LIWC frame
liwc["participant_id"] = liwc["participant_id"].astype(str)
llm_df["participant_id"] = llm_df["participant_id"].astype(str)
merged = liwc.merge(llm_df[["participant_id", "depression_score", "anxiety_score", "evidence"]],
                    on="participant_id", how="left")
merged.to_csv(OUT / "00_merged_with_llm_scores.csv", index=False)


# Step 2: Compute correlations between raw LLM scores and PHQ-4 scores
print(f"\n=== Step 2: LLM-judge raw correlations ===")
corr_rows = []
for llm_col, phq_col, label in [
    ("depression_score", "PHQ4_Depression", "LLM dep -> PHQ-4 dep"),
    ("depression_score", "PHQ4_Anxiety",    "LLM dep -> PHQ-4 anx"),
    ("anxiety_score",    "PHQ4_Anxiety",    "LLM anx -> PHQ-4 anx"),
    ("anxiety_score",    "PHQ4_Depression", "LLM anx -> PHQ-4 dep"),
]:
    sub = merged.dropna(subset=[llm_col, phq_col])
    if len(sub) < 5:
        continue
    rp, pp = stats.pearsonr(sub[llm_col], sub[phq_col])
    rs, ps = stats.spearmanr(sub[llm_col], sub[phq_col])
    print(f"  {label}: pearson r={rp:.3f} (p={pp:.4f}), spearman r={rs:.3f} (p={ps:.4f}), N={len(sub)}")
    corr_rows.append({
        "Comparison": label, "N": len(sub),
        "Pearson_r": rp, "Pearson_p": pp,
        "Spearman_r": rs, "Spearman_p": ps,
    })
pd.DataFrame(corr_rows).to_csv(OUT / "01_llm_score_correlations.csv", index=False)


# Step 3: Classifiers
print(f"\n=== Step 3: Cross-validated classifiers ===")
results = []
for outcome_col, outcome_label in [("PHQ2_Dep_Pos", "Depression"), ("GAD2_Anx_Pos", "Anxiety")]:
    sub = merged.copy()
    y = sub[outcome_col].values.astype(int)
    n_pos = int(y.sum())
    n = len(y)
    if n_pos < 10 or (n - n_pos) < 10:
        print(f"\n  {outcome_label}: insufficient class balance ({n_pos}/{n}), skipping")
        continue
    print(f"\n  --- {outcome_label}: positives={n_pos}/{n} (base rate {n_pos/n:.2f}) ---")

    # (A) LIWC baseline
    X_liwc = StandardScaler().fit_transform(sub[LIWC_FEATS].fillna(0).values)
    probs_liwc = cv_predict(X_liwc, y,
        lambda: LogisticRegressionCV(Cs=10, cv=3, penalty="l2", solver="lbfgs",
                                     class_weight="balanced", max_iter=2000),
        n_folds=args.cv_folds, n_repeats=args.cv_repeats, seed=args.seed)
    auc_a, lo_a, hi_a = auc_with_bootstrap_ci(y, probs_liwc, n_boot=args.n_boot_auc)
    thr_a, tpr_a, fpr_a = youden_threshold(y, probs_liwc)
    rec_a = topk_recall(y, probs_liwc, 0.30)
    print(f"    (A) LIWC L2:        AUC={auc_a:.3f} [{lo_a:.3f}, {hi_a:.3f}], top30%-recall={rec_a:.3f}")
    results.append({"Outcome": outcome_label, "Model": "(A) LIWC L2", "AUC": auc_a, "AUC_CI_low": lo_a, "AUC_CI_high": hi_a, "Top30_recall": rec_a, "n": n, "n_pos": n_pos})

    # (B) LLM judge alone (no CV needed; it's a single zero-shot per participant)
    llm_score_col = "depression_score" if outcome_col == "PHQ2_Dep_Pos" else "anxiety_score"
    llm_x = sub[llm_score_col].values.astype(float)
    valid = ~np.isnan(llm_x)
    if valid.sum() >= 20:
        # Scale to 0-1 for direct AUC
        probs_b = (llm_x[valid] - np.nanmin(llm_x[valid])) / max(1e-9, (np.nanmax(llm_x[valid]) - np.nanmin(llm_x[valid])))
        y_v = y[valid]
        auc_b, lo_b, hi_b = auc_with_bootstrap_ci(y_v, probs_b, n_boot=args.n_boot_auc)
        thr_b, tpr_b, fpr_b = youden_threshold(y_v, probs_b)
        rec_b = topk_recall(y_v, probs_b, 0.30)
        print(f"    (B) LLM judge:      AUC={auc_b:.3f} [{lo_b:.3f}, {hi_b:.3f}], top30%-recall={rec_b:.3f}, N_with_score={int(valid.sum())}")
        results.append({"Outcome": outcome_label, "Model": "(B) LLM judge",
                        "AUC": auc_b, "AUC_CI_low": lo_b, "AUC_CI_high": hi_b,
                        "Top30_recall": rec_b, "n": int(valid.sum()), "n_pos": int(y_v.sum())})

        # (C) Combined LIWC + LLM (CV; only on participants with LLM score)
        X_combined = np.column_stack([X_liwc[valid], llm_x[valid].reshape(-1, 1)])
        X_combined = StandardScaler().fit_transform(X_combined)
        probs_c = cv_predict(X_combined, y_v,
            lambda: LogisticRegressionCV(Cs=10, cv=3, penalty="l2", solver="lbfgs",
                                         class_weight="balanced", max_iter=2000),
            n_folds=args.cv_folds, n_repeats=args.cv_repeats, seed=args.seed)
        auc_c, lo_c, hi_c = auc_with_bootstrap_ci(y_v, probs_c, n_boot=args.n_boot_auc)
        thr_c, tpr_c, fpr_c = youden_threshold(y_v, probs_c)
        rec_c = topk_recall(y_v, probs_c, 0.30)
        print(f"    (C) LIWC + LLM:     AUC={auc_c:.3f} [{lo_c:.3f}, {hi_c:.3f}], top30%-recall={rec_c:.3f}")
        results.append({"Outcome": outcome_label, "Model": "(C) LIWC + LLM",
                        "AUC": auc_c, "AUC_CI_low": lo_c, "AUC_CI_high": hi_c,
                        "Top30_recall": rec_c, "n": int(valid.sum()), "n_pos": int(y_v.sum())})
    else:
        print(f"    (B) LLM judge:      skipped, only {int(valid.sum())} valid scores")
        probs_b = None

    # ROC plot for this outcome
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y, probs_liwc)
    ax.plot(fpr, tpr, label=f"(A) LIWC L2  AUC={auc_a:.3f}")
    if probs_b is not None:
        fpr_, tpr_, _ = roc_curve(y_v, probs_b)
        ax.plot(fpr_, tpr_, label=f"(B) LLM judge  AUC={auc_b:.3f}")
        fpr_, tpr_, _ = roc_curve(y_v, probs_c)
        ax.plot(fpr_, tpr_, label=f"(C) Combined  AUC={auc_c:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC: predicting PHQ-2 / GAD-2 {outcome_label}+\n(N={n}, positives={n_pos})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "figures" / f"roc_{outcome_label.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()


pd.DataFrame(results).to_csv(OUT / "02_classifier_aucs.csv", index=False)


# ============================================================
# Manifest
# ============================================================
manifest = {
    "timestamp": TS,
    "args": vars(args),
    "N_total": int(len(merged)),
    "Dep_pos_N": int(merged["PHQ2_Dep_Pos"].sum()),
    "Anx_pos_N": int(merged["GAD2_Anx_Pos"].sum()),
    "LLM_scores_obtained": int(merged["depression_score"].notna().sum()),
    "LIWC_features_used": LIWC_FEATS,
    "files": [
        "00_merged_with_llm_scores.csv",
        "01_llm_score_correlations.csv",
        "02_classifier_aucs.csv",
        "figures/roc_depression.png",
        "figures/roc_anxiety.png",
        "cache/llm_judge_scores.csv",
    ],
}
with open(OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)

print(f"\nDone. Output: {OUT}")
