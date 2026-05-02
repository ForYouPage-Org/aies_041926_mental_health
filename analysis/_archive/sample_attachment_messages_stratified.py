#!/usr/bin/env python3
"""
Create participant-stratified message samples for exclusive attachment groups.

This script excludes participants who are simultaneously high in attachment
anxiety and avoidance, then samples up to N user messages per remaining
participant. Sampling is spread across conversations where possible so each
participant is represented without being dominated by a single thread.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR.parent
ECR_PATH = BASE_DIR / "data" / "ecr_scores_validated_clean.csv"
CONVERSATIONS_PATH = WORKSPACE_DIR / "data" / "conversations_full_20251007.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "260111T1239_attachment_conversations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample exclusive high-anxiety and high-avoidance messages."
    )
    parser.add_argument(
        "--anxiety-threshold",
        type=float,
        default=None,
        help="Explicit high-anxiety cutoff. Defaults to the 75th percentile.",
    )
    parser.add_argument(
        "--avoidance-threshold",
        type=float,
        default=None,
        help="Explicit high-avoidance cutoff. Defaults to the 75th percentile.",
    )
    parser.add_argument(
        "--min-avoidance-participants",
        type=int,
        default=0,
        help=(
            "If > 0 and no explicit avoidance threshold is supplied, lower the "
            "avoidance cutoff just enough to reach this many exclusive "
            "avoidance participants with user messages."
        ),
    )
    parser.add_argument(
        "--messages-per-participant",
        type=int,
        default=10,
        help="Maximum user messages to sample per participant.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where sampled CSV/JSON outputs will be written.",
    )
    return parser.parse_args()


def percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def load_ecr_scores() -> pd.DataFrame:
    ecr_df = pd.read_csv(ECR_PATH)
    ecr_df["Prolific_ID"] = ecr_df["Prolific_ID"].astype(str).str.strip()
    return ecr_df


def build_attachment_groups(
    ecr_df: pd.DataFrame,
    anxiety_threshold: float,
    avoidance_threshold: float,
) -> dict[str, set[str]]:
    high_anxiety_ids = set(
        ecr_df.loc[ecr_df["ECR_Anxiety"] > anxiety_threshold, "Prolific_ID"]
    )
    high_avoidance_ids = set(
        ecr_df.loc[ecr_df["ECR_Avoidance"] > avoidance_threshold, "Prolific_ID"]
    )
    overlap_ids = high_anxiety_ids & high_avoidance_ids

    groups = {
        "high_anxiety": high_anxiety_ids,
        "high_avoidance": high_avoidance_ids,
        "overlap": overlap_ids,
        "anxiety_only": high_anxiety_ids - high_avoidance_ids,
        "avoidance_only": high_avoidance_ids - high_anxiety_ids,
    }
    return groups


def find_participants_with_user_messages(target_ids: set[str]) -> set[str]:
    participants_with_messages: set[str] = set()

    for chunk in pd.read_csv(CONVERSATIONS_PATH, chunksize=100000, low_memory=False):
        chunk["participant_id"] = chunk["participant_id"].astype(str).str.strip()
        filtered = chunk[
            chunk["participant_id"].isin(target_ids) & (chunk["user_or_ai"] == "user")
        ].copy()
        if filtered.empty:
            continue

        filtered["message"] = filtered["message"].fillna("").astype(str)
        filtered = filtered[filtered["message"].str.strip() != ""]
        if filtered.empty:
            continue

        participants_with_messages.update(filtered["participant_id"].unique().tolist())

    return participants_with_messages


def choose_avoidance_threshold(
    ecr_df: pd.DataFrame,
    anxiety_threshold: float,
    min_avoidance_participants: int,
) -> tuple[float, dict[str, object]]:
    participants_with_messages = find_participants_with_user_messages(
        set(ecr_df["Prolific_ID"])
    )
    default_threshold = float(ecr_df["ECR_Avoidance"].quantile(0.75))
    high_anxiety_ids = set(
        ecr_df.loc[ecr_df["ECR_Anxiety"] > anxiety_threshold, "Prolific_ID"]
    )

    candidate_rows: list[dict[str, object]] = []
    for threshold in sorted(ecr_df["ECR_Avoidance"].dropna().unique(), reverse=True):
        high_avoidance_ids = set(
            ecr_df.loc[ecr_df["ECR_Avoidance"] > float(threshold), "Prolific_ID"]
        )
        overlap_ids = high_anxiety_ids & high_avoidance_ids
        avoidance_only_ids = high_avoidance_ids - high_anxiety_ids
        avoidance_only_with_messages = avoidance_only_ids & participants_with_messages
        candidate_rows.append(
            {
                "threshold": float(threshold),
                "high_avoidance_total": len(high_avoidance_ids),
                "avoidance_only_total": len(avoidance_only_ids),
                "avoidance_only_with_messages": len(avoidance_only_with_messages),
                "overlap_total": len(overlap_ids),
            }
        )

    for row in candidate_rows:
        if row["avoidance_only_with_messages"] >= min_avoidance_participants:
            return float(row["threshold"]), {
                "method": "highest_threshold_meeting_min_avoidance_participants",
                "requested_min_avoidance_participants": min_avoidance_participants,
                "default_avoidance_threshold": default_threshold,
                "selected_avoidance_threshold": float(row["threshold"]),
                "high_avoidance_total": int(row["high_avoidance_total"]),
                "avoidance_only_total": int(row["avoidance_only_total"]),
                "avoidance_only_with_messages": int(
                    row["avoidance_only_with_messages"]
                ),
                "overlap_total": int(row["overlap_total"]),
            }

    raise ValueError(
        "No avoidance threshold produced the requested minimum number of "
        "exclusive avoidance participants with user messages."
    )


def resolve_thresholds(
    ecr_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, object]]:
    anxiety_threshold = (
        float(args.anxiety_threshold)
        if args.anxiety_threshold is not None
        else float(ecr_df["ECR_Anxiety"].quantile(0.75))
    )

    if args.avoidance_threshold is not None:
        avoidance_threshold = float(args.avoidance_threshold)
        selection_metadata = {
            "method": "explicit_avoidance_threshold",
            "selected_avoidance_threshold": avoidance_threshold,
        }
    elif args.min_avoidance_participants > 0:
        avoidance_threshold, selection_metadata = choose_avoidance_threshold(
            ecr_df=ecr_df,
            anxiety_threshold=anxiety_threshold,
            min_avoidance_participants=args.min_avoidance_participants,
        )
    else:
        avoidance_threshold = float(ecr_df["ECR_Avoidance"].quantile(0.75))
        selection_metadata = {
            "method": "default_75th_percentile",
            "selected_avoidance_threshold": avoidance_threshold,
        }

    thresholds = {
        "anxiety_threshold": anxiety_threshold,
        "avoidance_threshold": avoidance_threshold,
    }
    return thresholds, selection_metadata


def load_user_messages(target_ids: set[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(CONVERSATIONS_PATH, chunksize=100000, low_memory=False):
        chunk["participant_id"] = chunk["participant_id"].astype(str).str.strip()
        filtered = chunk[
            chunk["participant_id"].isin(target_ids) & (chunk["user_or_ai"] == "user")
        ].copy()
        if filtered.empty:
            continue

        filtered["message"] = filtered["message"].fillna("").astype(str)
        filtered = filtered[filtered["message"].str.strip() != ""]
        if filtered.empty:
            continue

        chunks.append(filtered)

    if not chunks:
        raise RuntimeError("No user messages found for the requested participants.")

    messages_df = pd.concat(chunks, ignore_index=True)
    if "word_count" not in messages_df.columns:
        messages_df["word_count"] = messages_df["message"].str.split().str.len()
    if "char_count" not in messages_df.columns:
        messages_df["char_count"] = messages_df["message"].str.len()
    return messages_df


def sample_participant_rows(
    participant_df: pd.DataFrame,
    n_target: int,
    rng: np.random.Generator,
) -> pd.Index:
    if participant_df.empty or n_target <= 0:
        return participant_df.index[:0]

    if len(participant_df) <= n_target:
        return participant_df.index

    selected: list[int] = []
    leftovers: list[int] = []

    grouped = list(participant_df.groupby("conversation_id", sort=False))
    rng.shuffle(grouped)

    for _, conv_df in grouped:
        row_indices = conv_df.index.to_numpy()
        first_pick = int(rng.choice(row_indices))
        selected.append(first_pick)
        leftovers.extend(int(idx) for idx in row_indices if int(idx) != first_pick)

        if len(selected) == n_target:
            return pd.Index(selected)

    remaining = n_target - len(selected)
    if remaining > 0:
        extra = rng.choice(np.array(leftovers), size=remaining, replace=False)
        selected.extend(int(idx) for idx in extra.tolist())

    return pd.Index(selected)


def sample_group_messages(
    messages_df: pd.DataFrame,
    participant_ids: set[str],
    label: str,
    messages_per_participant: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_df = messages_df[messages_df["participant_id"].isin(participant_ids)].copy()
    sampled_frames: list[pd.DataFrame] = []
    participant_rows: list[dict[str, object]] = []

    for participant_id in sorted(participant_ids):
        participant_df = group_df[group_df["participant_id"] == participant_id].copy()
        n_available = len(participant_df)
        n_sample = min(messages_per_participant, n_available)

        if n_sample > 0:
            sampled_idx = sample_participant_rows(participant_df, n_sample, rng)
            sampled_df = participant_df.loc[sampled_idx].copy()
            sampled_df["attachment_group"] = label
            sampled_frames.append(sampled_df)
            n_sampled_conversations = sampled_df["conversation_id"].nunique()
        else:
            n_sampled_conversations = 0

        participant_rows.append(
            {
                "participant_id": participant_id,
                "group": label,
                "n_messages_available": n_available,
                "n_messages_sampled": n_sample,
                "n_conversations_available": participant_df["conversation_id"].nunique(),
                "n_conversations_sampled": n_sampled_conversations,
            }
        )

    if sampled_frames:
        sampled = pd.concat(sampled_frames, ignore_index=True)
    else:
        sampled = group_df.iloc[0:0].copy()
        sampled["attachment_group"] = pd.Series(dtype="object")

    sort_columns = [
        column
        for column in ["participant_id", "timestamp", "conversation_id", "message_id"]
        if column in sampled.columns
    ]
    if sort_columns:
        sampled = sampled.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    participant_summary = pd.DataFrame(participant_rows)
    return sampled, participant_summary


def build_group_summary(
    participant_summary: pd.DataFrame,
    sampled_df: pd.DataFrame,
    target_messages: int,
) -> dict[str, object]:
    available_mask = participant_summary["n_messages_available"] > 0
    sampled_mask = participant_summary["n_messages_sampled"] > 0

    return {
        "participants_total": int(len(participant_summary)),
        "participants_with_messages": int(available_mask.sum()),
        "participants_sampled": int(sampled_mask.sum()),
        "participants_below_target": int(
            (
                available_mask
                & (participant_summary["n_messages_available"] < target_messages)
            ).sum()
        ),
        "messages_available_total": int(participant_summary["n_messages_available"].sum()),
        "messages_sampled_total": int(participant_summary["n_messages_sampled"].sum()),
        "conversations_available_total": int(
            participant_summary["n_conversations_available"].sum()
        ),
        "conversations_sampled_total": int(sampled_df["conversation_id"].nunique()),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ecr_df = load_ecr_scores()
    thresholds, threshold_selection = resolve_thresholds(ecr_df, args)
    groups = build_attachment_groups(
        ecr_df=ecr_df,
        anxiety_threshold=thresholds["anxiety_threshold"],
        avoidance_threshold=thresholds["avoidance_threshold"],
    )
    target_ids = groups["anxiety_only"] | groups["avoidance_only"]
    messages_df = load_user_messages(target_ids)

    ecr_columns = ["Prolific_ID", "ECR_Anxiety", "ECR_Avoidance"]

    anxiety_sample, anxiety_participants = sample_group_messages(
        messages_df=messages_df,
        participant_ids=groups["anxiety_only"],
        label="High_Anxiety_Only",
        messages_per_participant=args.messages_per_participant,
        rng=np.random.default_rng(args.seed),
    )
    avoidance_sample, avoidance_participants = sample_group_messages(
        messages_df=messages_df,
        participant_ids=groups["avoidance_only"],
        label="High_Avoidance_Only",
        messages_per_participant=args.messages_per_participant,
        rng=np.random.default_rng(args.seed + 1),
    )

    anxiety_sample = anxiety_sample.merge(
        ecr_df[ecr_columns],
        left_on="participant_id",
        right_on="Prolific_ID",
        how="left",
    )
    avoidance_sample = avoidance_sample.merge(
        ecr_df[ecr_columns],
        left_on="participant_id",
        right_on="Prolific_ID",
        how="left",
    )

    participant_summary = pd.concat(
        [anxiety_participants, avoidance_participants], ignore_index=True
    ).merge(
        ecr_df[ecr_columns],
        left_on="participant_id",
        right_on="Prolific_ID",
        how="left",
    )

    combined = pd.concat([anxiety_sample, avoidance_sample], ignore_index=True)

    anxiety_path = output_dir / "high_anxiety_only_messages_stratified.csv"
    avoidance_path = output_dir / "high_avoidance_only_messages_stratified.csv"
    combined_path = output_dir / "all_sampled_messages_no_cooccurrence_stratified.csv"
    participant_path = output_dir / "participant_summary_no_cooccurrence_stratified.csv"
    summary_path = output_dir / "sampling_summary_no_cooccurrence_stratified.json"

    anxiety_sample.to_csv(anxiety_path, index=False)
    avoidance_sample.to_csv(avoidance_path, index=False)
    combined.to_csv(combined_path, index=False)
    participant_summary.to_csv(participant_path, index=False)

    anxiety_sampled_participants = set(anxiety_sample["participant_id"].astype(str))
    avoidance_sampled_participants = set(avoidance_sample["participant_id"].astype(str))
    overlap_sampled_participants = anxiety_sampled_participants & avoidance_sampled_participants

    summary = {
        "timestamp": datetime.now().isoformat(),
        "sampling_strategy": "exclusive_attachment_groups_participant_stratified",
        "random_seed": args.seed,
        "messages_per_participant": args.messages_per_participant,
        **thresholds,
        "threshold_selection": threshold_selection,
        "eligible_participant_overlap": {
            "high_anxiety": len(groups["high_anxiety"]),
            "high_avoidance": len(groups["high_avoidance"]),
            "both_high": len(groups["overlap"]),
            "union_high": len(groups["high_anxiety"] | groups["high_avoidance"]),
            "both_pct_of_high_anxiety": percent(
                len(groups["overlap"]), len(groups["high_anxiety"])
            ),
            "both_pct_of_high_avoidance": percent(
                len(groups["overlap"]), len(groups["high_avoidance"])
            ),
            "both_pct_of_union": percent(
                len(groups["overlap"]),
                len(groups["high_anxiety"] | groups["high_avoidance"]),
            ),
        },
        "exclusive_groups": {
            "anxiety_only_participants": len(groups["anxiety_only"]),
            "avoidance_only_participants": len(groups["avoidance_only"]),
        },
        "high_anxiety_only_sample": build_group_summary(
            anxiety_participants, anxiety_sample, args.messages_per_participant
        ),
        "high_avoidance_only_sample": build_group_summary(
            avoidance_participants, avoidance_sample, args.messages_per_participant
        ),
        "sample_overlap_check": {
            "participants_in_both_files": len(overlap_sampled_participants),
            "identical_conversation_ids_in_both_files": int(
                len(
                    set(anxiety_sample["conversation_id"].astype(str))
                    & set(avoidance_sample["conversation_id"].astype(str))
                )
            ),
        },
        "output_files": {
            "high_anxiety_only_messages": anxiety_path.name,
            "high_avoidance_only_messages": avoidance_path.name,
            "combined": combined_path.name,
            "participant_summary": participant_path.name,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
