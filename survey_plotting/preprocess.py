"""Build a tidy (long) DataFrame from the aggregate pickle's survey columns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config


def build_tidy_survey_df(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    """Convert nested survey dicts in *aggregate_df* to a long DataFrame.

    Each row in the output represents one
    (scenario, model, incentive, question, role, turn) observation.
    """
    records: list[dict[str, Any]] = []

    for _, row in aggregate_df.iterrows():
        scenario = row["scenario_id"]
        model = row["model"]
        inc_dir = row.get("incentive_direction")
        inc_type = row.get("incentive_type")

        pub = row.get("survey-public") or {}
        priv = row.get("survey-private") or {}

        all_q_keys = sorted(
            {k for k in list(pub.keys()) + list(priv.keys()) if k.startswith("Q")},
            key=lambda k: int(k[1:]),
        )

        for q_key in all_q_keys:
            q_num = int(q_key[1:])
            pub_block = pub.get(q_key, {})
            priv_block = priv.get(q_key, {})
            q_family = (
                pub_block.get("question_group")
                or priv_block.get("question_group")
                or config.FAMILY_MAP.get(q_num, "unknown")
            )
            q_text = pub_block.get("question") or priv_block.get("question", "")

            for role in ("alpha", "beta"):
                pub_part = pub_block.get(role, {})
                priv_part = priv_block.get(role, {})

                turns = pub_part.get("debate_turn") or priv_part.get("debate_turn") or []
                pub_scores = pub_part.get("response_score") or []
                pub_ses = pub_part.get("standard_error") or []
                priv_scores = priv_part.get("response_score") or []
                priv_ses = priv_part.get("standard_error") or []

                for i, turn in enumerate(turns):
                    records.append(
                        {
                            "scenario": scenario,
                            "model": model,
                            "incentive_direction": inc_dir,
                            "incentive_type": inc_type,
                            "question_number": q_num,
                            "question_family": q_family,
                            "question_text": q_text,
                            "participant_role": role,
                            "turn": int(turn),
                            "public_score": pub_scores[i] if i < len(pub_scores) else np.nan,
                            "private_score": priv_scores[i] if i < len(priv_scores) else np.nan,
                            "public_se": pub_ses[i] if i < len(pub_ses) else np.nan,
                            "private_se": priv_ses[i] if i < len(priv_ses) else np.nan,
                        }
                    )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    _enrich(df)
    return (
        df.sort_values(["scenario", "model", "question_order", "participant_role", "turn"])
        .reset_index(drop=True)
    )


def _enrich(df: pd.DataFrame) -> None:
    """Add derived columns in-place."""
    df["gap_signed"] = df["public_score"] - df["private_score"]
    df["gap_abs"] = df["gap_signed"].abs()
    df["question_order"] = df["question_number"]
    df["family_order"] = df["question_family"].map(config.FAMILY_ORDER).fillna(0).astype(int)
    df["direction_label"] = np.where(
        df["gap_signed"] > 0,
        "positive",
        np.where(df["gap_signed"] < 0, "negative", "zero"),
    )


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of problems found in the tidy DataFrame (empty = OK)."""
    required = [
        "scenario",
        "model",
        "question_number",
        "question_family",
        "participant_role",
        "turn",
        "public_score",
        "private_score",
        "gap_signed",
        "gap_abs",
    ]
    problems: list[str] = []
    for col in required:
        if col not in df.columns:
            problems.append(f"Missing column: {col}")
    if not problems:
        n_missing = int(df[["public_score", "private_score"]].isna().sum().sum())
        if n_missing:
            problems.append(f"{n_missing} NaN values in public/private scores")
    return problems
