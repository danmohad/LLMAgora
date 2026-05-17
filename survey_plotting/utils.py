"""Utility helpers for survey plotting and analysis."""

from __future__ import annotations

import pandas as pd


def _filter_df(
    df: pd.DataFrame,
    *,
    scenario: str | None = None,
    models: list[str] | None = None,
    roles: list[str] | None = None,
    families: list[str] | None = None,
) -> pd.DataFrame:
    """Apply standard survey DataFrame filters."""
    out = df
    if scenario is not None:
        out = out[out["scenario"] == scenario]
    if models is not None:
        out = out[out["model"].isin(models)]
    if roles is not None:
        out = out[out["participant_role"].isin(roles)]
    if families is not None:
        out = out[out["question_family"].isin(families)]
    return out
