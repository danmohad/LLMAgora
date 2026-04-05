"""Reusable aggregation helpers for the tidy survey DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .utils import _filter_df


# ── Per-model, per-question ──────────────────────────────────────────
def by_model_question(
    df: pd.DataFrame,
    scenario: str | None = None,
    *,
    models: list[str] | None = None,
    roles: list[str] | None = None,
    metric: str = "gap_signed",
) -> pd.DataFrame:
    """Mean *metric* per (model, question_number), averaged over turns and roles."""
    sub = _filter_df(df, scenario=scenario, models=models, roles=roles)
    return (
        sub.groupby(["model", "question_number", "question_family"], observed=True)[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: f"mean_{metric}"})
        .sort_values(["model", "question_number"])
    )


# ── Per-model, per-family ────────────────────────────────────────────
def by_model_family(
    df: pd.DataFrame,
    scenario: str | None = None,
    *,
    models: list[str] | None = None,
    roles: list[str] | None = None,
    metric: str = "gap_signed",
) -> pd.DataFrame:
    """Mean *metric* per (model, question_family)."""
    sub = _filter_df(df, scenario=scenario, models=models, roles=roles)
    return (
        sub.groupby(["model", "question_family"], observed=True)[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: f"mean_{metric}"})
    )


# ── Per-turn, per-family (for dynamics) ──────────────────────────────
def by_turn_family(
    df: pd.DataFrame,
    scenario: str | None = None,
    *,
    models: list[str] | None = None,
    roles: list[str] | None = None,
    metric: str = "gap_signed",
) -> pd.DataFrame:
    """Mean *metric* per (model, turn, question_family)."""
    sub = _filter_df(df, scenario=scenario, models=models, roles=roles)
    return (
        sub.groupby(["model", "turn", "question_family"], observed=True)[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: f"mean_{metric}"})
    )


# ── Role asymmetry ───────────────────────────────────────────────────
def role_asymmetry(
    df: pd.DataFrame,
    scenario: str | None = None,
    *,
    models: list[str] | None = None,
    metric: str = "gap_signed",
) -> pd.DataFrame:
    """Mean *metric* per (model, participant_role, question_family)."""
    sub = _filter_df(df, scenario=scenario, models=models)
    agg = (
        sub.groupby(["model", "participant_role", "question_family"], observed=True)[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: f"mean_{metric}"})
    )
    return agg


# ── Cross-scenario summary ───────────────────────────────────────────
def cross_scenario(
    df: pd.DataFrame,
    model: str | None = None,
    *,
    models: list[str] | None = None,
    metric: str = "gap_signed",
) -> pd.DataFrame:
    """Mean *metric* per (scenario, model, question_family)."""
    _models = [model] if model and models is None else models
    sub = _filter_df(df, models=_models)
    return (
        sub.groupby(["scenario", "model", "question_family"], observed=True)[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: f"mean_{metric}"})
    )


# ── Top-K questions by divergence ────────────────────────────────────
def top_questions(
    df: pd.DataFrame,
    scenario: str | None = None,
    *,
    k: int = 5,
    metric: str = "gap_abs",
) -> list[int]:
    """Return the *k* question numbers with highest mean *metric*."""
    sub = _filter_df(df, scenario=scenario)
    ranked = sub.groupby("question_number")[metric].mean().sort_values(ascending=False)
    return ranked.head(k).index.tolist()


# ── Peak / persistence / first-onset ─────────────────────────────────
def peak_persistence(
    df: pd.DataFrame,
    scenario: str | None = None,
    *,
    models: list[str] | None = None,
    threshold: float = config.DIVERGENCE_THRESHOLD,
) -> pd.DataFrame:
    """Compute peak_gap_abs, persistence, and first_above_threshold per group."""
    sub = _filter_df(df, scenario=scenario, models=models)
    groups = sub.groupby(["model", "question_family", "participant_role"], observed=True)

    rows = []
    for (mdl, fam, role), g in groups:
        vals = g["gap_abs"].values
        rows.append(
            {
                "model": mdl,
                "question_family": fam,
                "participant_role": role,
                "peak_gap_abs": float(np.nanmax(vals)) if len(vals) else np.nan,
                "mean_gap_abs": float(np.nanmean(vals)) if len(vals) else np.nan,
                "persistence": float((vals >= threshold).sum() / max(len(vals), 1)),
                "n_above_threshold": int((vals >= threshold).sum()),
                "first_above_threshold": int(
                    g.loc[g["gap_abs"] >= threshold, "turn"].min()
                )
                if (vals >= threshold).any()
                else np.nan,
                "proportion_nonzero": float((vals != 0).sum() / max(len(vals), 1)),
            }
        )
    return pd.DataFrame(rows)


# ── Comprehensive summary statistics ─────────────────────────────────
def compute_summary_stats(
    df: pd.DataFrame,
    scenario: str | None = None,
    *,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Wide summary table: one row per (model, question_family)."""
    sub = _filter_df(df, scenario=scenario, models=models)
    groups = sub.groupby(["model", "question_family"], observed=True)

    def _agg(g: pd.DataFrame) -> pd.Series:
        gs = g["gap_signed"]
        ga = g["gap_abs"]
        alpha_ga = g.loc[g["participant_role"] == "alpha", "gap_abs"].mean()
        beta_ga = g.loc[g["participant_role"] == "beta", "gap_abs"].mean()
        return pd.Series(
            {
                "mean_signed": gs.mean(),
                "mean_abs": ga.mean(),
                "median_abs": ga.median(),
                "std_signed": gs.std(),
                "peak_abs": ga.max(),
                "cumulative_abs": ga.sum(),
                "proportion_nonzero": (ga != 0).mean(),
                "alpha_beta_asymmetry": abs(alpha_ga - beta_ga)
                if not (np.isnan(alpha_ga) or np.isnan(beta_ga))
                else np.nan,
            }
        )

    return groups.apply(_agg, include_groups=False).reset_index()
