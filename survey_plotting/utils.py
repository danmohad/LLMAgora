"""Utility helpers for the survey plotting module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def short_model(name: str) -> str:
    """Shorten ``'provider/model-name'`` to ``'model-name'``."""
    return name.split("/", 1)[-1] if "/" in name else name


def _filter_df(
    df: pd.DataFrame,
    *,
    scenario: str | None = None,
    models: list[str] | None = None,
    roles: list[str] | None = None,
    families: list[str] | None = None,
) -> pd.DataFrame:
    """Apply standard column filters and return a copy."""
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


def add_family_separators(
    ax: plt.Axes,
    positions: Sequence[float] | None = None,
    orientation: str = "vertical",
    **kwargs: Any,
) -> None:
    """Draw dashed lines between question families on *ax*."""
    if positions is None:
        positions = config.FAMILY_BOUNDARIES_AFTER_Q
    style: dict[str, Any] = {"color": "0.3", "linewidth": 0.8, "linestyle": "--"}
    style.update(kwargs)
    draw = ax.axvline if orientation == "vertical" else ax.axhline
    for pos in positions:
        draw(pos, **style)


def save_figure(
    fig: plt.Figure,
    path: str | Path | None,
    dpi: int = config.DEFAULT_DPI,
) -> None:
    """Save *fig* to *path*, creating parent directories as needed."""
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")


def ensure_output_dirs(base: str | Path = "outputs/plots") -> dict[str, Path]:
    """Create the recommended output folder tree and return path mapping."""
    base = Path(base)
    dirs = {
        "by_scenario": base / "by_scenario",
        "by_model": base / "by_model",
        "appendix": base / "appendix",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def annotate_top_n(
    ax: plt.Axes,
    x_positions: Sequence[float],
    y_values: Sequence[float],
    labels: Sequence[str],
    n: int = 3,
    fontsize: int = 7,
) -> None:
    """Label the top *n* points by absolute magnitude."""
    arr = np.asarray(y_values, dtype=float)
    xp = np.asarray(x_positions, dtype=float)
    top_idx = np.argsort(np.abs(arr))[-n:]
    for idx in top_idx:
        ax.annotate(
            labels[idx],
            xy=(xp[idx], arr[idx]),
            fontsize=fontsize,
            ha="center",
            va="bottom",
        )


def scenario_title(scenario: str) -> str:
    """Human-readable title from a scenario_id slug."""
    return scenario.replace("_", " ").title()


def metric_label(metric: str) -> str:
    """Y-axis label for a metric name."""
    return config.METRIC_LABELS.get(metric, metric)


def _q_label(q_num: int) -> str:
    return f"Q{q_num}"


def _q_labels(q_nums: Sequence[int]) -> list[str]:
    return [_q_label(q) for q in q_nums]


def generate_text_summary(df: pd.DataFrame, scenario: str) -> str:
    """Return a short auto-generated text summary for *scenario*."""
    sub = _filter_df(df, scenario=scenario)
    if sub.empty:
        return f"No data for scenario '{scenario}'."

    lines: list[str] = [f"Scenario: {scenario_title(scenario)}"]

    model_div = sub.groupby("model")["gap_abs"].mean().sort_values(ascending=False)
    lines.append(
        f"  Highest-divergence model: {short_model(model_div.index[0])} "
        f"(mean |gap| = {model_div.iloc[0]:.3f})"
    )

    fam_div = sub.groupby("question_family")["gap_abs"].mean().sort_values(ascending=False)
    lines.append(
        f"  Most divergent family: {fam_div.index[0]} "
        f"(mean |gap| = {fam_div.iloc[0]:.3f})"
    )

    q_spread = (
        sub.groupby("question_number")
        .apply(lambda g: g.groupby("model")["gap_abs"].mean().std(), include_groups=False)
        .sort_values(ascending=False)
    )
    top_qs = q_spread.head(3).index.tolist()
    lines.append(
        f"  Largest cross-model spread: Q{', Q'.join(str(q) for q in top_qs)}"
    )

    mid = sub["turn"].min() + (sub["turn"].max() - sub["turn"].min()) / 2
    early = sub.loc[sub["turn"] <= mid, "gap_abs"].mean()
    late = sub.loc[sub["turn"] > mid, "gap_abs"].mean()
    pattern = "early-onset" if early >= late else "late-emerging"
    lines.append(f"  Divergence pattern: {pattern} (early={early:.3f}, late={late:.3f})")

    role_div = sub.groupby("participant_role")["gap_abs"].mean()
    stronger = role_div.idxmax()
    lines.append(f"  Stronger role: {stronger} (mean |gap| = {role_div[stronger]:.3f})")

    return "\n".join(lines)
