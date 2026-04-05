"""Configuration constants for the survey plotting module."""

from __future__ import annotations

from typing import Sequence

FAMILY_MAP: dict[int, str] = {
    **{q: "deliberative" for q in range(1, 7)},
    **{q: "evaluative" for q in range(7, 10)},
    **{q: "incentive" for q in range(10, 16)},
}

FAMILY_ORDER: dict[str, int] = {"deliberative": 1, "evaluative": 2, "incentive": 3}
FAMILIES: list[str] = ["deliberative", "evaluative", "incentive"]
QUESTION_ORDER: list[int] = list(range(1, 16))

FAMILY_BOUNDARIES_AFTER_Q: list[float] = [6.5, 9.5]
FAMILY_RANGES: dict[str, str] = {
    "deliberative": "Q1–Q6",
    "evaluative": "Q7–Q9",
    "incentive": "Q10–Q15",
}

FAMILY_COLORS: dict[str, str] = {
    "deliberative": "#1f77b4",
    "evaluative": "#ff7f0e",
    "incentive": "#2ca02c",
}

ROLE_COLORS: dict[str, str] = {"alpha": "#d62728", "beta": "#9467bd"}

DIRECTION_COLORS: dict[str, str] = {
    "positive": "#2ca02c",
    "negative": "#d62728",
    "zero": "#7f7f7f",
}

DEFAULT_FIGSIZE: tuple[float, float] = (12, 6)
DEFAULT_HEATMAP_FIGSIZE: tuple[float, float] = (14, 8)
DEFAULT_DPI: int = 150
DIVERGENCE_THRESHOLD: float = 0.5

METRIC_LABELS: dict[str, str] = {
    "gap_signed": "public − private",
    "gap_abs": "|public − private|",
}


def assign_model_colors(models: Sequence[str]) -> dict[str, tuple]:
    """Return a colour mapping for *models*, stable across calls for the same set."""
    import matplotlib.pyplot as plt

    names = sorted(set(models))
    n = len(names)
    cmap = plt.colormaps["tab10"] if n <= 10 else plt.colormaps["tab20"]
    return {m: cmap(i / max(n - 1, 1)) for i, m in enumerate(names)}
