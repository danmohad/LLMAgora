"""Cross-scenario analysis plots (Sections F1, F2).

Every plot is split by participant role (alpha / beta) by default.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, summarize
from .utils import (
    _filter_df,
    metric_label,
    save_figure,
    scenario_title,
    short_model,
)

_ROLES = ["alpha", "beta"]


# ═════════════════════════════════════════════════════════════════════
# F1 — Fixed-model cross-scenario family summary
# ═════════════════════════════════════════════════════════════════════

def fixed_model_family_summary(
    df: pd.DataFrame,
    model: str,
    *,
    metric: str = "gap_signed",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """F1a: lollipop — x = scenario, y = mean gap, colour = family — per role."""
    sub = _filter_df(df, models=[model])
    scenarios = sorted(sub["scenario"].unique())
    families = config.FAMILIES
    n_fam = len(families)
    bar_width = 0.7 / n_fam

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(scenarios) * 3), 5), sharey=True)

    for ax, role in zip(axes, _ROLES):
        rs = sub[sub["participant_role"] == role]
        agg = (
            rs.groupby(["scenario", "question_family"])[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: f"mean_{metric}"})
        )
        col = f"mean_{metric}"
        for fi, fam in enumerate(families):
            fa = agg[agg["question_family"] == fam].set_index("scenario").reindex(scenarios)
            x = np.arange(len(scenarios)) + fi * bar_width
            vals = fa[col].values
            color = config.FAMILY_COLORS[fam]
            ax.vlines(x, 0, vals, colors=color, linewidth=1.5)
            ax.plot(x, vals, "o", color=color, markersize=8,
                    label=fam if role == _ROLES[0] else "")

        ax.set_xticks(np.arange(len(scenarios)) + bar_width * (n_fam - 1) / 2)
        ax.set_xticklabels([scenario_title(s) for s in scenarios], rotation=20,
                           ha="right", fontsize=9)
        ax.axhline(0, color="0.5", linewidth=0.5)
        ax.set_title(role, fontsize=11)
        if role == _ROLES[0]:
            ax.set_ylabel(metric_label(metric))
            ax.legend(title="family", fontsize=8)

    fig.suptitle(
        f"Cross-scenario family summary — {short_model(model)}\n"
        f"mean {metric_label(metric)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def fixed_model_family_heatmap(
    df: pd.DataFrame,
    model: str,
    *,
    metric: str = "gap_signed",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """F1b: heatmap — rows = scenario, columns = family — per role."""
    sub = _filter_df(df, models=[model])

    fig, axes = plt.subplots(1, 2, figsize=(12, max(3, sub["scenario"].nunique() * 0.8)),
                             sharey=True)

    for ax, role in zip(axes, _ROLES):
        rs = sub[sub["participant_role"] == role]
        agg = (
            rs.groupby(["scenario", "question_family"])[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: f"mean_{metric}"})
        )
        col = f"mean_{metric}"
        piv = agg.pivot_table(index="scenario", columns="question_family", values=col)
        piv = piv.reindex(columns=config.FAMILIES)
        piv = piv.loc[sorted(piv.index)]

        is_signed = metric == "gap_signed"
        finite = piv.values[np.isfinite(piv.values)]
        vmax = np.abs(finite).max() if finite.size else 1
        cmap = "RdBu_r" if is_signed else "YlOrRd"
        vmin = -vmax if is_signed else 0

        im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(config.FAMILIES, fontsize=10)
        ax.set_yticks(range(piv.shape[0]))
        if role == _ROLES[0]:
            ax.set_yticklabels([scenario_title(s) for s in piv.index], fontsize=9)

        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(v) > vmax * 0.65 else "black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(role, fontsize=11)

    fig.suptitle(
        f"Scenario × Family — {short_model(model)}\nmean {metric_label(metric)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════════════
# F2 — Fixed-model cross-scenario turn dynamics
# ═════════════════════════════════════════════════════════════════════

def fixed_model_turn_dynamics(
    df: pd.DataFrame,
    model: str,
    *,
    metric: str = "gap_signed",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """F2: x = turn, y = mean gap, facet by family, colour = scenario — rows = role."""
    sub = _filter_df(df, models=[model])
    families = config.FAMILIES
    scenarios = sorted(sub["scenario"].unique())
    turns = sorted(sub["turn"].unique())

    n_scen = len(scenarios)
    cmap = plt.colormaps["Set2"] if n_scen <= 8 else plt.colormaps["tab20"]
    scen_colors = {s: cmap(i / max(n_scen - 1, 1)) for i, s in enumerate(scenarios)}

    fig, axes = plt.subplots(2, len(families),
                             figsize=(5 * len(families), 9), sharey=True)

    for ri, role in enumerate(_ROLES):
        rs = sub[sub["participant_role"] == role]
        for fi, fam in enumerate(families):
            ax = axes[ri, fi]
            for scen in scenarios:
                sf = rs[(rs["scenario"] == scen) & (rs["question_family"] == fam)]
                if sf.empty:
                    continue
                agg = sf.groupby("turn")[metric].mean().reindex(turns)
                ax.plot(turns, agg.values, "o-", color=scen_colors[scen],
                        markersize=5, linewidth=1.4, label=scenario_title(scen))
            ax.set_xlabel("debate turn")
            ax.set_xticks(turns)
            if ri == 0:
                ax.set_title(fam, fontsize=10)
            if fi == 0:
                ax.set_ylabel(f"{role}\n{metric_label(metric)}", fontsize=9)
            if metric == "gap_signed":
                ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(labels), 4), fontsize=8)
    fig.suptitle(
        f"Cross-scenario turn dynamics — {short_model(model)}\n"
        f"mean {metric_label(metric)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
