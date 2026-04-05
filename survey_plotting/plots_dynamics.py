"""Turn-level dynamics: line plots, heatmaps, top-K traces, peak/persistence.

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
    _q_labels,
    add_family_separators,
    metric_label,
    save_figure,
    scenario_title,
    short_model,
)

_ROLES = ["alpha", "beta"]


# ═════════════════════════════════════════════════════════════════════
# B4 — Turn dynamics by family across models
# ═════════════════════════════════════════════════════════════════════

def turn_dynamics_by_family(
    df: pd.DataFrame,
    scenario: str,
    *,
    metric: str = "gap_signed",
    models: list[str] | None = None,
    max_overlay: int = 8,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """B4: x = turn, y = mean gap, one line per model — rows = role, cols = family."""
    sub = _filter_df(df, scenario=scenario, models=models)
    model_order = sorted(sub["model"].unique())
    families = config.FAMILIES
    turns = sorted(sub["turn"].unique())
    model_colors = config.assign_model_colors(model_order)

    fig, axes = plt.subplots(2, len(families),
                             figsize=(5 * len(families), 9), sharey=True)

    for ri, role in enumerate(_ROLES):
        agg = summarize.by_turn_family(
            df, scenario, models=models, roles=[role], metric=metric,
        )
        col = f"mean_{metric}"
        for fi, fam in enumerate(families):
            ax = axes[ri, fi]
            for mdl in model_order:
                fa = agg[(agg["model"] == mdl) & (agg["question_family"] == fam)]
                if fa.empty:
                    continue
                fa = fa.sort_values("turn")
                ax.plot(fa["turn"], fa[col], "o-", color=model_colors[mdl],
                        markersize=4, linewidth=1.3, label=short_model(mdl))
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
               ncol=min(len(labels), 5), fontsize=7)
    fig.suptitle(
        f"Turn dynamics — mean {metric_label(metric)}\n{scenario_title(scenario)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════════════
# C1 — Turn × question heatmap for one model
# ═════════════════════════════════════════════════════════════════════

def model_turn_question_heatmap(
    df: pd.DataFrame,
    scenario: str,
    model: str,
    *,
    metric: str = "gap_signed",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """C1: rows = Q1–Q15, columns = turn — one panel per role."""
    sub = _filter_df(df, scenario=scenario, models=[model])

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    for ax, role in zip(axes, _ROLES):
        rs = sub[sub["participant_role"] == role]
        piv = rs.pivot_table(index="question_number", columns="turn",
                             values=metric, aggfunc="mean")
        piv = piv.reindex(index=config.QUESTION_ORDER)

        is_signed = metric == "gap_signed"
        finite = piv.values[np.isfinite(piv.values)]
        vmax = np.abs(finite).max() if finite.size else 1
        cmap = "RdBu_r" if is_signed else "YlOrRd"
        vmin = -vmax if is_signed else 0

        im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([str(int(t)) for t in piv.columns])
        ax.set_xlabel("debate turn")
        if role == _ROLES[0]:
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels(_q_labels(piv.index))
            ax.set_ylabel("question")

        for bq in config.FAMILY_BOUNDARIES_AFTER_Q:
            ax.axhline(bq - 0.5, color="white", linewidth=2)

        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > vmax * 0.65 else "black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(role, fontsize=11)

    fig.suptitle(
        f"Question × Turn — {metric_label(metric)}\n"
        f"{short_model(model)} · {scenario_title(scenario)}",
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
# C2 — Cross-model turn heatmap for a family
# ═════════════════════════════════════════════════════════════════════

def cross_model_family_turn_heatmap(
    df: pd.DataFrame,
    scenario: str,
    family: str,
    *,
    metric: str = "gap_signed",
    models: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """C2: rows = model, columns = question×turn — one panel per role."""
    sub = _filter_df(df, scenario=scenario, models=models, families=[family])
    turns = sorted(sub["turn"].unique())
    q_nums = sorted(sub[sub["question_family"] == family]["question_number"].unique())
    col_order = [(q, t) for q in q_nums for t in turns]

    fig, axes = plt.subplots(2, 1,
                             figsize=(max(10, len(col_order) * 0.7),
                                      max(8, sub["model"].nunique() * 1.0)),
                             sharey=True)

    for ax, role in zip(axes, _ROLES):
        rs = sub[sub["participant_role"] == role]
        piv = rs.pivot_table(index="model", columns=["question_number", "turn"],
                             values=metric, aggfunc="mean")
        piv = piv.reindex(columns=col_order)
        piv = piv.loc[sorted(piv.index)]

        is_signed = metric == "gap_signed"
        finite = piv.values[np.isfinite(piv.values)]
        vmax = np.abs(finite).max() if finite.size else 1
        cmap = "RdBu_r" if is_signed else "YlOrRd"
        vmin = -vmax if is_signed else 0

        im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(col_order)))
        ax.set_xticklabels([f"Q{q}·t{t}" for q, t in col_order], fontsize=5.5,
                           rotation=60, ha="right")
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels([short_model(m) for m in piv.index], fontsize=8)

        q_boundaries = [(i * len(turns) - 0.5) for i in range(1, len(q_nums))]
        for xb in q_boundaries:
            ax.axvline(xb, color="white", linewidth=1.5)

        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.set_title(f"{family} — {role}", fontsize=10)

    fig.suptitle(
        f"Model × (Question·Turn) — {family}\n"
        f"mean {metric_label(metric)} · {scenario_title(scenario)}",
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
# C3 — Top-K question traces (always split by role)
# ═════════════════════════════════════════════════════════════════════

def top_k_question_traces(
    df: pd.DataFrame,
    scenario: str,
    *,
    k: int = 5,
    metric: str = "gap_signed",
    models: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """C3: line traces for the top-K most divergent questions — rows = role."""
    top_qs = summarize.top_questions(df, scenario, k=k, metric="gap_abs")
    sub = _filter_df(df, scenario=scenario, models=models)
    sub = sub[sub["question_number"].isin(top_qs)]
    model_order = sorted(sub["model"].unique())
    model_colors = config.assign_model_colors(model_order)
    turns = sorted(sub["turn"].unique())

    n_q = len(top_qs)
    fig, axes = plt.subplots(2, n_q, figsize=(3.5 * n_q, 8), sharey=True)
    axes = np.atleast_2d(axes)

    for ri, role in enumerate(_ROLES):
        for qi, q_num in enumerate(top_qs):
            ax = axes[ri, qi]
            qf = sub[(sub["question_number"] == q_num) & (sub["participant_role"] == role)]
            family = config.FAMILY_MAP.get(q_num, "")

            for mdl in model_order:
                mf = qf[qf["model"] == mdl].groupby("turn")[metric].mean()
                if not mf.empty:
                    ax.plot(mf.index, mf.values, "o-", color=model_colors[mdl],
                            markersize=4, linewidth=1.2, label=short_model(mdl))

            if metric == "gap_signed":
                ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
            ax.set_title(f"Q{q_num} ({family}) — {role}", fontsize=8)
            ax.set_xlabel("debate turn")
            ax.set_xticks(turns)
            if qi == 0:
                ax.set_ylabel(metric_label(metric))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(labels), 6), fontsize=7)
    fig.suptitle(
        f"Top-{k} questions by mean |gap| — {metric_label(metric)}\n"
        f"{scenario_title(scenario)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════════════
# C4 — Peak / persistence summary (already shows roles — no split)
# ═════════════════════════════════════════════════════════════════════

def peak_persistence_plot(
    df: pd.DataFrame,
    scenario: str,
    *,
    models: list[str] | None = None,
    threshold: float = config.DIVERGENCE_THRESHOLD,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """C4: heatmap of peak gap and persistence per model × family × role."""
    pp = summarize.peak_persistence(df, scenario, models=models, threshold=threshold)
    if pp.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    families = config.FAMILIES
    metrics_to_show = [("peak_gap_abs", "peak |gap|"), ("persistence", "persistence"),
                       ("mean_gap_abs", "mean |gap|")]

    fig, axes = plt.subplots(2, 3,
                             figsize=(16, max(6, pp["model"].nunique() * 0.8)),
                             sharey="row")

    for ri, role in enumerate(_ROLES):
        rp = pp[pp["participant_role"] == role]
        for ci, (mcol, mtitle) in enumerate(metrics_to_show):
            ax = axes[ri, ci]
            piv = rp.pivot_table(
                index="model", columns="question_family", values=mcol,
            )
            piv = piv.reindex(columns=families)
            piv = piv.loc[sorted(piv.index)]

            vmax = np.nanmax(piv.values[np.isfinite(piv.values)]) if piv.size else 1
            im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(families)))
            ax.set_xticklabels(families, fontsize=9)
            ax.set_yticks(range(len(piv)))
            ax.set_yticklabels([short_model(m) for m in piv.index], fontsize=7)
            if ri == 0:
                ax.set_title(mtitle, fontsize=10)
            if ci == 0:
                ax.set_ylabel(role, fontsize=10)

            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if v > vmax * 0.65 else "black")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Peak & persistence (threshold={threshold}) — {scenario_title(scenario)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
