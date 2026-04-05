"""Turn-level dynamics: line plots, heatmaps, top-K traces, peak/persistence."""

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


# ═════════════════════════════════════════════════════════════════════
# B4 — Turn dynamics by family across models
# ═════════════════════════════════════════════════════════════════════

def turn_dynamics_by_family(
    df: pd.DataFrame,
    scenario: str,
    *,
    metric: str = "gap_signed",
    models: list[str] | None = None,
    roles: list[str] | None = None,
    max_overlay: int = 8,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """B4: x = turn, y = mean gap, one line per model, faceted by family.

    When the number of models exceeds *max_overlay*, switch to
    faceting by model with family-coloured lines.
    """
    agg = summarize.by_turn_family(df, scenario, models=models, roles=roles, metric=metric)
    col = f"mean_{metric}"
    model_order = sorted(agg["model"].unique())
    families = config.FAMILIES
    turns = sorted(agg["turn"].unique())

    facet_by_model = len(model_order) > max_overlay

    if facet_by_model:
        n_mod = len(model_order)
        ncols = min(4, n_mod)
        nrows = (n_mod + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharey=True)
        axes_flat = np.array(axes).flatten()

        for mi, mdl in enumerate(model_order):
            ax = axes_flat[mi]
            for fam in families:
                fa = agg[(agg["model"] == mdl) & (agg["question_family"] == fam)]
                if fa.empty:
                    continue
                fa = fa.sort_values("turn")
                ax.plot(fa["turn"], fa[col], "o-", color=config.FAMILY_COLORS[fam],
                        markersize=4, label=fam)
            ax.set_title(short_model(mdl), fontsize=9)
            ax.set_xlabel("debate turn")
            if mi % ncols == 0:
                ax.set_ylabel(metric_label(metric))
            if metric == "gap_signed":
                ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
            ax.set_xticks(turns)

        for j in range(len(model_order), len(axes_flat)):
            axes_flat[j].set_visible(False)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
                   ncol=len(families), fontsize=8)
    else:
        model_colors = config.assign_model_colors(model_order)
        fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 5), sharey=True)
        if len(families) == 1:
            axes = [axes]

        for ax, fam in zip(axes, families):
            for mdl in model_order:
                fa = agg[(agg["model"] == mdl) & (agg["question_family"] == fam)]
                if fa.empty:
                    continue
                fa = fa.sort_values("turn")
                ax.plot(fa["turn"], fa[col], "o-", color=model_colors[mdl],
                        markersize=4, linewidth=1.3, label=short_model(mdl))
            ax.set_xlabel("debate turn")
            ax.set_title(fam, fontsize=10)
            if ax is axes[0]:
                ax.set_ylabel(metric_label(metric))
            if metric == "gap_signed":
                ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
            ax.set_xticks(turns)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
                   ncol=min(len(labels), 5), fontsize=7)

    fig.suptitle(
        f"Turn dynamics — mean {metric_label(metric)}\n{scenario_title(scenario)}",
        fontsize=11,
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
    roles: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """C1: rows = Q1–Q15, columns = turn, cell = mean gap for a single model."""
    sub = _filter_df(df, scenario=scenario, models=[model], roles=roles)
    piv = sub.pivot_table(index="question_number", columns="turn", values=metric, aggfunc="mean")
    piv = piv.reindex(index=config.QUESTION_ORDER)
    turns = sorted(sub["turn"].unique())

    is_signed = metric == "gap_signed"
    vmax = np.nanmax(np.abs(piv.values[np.isfinite(piv.values)])) if piv.size else 1
    cmap = "RdBu_r" if is_signed else "YlOrRd"
    vmin = -vmax if is_signed else 0

    fig, ax = plt.subplots(figsize=(max(6, len(turns) * 1.2), 8))
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([str(int(t)) for t in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(_q_labels(piv.index))
    ax.set_xlabel("debate turn")
    ax.set_ylabel("question")

    for bq in config.FAMILY_BOUNDARIES_AFTER_Q:
        ax.axhline(bq - 0.5, color="white", linewidth=2)
    family_labels = {3: "deliberative", 8: "evaluative", 12.5: "incentive"}
    for y_pos, label in family_labels.items():
        ax.text(-0.8, y_pos - 1, label, fontsize=8, ha="right", va="center", fontstyle="italic",
                transform=ax.get_yaxis_transform())

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > vmax * 0.65 else "black")

    fig.colorbar(im, ax=ax, label=metric_label(metric))
    ax.set_title(
        f"Question × Turn — {metric_label(metric)}\n"
        f"{short_model(model)} · {scenario_title(scenario)}",
        fontsize=11,
    )
    fig.tight_layout()
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
    """C2: rows = model, columns = question×turn, for one family."""
    sub = _filter_df(df, scenario=scenario, models=models, families=[family])
    turns = sorted(sub["turn"].unique())
    q_nums = sorted(sub[sub["question_family"] == family]["question_number"].unique())

    piv = sub.pivot_table(index="model", columns=["question_number", "turn"],
                          values=metric, aggfunc="mean")
    col_order = [(q, t) for q in q_nums for t in turns]
    piv = piv.reindex(columns=col_order)
    piv = piv.loc[sorted(piv.index)]

    is_signed = metric == "gap_signed"
    vmax = np.nanmax(np.abs(piv.values[np.isfinite(piv.values)])) if piv.size else 1
    cmap = "RdBu_r" if is_signed else "YlOrRd"
    vmin = -vmax if is_signed else 0

    fig, ax = plt.subplots(figsize=(max(10, len(col_order) * 0.7), max(4, len(piv) * 0.6)))
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([f"Q{q}·t{t}" for q, t in col_order], fontsize=6, rotation=60, ha="right")
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([short_model(m) for m in piv.index], fontsize=8)

    q_boundaries = []
    for i in range(1, len(q_nums)):
        q_boundaries.append(i * len(turns) - 0.5)
    for xb in q_boundaries:
        ax.axvline(xb, color="white", linewidth=1.5)

    fig.colorbar(im, ax=ax, label=metric_label(metric))
    ax.set_title(
        f"Model × (Question·Turn) — {family}\n"
        f"mean {metric_label(metric)} · {scenario_title(scenario)}",
        fontsize=11,
    )
    fig.tight_layout()
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════════════
# C3 — Top-K question traces
# ═════════════════════════════════════════════════════════════════════

def top_k_question_traces(
    df: pd.DataFrame,
    scenario: str,
    *,
    k: int = 5,
    metric: str = "gap_signed",
    models: list[str] | None = None,
    roles: list[str] | None = None,
    split_roles: bool = False,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """C3: line traces for the top-K most divergent questions."""
    top_qs = summarize.top_questions(df, scenario, k=k, metric="gap_abs")
    sub = _filter_df(df, scenario=scenario, models=models, roles=roles)
    sub = sub[sub["question_number"].isin(top_qs)]
    model_order = sorted(sub["model"].unique())
    model_colors = config.assign_model_colors(model_order)
    turns = sorted(sub["turn"].unique())

    n_q = len(top_qs)
    nrows = 2 if split_roles else 1
    fig, axes = plt.subplots(nrows, n_q, figsize=(3.5 * n_q, 4 * nrows), sharey=True)
    axes = np.atleast_2d(axes)

    role_list = ["alpha", "beta"] if split_roles else [None]
    for ri, role in enumerate(role_list):
        for qi, q_num in enumerate(top_qs):
            ax = axes[ri, qi]
            qf = sub[sub["question_number"] == q_num]
            if role:
                qf = qf[qf["participant_role"] == role]
            family = config.FAMILY_MAP.get(q_num, "")

            for mdl in model_order:
                mf = qf[qf["model"] == mdl].groupby("turn")[metric].mean()
                if not mf.empty:
                    ax.plot(mf.index, mf.values, "o-", color=model_colors[mdl],
                            markersize=4, linewidth=1.2, label=short_model(mdl))

            if metric == "gap_signed":
                ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
            role_tag = f" ({role})" if role else ""
            ax.set_title(f"Q{q_num} ({family}){role_tag}", fontsize=9)
            ax.set_xlabel("debate turn")
            ax.set_xticks(turns)
            if qi == 0:
                ax.set_ylabel(metric_label(metric))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(labels), 6), fontsize=7)
    fig.suptitle(
        f"Top-{k} questions by mean |gap| — {metric_label(metric)}\n{scenario_title(scenario)}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════════════
# C4 — Peak / persistence summary
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

    pp["label"] = pp["model"].map(short_model) + " | " + pp["participant_role"]
    families = config.FAMILIES

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, pp["label"].nunique() * 0.4)),
                             sharey=True)
    metrics_to_show = [("peak_gap_abs", "peak |gap|"), ("persistence", "persistence"),
                       ("mean_gap_abs", "mean |gap|")]

    for ax, (mcol, mtitle) in zip(axes, metrics_to_show):
        piv = pp.pivot_table(index="label", columns="question_family", values=mcol)
        piv = piv.reindex(columns=families)
        piv = piv.loc[sorted(piv.index)]

        vmax = np.nanmax(piv.values[np.isfinite(piv.values)]) if piv.size else 1
        im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, fontsize=9)
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels(piv.index, fontsize=7)
        ax.set_title(mtitle, fontsize=10)

        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if v > vmax * 0.65 else "black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Peak & persistence (threshold={threshold}) — {scenario_title(scenario)}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
