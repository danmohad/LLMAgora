"""Summary-level plots: heatmaps, lollipops, ranked dots, role asymmetry."""

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
    annotate_top_n,
    metric_label,
    save_figure,
    scenario_title,
    short_model,
)

# ═════════════════════════════════════════════════════════════════════
# A1 — Question coverage heatmap
# ═════════════════════════════════════════════════════════════════════

def coverage_heatmap(
    df: pd.DataFrame,
    scenario: str,
    *,
    metric: str = "gap_abs",
    models: list[str] | None = None,
    roles: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """A1: question × turn heatmap of observation count or mean gap_abs."""
    sub = _filter_df(df, scenario=scenario, models=models, roles=roles)
    piv = sub.pivot_table(
        index="question_number",
        columns="turn",
        values=metric,
        aggfunc="count" if metric == "count" else "mean",
    ).reindex(index=config.QUESTION_ORDER)

    fig, ax = plt.subplots(figsize=config.DEFAULT_HEATMAP_FIGSIZE)
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([str(int(c)) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(_q_labels(piv.index))
    ax.set_xlabel("debate turn")
    val_label = "obs count" if metric == "count" else metric_label(metric)
    ax.set_ylabel("question")
    fig.colorbar(im, ax=ax, label=val_label)

    for boundary_q in config.FAMILY_BOUNDARIES_AFTER_Q:
        y_pos = boundary_q - 0.5
        ax.axhline(y_pos, color="white", linewidth=2)

    ax.set_title(f"Coverage: {scenario_title(scenario)}")
    fig.tight_layout()
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════════════
# B1 — Model × question heatmap of mean gap
# ═════════════════════════════════════════════════════════════════════

def model_question_heatmap(
    df: pd.DataFrame,
    scenario: str,
    *,
    metric: str = "gap_signed",
    models: list[str] | None = None,
    roles: list[str] | None = None,
    annotate: bool = True,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """B1: rows = models, columns = Q1–Q15, cell = mean gap."""
    agg = summarize.by_model_question(df, scenario, models=models, roles=roles, metric=metric)
    col = f"mean_{metric}"
    piv = agg.pivot_table(index="model", columns="question_number", values=col)
    piv = piv.reindex(columns=config.QUESTION_ORDER)
    piv = piv.loc[sorted(piv.index)]

    is_signed = metric == "gap_signed"
    cmap = "RdBu_r" if is_signed else "YlOrRd"
    vmin = -piv.values[np.isfinite(piv.values)].max() if is_signed else 0
    vmax = piv.values[np.isfinite(piv.values)].max()

    fig, ax = plt.subplots(figsize=config.DEFAULT_HEATMAP_FIGSIZE)
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(_q_labels(piv.columns), fontsize=9)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([short_model(m) for m in piv.index], fontsize=9)
    ax.set_xlabel("question")
    ax.set_ylabel("model")

    for bq in config.FAMILY_BOUNDARIES_AFTER_Q:
        x_pos = bq - 0.5
        ax.axvline(x_pos, color="white", linewidth=2.5)

    if annotate:
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(v) > vmax * 0.65 else "black")

    fig.colorbar(im, ax=ax, label=metric_label(metric))
    ax.set_title(
        f"Model × Question — mean {metric_label(metric)}\n{scenario_title(scenario)}",
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
# B2 — Model × family lollipop / dot summary
# ═════════════════════════════════════════════════════════════════════

def model_family_lollipop(
    df: pd.DataFrame,
    scenario: str,
    *,
    metric: str = "gap_signed",
    models: list[str] | None = None,
    roles: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """B2: x = model, y = mean metric, colour = family."""
    agg = summarize.by_model_family(df, scenario, models=models, roles=roles, metric=metric)
    col = f"mean_{metric}"
    model_order = sorted(agg["model"].unique())
    n_fam = len(config.FAMILIES)
    bar_width = 0.7 / n_fam

    fig, ax = plt.subplots(figsize=(max(10, len(model_order) * 1.2), 6))
    for fi, fam in enumerate(config.FAMILIES):
        sub = agg[agg["question_family"] == fam]
        vals = sub.set_index("model").reindex(model_order)[col].values
        x = np.arange(len(model_order)) + fi * bar_width
        ax.stem(
            x,
            vals,
            linefmt="-",
            markerfmt="o",
            basefmt=" ",
            label=fam,
        )
        for xi, yi in zip(x, vals):
            if np.isfinite(yi):
                color = config.FAMILY_COLORS[fam]
                ax.plot(xi, yi, "o", color=color, markersize=8, zorder=5)
                ax.vlines(xi, 0, yi, colors=color, linewidth=1.5)

    ax.set_xticks(np.arange(len(model_order)) + bar_width * (n_fam - 1) / 2)
    ax.set_xticklabels([short_model(m) for m in model_order], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(metric_label(metric))
    ax.legend(title="family")
    ax.set_title(
        f"Family-level divergence by model — mean {metric_label(metric)}\n{scenario_title(scenario)}",
        fontsize=11,
    )
    ax.axhline(0, color="0.5", linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════════════
# B3 — Ranked model divergence
# ═════════════════════════════════════════════════════════════════════

def ranked_model_divergence(
    df: pd.DataFrame,
    scenario: str,
    *,
    metric: str = "gap_abs",
    by_family: bool = False,
    models: list[str] | None = None,
    n_annotate: int = 3,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """B3: horizontal lollipop ranked by cumulative divergence."""
    sub = _filter_df(df, scenario=scenario, models=models)

    if by_family:
        agg = sub.groupby(["model", "question_family"])[metric].sum().reset_index()
        families = config.FAMILIES
        fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 6), sharey=False)
        if len(families) == 1:
            axes = [axes]
        for ax, fam in zip(axes, families):
            fa = agg[agg["question_family"] == fam].sort_values(metric)
            ax.barh(
                [short_model(m) for m in fa["model"]],
                fa[metric],
                color=config.FAMILY_COLORS[fam],
                height=0.6,
            )
            ax.set_xlabel(f"cumulative {metric_label(metric)}")
            ax.set_title(fam)
        fig.suptitle(
            f"Ranked cumulative divergence by family — {scenario_title(scenario)}",
            fontsize=11,
        )
    else:
        agg = sub.groupby("model")[metric].sum().sort_values()
        fig, ax = plt.subplots(figsize=(8, max(4, len(agg) * 0.55)))
        labels = [short_model(m) for m in agg.index]
        colors = config.assign_model_colors(agg.index)
        ax.barh(
            labels,
            agg.values,
            color=[colors[m] for m in agg.index],
            height=0.6,
        )
        ax.set_xlabel(f"cumulative {metric_label(metric)}")
        ax.set_title(
            f"Ranked cumulative divergence — {scenario_title(scenario)}",
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
# B6 — Role asymmetry dumbbell
# ═════════════════════════════════════════════════════════════════════

def role_asymmetry_dumbbell(
    df: pd.DataFrame,
    scenario: str,
    *,
    metric: str = "gap_signed",
    models: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """B6: dumbbell plot — alpha vs beta per model, faceted by family."""
    agg = summarize.role_asymmetry(df, scenario, models=models, metric=metric)
    col = f"mean_{metric}"
    families = config.FAMILIES
    model_order = sorted(agg["model"].unique())
    n_mod = len(model_order)

    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), max(4, n_mod * 0.5)),
                             sharey=True)
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        fa = agg[agg["question_family"] == fam]
        for i, mdl in enumerate(model_order):
            row_a = fa[(fa["model"] == mdl) & (fa["participant_role"] == "alpha")]
            row_b = fa[(fa["model"] == mdl) & (fa["participant_role"] == "beta")]
            va = row_a[col].values[0] if len(row_a) else np.nan
            vb = row_b[col].values[0] if len(row_b) else np.nan
            if np.isfinite(va) and np.isfinite(vb):
                ax.plot([va, vb], [i, i], color="0.6", linewidth=1.5, zorder=1)
            if np.isfinite(va):
                ax.scatter(va, i, color=config.ROLE_COLORS["alpha"], s=50, zorder=3, label="alpha" if i == 0 else "")
            if np.isfinite(vb):
                ax.scatter(vb, i, color=config.ROLE_COLORS["beta"], s=50, zorder=3, label="beta" if i == 0 else "")

        ax.set_yticks(range(n_mod))
        ax.set_yticklabels([short_model(m) for m in model_order], fontsize=8)
        ax.set_xlabel(metric_label(metric))
        ax.set_title(fam)
        if metric == "gap_signed":
            ax.axvline(0, color="0.5", linewidth=0.5, linestyle=":")
        if fam == families[0]:
            ax.legend(loc="lower right", fontsize=7)

    fig.suptitle(
        f"Role asymmetry (alpha vs beta) — mean {metric_label(metric)}\n{scenario_title(scenario)}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════════════
# B7 — Question-level model spread
# ═════════════════════════════════════════════════════════════════════

def question_model_spread(
    df: pd.DataFrame,
    scenario: str,
    *,
    metric: str = "gap_signed",
    models: list[str] | None = None,
    roles: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """B7: x = question, y = mean gap, one point per model, faceted by family."""
    agg = summarize.by_model_question(df, scenario, models=models, roles=roles, metric=metric)
    col = f"mean_{metric}"
    families = config.FAMILIES
    colors = config.assign_model_colors(agg["model"].unique())

    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 5), sharey=True)
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        fa = agg[agg["question_family"] == fam]
        q_nums = sorted(fa["question_number"].unique())
        for mdl in sorted(fa["model"].unique()):
            mf = fa[fa["model"] == mdl].set_index("question_number").reindex(q_nums)
            ax.plot(q_nums, mf[col].values, "o-", color=colors[mdl], markersize=5,
                    linewidth=0.8, alpha=0.7, label=short_model(mdl))

        ax.set_xticks(q_nums)
        ax.set_xticklabels(_q_labels(q_nums), fontsize=8)
        ax.set_xlabel("question")
        ax.set_title(fam)
        if fam == families[0]:
            ax.set_ylabel(metric_label(metric))
        if metric == "gap_signed":
            ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(labels), 5), fontsize=7)
    fig.suptitle(
        f"Question-level model spread — mean {metric_label(metric)}\n{scenario_title(scenario)}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
