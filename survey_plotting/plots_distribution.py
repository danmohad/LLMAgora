"""Distribution plots: violin, box, paired public/private distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .utils import (
    _filter_df,
    metric_label,
    save_figure,
    scenario_title,
    short_model,
)


# ═════════════════════════════════════════════════════════════════════
# A2 — Public vs private distribution check
# ═════════════════════════════════════════════════════════════════════

def public_private_distribution(
    df: pd.DataFrame,
    scenario: str,
    *,
    models: list[str] | None = None,
    families: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """A2: paired violin of public vs private scores, faceted by model."""
    sub = _filter_df(df, scenario=scenario, models=models, families=families)
    model_order = sorted(sub["model"].unique())
    n_mod = len(model_order)

    fig, axes = plt.subplots(1, n_mod, figsize=(max(4, 3.5 * n_mod), 5), sharey=True)
    if n_mod == 1:
        axes = [axes]

    for ax, mdl in zip(axes, model_order):
        ms = sub[sub["model"] == mdl]
        pub_vals = ms["public_score"].dropna().values
        priv_vals = ms["private_score"].dropna().values

        data = []
        positions = []
        colors_list = []
        if len(pub_vals) > 1:
            data.append(pub_vals)
            positions.append(0)
            colors_list.append("#1f77b4")
        if len(priv_vals) > 1:
            data.append(priv_vals)
            positions.append(1)
            colors_list.append("#ff7f0e")

        if data:
            parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors_list[i])
                pc.set_alpha(0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["public", "private"], fontsize=9)
        ax.set_title(short_model(mdl), fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("score")

    fig.suptitle(
        f"Public vs Private score distributions\n{scenario_title(scenario)}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ═════════════════════════════════════════════════════════════════════
# B5 / D1 — Gap violin plots
# ═════════════════════════════════════════════════════════════════════

def gap_violin(
    df: pd.DataFrame,
    scenario: str,
    *,
    metric: str = "gap_signed",
    models: list[str] | None = None,
    roles: list[str] | None = None,
    split_roles: bool = False,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """B5/D1: violin of gap by model, faceted by family."""
    sub = _filter_df(df, scenario=scenario, models=models, roles=roles)
    families = config.FAMILIES
    model_order = sorted(sub["model"].unique())
    n_mod = len(model_order)

    if split_roles:
        fig, axes = plt.subplots(
            2, len(families),
            figsize=(5 * len(families), 9),
            sharey="row",
        )
        role_list = ["alpha", "beta"]
    else:
        fig, axes = plt.subplots(
            1, len(families),
            figsize=(5 * len(families), 5),
            sharey=True,
        )
        axes = np.atleast_2d(axes)
        role_list = [None]

    for ri, role in enumerate(role_list):
        for fi, fam in enumerate(families):
            ax = axes[ri, fi] if axes.ndim == 2 else axes[0, fi]
            fa = sub[sub["question_family"] == fam]
            if role is not None:
                fa = fa[fa["participant_role"] == role]

            data = []
            labels = []
            for mdl in model_order:
                vals = fa.loc[fa["model"] == mdl, metric].dropna().values
                if len(vals) > 1:
                    data.append(vals)
                else:
                    data.append([0.0])
                labels.append(short_model(mdl))

            if data:
                parts = ax.violinplot(data, showmeans=True, showmedians=True)
                model_colors = config.assign_model_colors(model_order)
                for pi, pc in enumerate(parts["bodies"]):
                    pc.set_facecolor(model_colors[model_order[pi]])
                    pc.set_alpha(0.6)

            ax.set_xticks(range(1, n_mod + 1))
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
            title_extra = f" ({role})" if role else ""
            ax.set_title(f"{fam}{title_extra}", fontsize=10)
            if fi == 0:
                ax.set_ylabel(metric_label(metric))
            if metric == "gap_signed":
                ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")

    fig.suptitle(
        f"Gap distribution by model — {metric_label(metric)}\n{scenario_title(scenario)}",
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
# D2 — Box plots
# ═════════════════════════════════════════════════════════════════════

def gap_boxplot(
    df: pd.DataFrame,
    scenario: str,
    *,
    metric: str = "gap_signed",
    models: list[str] | None = None,
    roles: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """D2: box plot of gap by model, faceted by family."""
    sub = _filter_df(df, scenario=scenario, models=models, roles=roles)
    families = config.FAMILIES
    model_order = sorted(sub["model"].unique())

    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 5), sharey=True)
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        fa = sub[sub["question_family"] == fam]
        data = []
        labels = []
        for mdl in model_order:
            vals = fa.loc[fa["model"] == mdl, metric].dropna().values
            data.append(vals if len(vals) else [np.nan])
            labels.append(short_model(mdl))

        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        model_colors = config.assign_model_colors(model_order)
        for pi, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(model_colors[model_order[pi]])
            patch.set_alpha(0.6)

        ax.set_xticks(range(1, len(model_order) + 1))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_title(fam, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(metric_label(metric))
        if metric == "gap_signed":
            ax.axhline(0, color="0.5", linewidth=0.5, linestyle=":")

    fig.suptitle(
        f"Gap box plot by model — {metric_label(metric)}\n{scenario_title(scenario)}",
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
# D5 — Role dumbbell (public mean vs private mean per model)
# ═════════════════════════════════════════════════════════════════════

def public_private_dumbbell(
    df: pd.DataFrame,
    scenario: str,
    *,
    models: list[str] | None = None,
    families: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """D5: dumbbell — public mean vs private mean per model."""
    sub = _filter_df(df, scenario=scenario, models=models, families=families)
    model_order = sorted(sub["model"].unique())
    n_mod = len(model_order)

    fig, ax = plt.subplots(figsize=(8, max(4, n_mod * 0.55)))
    for i, mdl in enumerate(model_order):
        ms = sub[sub["model"] == mdl]
        pub_m = ms["public_score"].mean()
        priv_m = ms["private_score"].mean()
        ax.plot([pub_m, priv_m], [i, i], color="0.6", linewidth=1.5, zorder=1)
        ax.scatter(pub_m, i, color="#1f77b4", s=50, zorder=3, label="public" if i == 0 else "")
        ax.scatter(priv_m, i, color="#ff7f0e", s=50, zorder=3, label="private" if i == 0 else "")

    ax.set_yticks(range(n_mod))
    ax.set_yticklabels([short_model(m) for m in model_order], fontsize=9)
    ax.set_xlabel("mean score")
    ax.legend(fontsize=8)
    ax.set_title(f"Public vs Private mean scores — {scenario_title(scenario)}", fontsize=11)
    fig.tight_layout()
    save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
