"""Advanced survey analysis helpers for public/private divergence studies."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .utils import _filter_df

_SCORE_COLUMNS = ("public_score", "private_score", "gap_signed")
_GROUP_COLUMNS = ("scenario", "model", "participant_role")


def _safe_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Return Pearson correlation or NaN when it is undefined."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 2:
        return float("nan")
    xa = xa[mask]
    ya = ya[mask]
    if np.allclose(xa, xa[0]) or np.allclose(ya, ya[0]):
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def _safe_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Return Spearman correlation or NaN when it is undefined."""
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average")
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average")
    return _safe_corr(x_rank.to_numpy(), y_rank.to_numpy())


def _zscore(series: pd.Series) -> pd.Series:
    """Return a zero-safe z-score."""
    std = float(series.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def _complete_cases(matrix: pd.DataFrame) -> pd.DataFrame:
    """Drop rows/columns that prevent multivariate analyses."""
    out = matrix.copy()
    out = out.dropna(axis=1, how="all")
    out = out.dropna(axis=0, how="any")
    return out


def _item_matrix(
    df: pd.DataFrame,
    *,
    score_col: str,
    family: str | None = None,
    scenario: str | None = None,
    model: str | None = None,
    role: str | None = None,
) -> pd.DataFrame:
    """Return an observation-by-item matrix for one score column."""
    sub = _filter_df(
        df,
        scenario=scenario,
        models=[model] if model is not None else None,
        roles=[role] if role is not None else None,
        families=[family] if family is not None else None,
    )
    if sub.empty:
        return pd.DataFrame()

    index_cols = [
        col
        for col in (
            "scenario",
            "model",
            "incentive_direction",
            "incentive_type",
            "participant_role",
            "turn",
        )
        if col in sub.columns
    ]
    return (
        sub.pivot_table(
            index=index_cols,
            columns="question_number",
            values=score_col,
            aggfunc="mean",
        )
        .sort_index(axis=1)
        .astype(float)
    )


def cronbach_alpha(matrix: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for a complete-case item matrix."""
    clean = _complete_cases(matrix)
    n_obs, n_items = clean.shape
    if n_obs < 2 or n_items < 2:
        return float("nan")
    item_variances = clean.var(axis=0, ddof=1)
    total_scores = clean.sum(axis=1)
    total_variance = float(total_scores.var(ddof=1))
    if total_variance <= 0 or not np.isfinite(total_variance):
        return float("nan")
    alpha = (n_items / (n_items - 1.0)) * (1.0 - float(item_variances.sum()) / total_variance)
    return float(alpha)


def average_inter_item_correlation(matrix: pd.DataFrame) -> float:
    """Mean off-diagonal correlation for a complete-case item matrix."""
    clean = _complete_cases(matrix)
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        return float("nan")
    corr = clean.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    if upper.empty:
        return float("nan")
    return float(upper.mean())


def reliability_by_family(
    df: pd.DataFrame,
    *,
    score_columns: tuple[str, ...] = _SCORE_COLUMNS,
) -> pd.DataFrame:
    """Compute family-level reliability metrics for each scenario/model/role."""
    rows: list[dict[str, Any]] = []
    group_cols = list(_GROUP_COLUMNS) + ["question_family"]

    for group_key, group in df.groupby(group_cols, observed=True):
        group_meta = dict(zip(group_cols, group_key, strict=False))
        for score_col in score_columns:
            matrix = _item_matrix(
                group,
                score_col=score_col,
                family=group_meta["question_family"],
                scenario=group_meta["scenario"],
                model=group_meta["model"],
                role=group_meta["participant_role"],
            )
            clean = _complete_cases(matrix)
            rows.append(
                {
                    **group_meta,
                    "score_type": score_col,
                    "n_observations": int(clean.shape[0]),
                    "n_items": int(clean.shape[1]),
                    "cronbach_alpha": cronbach_alpha(matrix),
                    "avg_inter_item_corr": average_inter_item_correlation(matrix),
                    "mean_score": float(np.nanmean(clean.to_numpy()))
                    if not clean.empty
                    else float("nan"),
                }
            )

    return pd.DataFrame(rows).sort_values(group_cols + ["score_type"]).reset_index(drop=True)


def principal_components_from_matrix(
    matrix: pd.DataFrame,
    *,
    max_components: int = 5,
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return scree data and loadings for a PCA run."""
    clean = _complete_cases(matrix)
    n_obs, n_items = clean.shape
    if n_obs < 2 or n_items < 2:
        return pd.DataFrame(), pd.DataFrame()

    values = clean.to_numpy(dtype=float)
    values = values - values.mean(axis=0, keepdims=True)
    if standardize:
        std = values.std(axis=0, ddof=1)
        keep = std > 0
        values = values[:, keep]
        columns = clean.columns[keep]
        if values.shape[1] < 2:
            return pd.DataFrame(), pd.DataFrame()
        values = values / std[keep]
    else:
        columns = clean.columns

    cov = np.cov(values, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    if total <= 0:
        return pd.DataFrame(), pd.DataFrame()

    n_comp = min(max_components, len(eigenvalues))
    explained = eigenvalues[:n_comp] / total
    cumulative = np.cumsum(explained)
    loadings = eigenvectors[:, :n_comp] * np.sqrt(eigenvalues[:n_comp])

    scree = pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, n_comp + 1)],
            "eigenvalue": eigenvalues[:n_comp],
            "explained_variance_ratio": explained,
            "cumulative_variance_ratio": cumulative,
        }
    )
    loading_df = pd.DataFrame(
        loadings,
        index=[int(col) for col in columns],
        columns=[f"PC{i}" for i in range(1, n_comp + 1)],
    ).rename_axis("question_number")
    return scree, loading_df.reset_index()


def dimensionality_report(
    df: pd.DataFrame,
    *,
    score_columns: tuple[str, ...] = ("public_score", "private_score"),
    max_components: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run PCA by scenario/model/role for the requested score columns."""
    scree_rows: list[pd.DataFrame] = []
    loading_rows: list[pd.DataFrame] = []

    for group_key, group in df.groupby(list(_GROUP_COLUMNS), observed=True):
        scenario, model, role = group_key
        for score_col in score_columns:
            matrix = _item_matrix(
                group,
                score_col=score_col,
                scenario=scenario,
                model=model,
                role=role,
            )
            scree, loadings = principal_components_from_matrix(
                matrix,
                max_components=max_components,
                standardize=True,
            )
            if not scree.empty:
                scree_rows.append(
                    scree.assign(
                        scenario=scenario,
                        model=model,
                        participant_role=role,
                        score_type=score_col,
                    )
                )
            if not loadings.empty:
                loading_rows.append(
                    loadings.assign(
                        scenario=scenario,
                        model=model,
                        participant_role=role,
                        score_type=score_col,
                    )
                )

    scree_df = pd.concat(scree_rows, ignore_index=True) if scree_rows else pd.DataFrame()
    loading_df = pd.concat(loading_rows, ignore_index=True) if loading_rows else pd.DataFrame()
    return scree_df, loading_df


def _hc1_standard_errors(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Compute HC1 robust standard errors."""
    n_obs, n_params = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = X.T @ (X * (residuals[:, None] ** 2))
    scale = n_obs / max(n_obs - n_params, 1)
    cov = scale * xtx_inv @ meat @ xtx_inv
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def fit_gap_regression(
    df: pd.DataFrame,
    *,
    outcome: str = "gap_abs",
    include_question_fixed_effects: bool = True,
) -> pd.DataFrame:
    """Fit a rich fixed-effects OLS as a notebook-friendly mixed-model proxy."""
    model_df = df.dropna(subset=[outcome]).copy()
    model_df["turn_centered"] = model_df["turn"] - float(model_df["turn"].mean())

    X_parts = [pd.DataFrame({"Intercept": np.ones(len(model_df), dtype=float)}, index=model_df.index)]
    X_parts.append(model_df[["turn_centered"]].astype(float))
    X_parts.append(pd.get_dummies(model_df["participant_role"], prefix="role", drop_first=True, dtype=float))
    X_parts.append(pd.get_dummies(model_df["question_family"], prefix="family", drop_first=True, dtype=float))
    X_parts.append(pd.get_dummies(model_df["model"], prefix="model", drop_first=True, dtype=float))
    X_parts.append(pd.get_dummies(model_df["scenario"], prefix="scenario", drop_first=True, dtype=float))

    if include_question_fixed_effects:
        X_parts.append(
            pd.get_dummies(
                model_df["question_number"].astype(str),
                prefix="question",
                drop_first=True,
                dtype=float,
            )
        )

    X = pd.concat(X_parts, axis=1)

    family_cols = [col for col in X.columns if col.startswith("family_")]
    role_cols = [col for col in X.columns if col.startswith("role_")]
    role_beta = X[role_cols[0]] if role_cols else pd.Series(0.0, index=X.index)
    for family_col in family_cols:
        suffix = family_col.removeprefix("family_")
        X[f"turn_x_{suffix}"] = X["turn_centered"] * X[family_col]
        X[f"role_beta_x_{suffix}"] = role_beta * X[family_col]

    y = model_df[outcome].to_numpy(dtype=float)
    X_mat = X.to_numpy(dtype=float)
    beta = np.linalg.pinv(X_mat) @ y
    fitted = X_mat @ beta
    residuals = y - fitted
    std_err = _hc1_standard_errors(X_mat, residuals)
    t_values = np.divide(beta, std_err, out=np.zeros_like(beta), where=std_err > 0)
    ci_half = 1.96 * std_err

    return (
        pd.DataFrame(
            {
                "term": X.columns,
                "estimate": beta,
                "std_error_hc1": std_err,
                "t_value": t_values,
                "ci_low": beta - ci_half,
                "ci_high": beta + ci_half,
                "abs_estimate": np.abs(beta),
            }
        )
        .sort_values("abs_estimate", ascending=False)
        .reset_index(drop=True)
    )


def bootstrap_group_means(
    df: pd.DataFrame,
    *,
    outcome: str = "gap_abs",
    group_cols: tuple[str, ...] = ("model", "participant_role", "question_family"),
    cluster_cols: tuple[str, ...] = ("question_number", "turn"),
    n_boot: int = 500,
    random_state: int = 0,
) -> pd.DataFrame:
    """Cluster bootstrap CIs for group means."""
    sub = df.dropna(subset=[outcome]).copy()
    if sub.empty:
        return pd.DataFrame()

    clusters = sub.loc[:, list(cluster_cols)].drop_duplicates().reset_index(drop=True)
    clusters["_cluster_id"] = np.arange(len(clusters))
    clustered = sub.merge(clusters, on=list(cluster_cols), how="left")

    point = (
        clustered.groupby(list(group_cols), observed=True)[outcome]
        .mean()
        .rename("estimate")
        .reset_index()
    )

    rng = np.random.default_rng(random_state)
    boot_rows: list[pd.DataFrame] = []
    for boot_idx in range(n_boot):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        weights = pd.Series(sampled).value_counts().rename_axis("_cluster_id").reset_index(name="_w")
        boot = clustered.merge(weights, on="_cluster_id", how="inner")
        agg = (
            boot.groupby(list(group_cols), observed=True)
            .apply(
                lambda g: float(np.average(g[outcome], weights=g["_w"])),
                include_groups=False,
            )
            .rename("boot_estimate")
            .reset_index()
            .assign(bootstrap=boot_idx)
        )
        boot_rows.append(agg)

    boot_df = pd.concat(boot_rows, ignore_index=True)
    summary = (
        boot_df.groupby(list(group_cols), observed=True)["boot_estimate"]
        .agg(
            bootstrap_mean="mean",
            bootstrap_std="std",
            ci_low=lambda s: float(np.quantile(s, 0.025)),
            ci_high=lambda s: float(np.quantile(s, 0.975)),
        )
        .reset_index()
    )
    return point.merge(summary, on=list(group_cols), how="left")


def bootstrap_model_ranks(
    df: pd.DataFrame,
    *,
    outcome: str = "gap_abs",
    scenario: str | None = None,
    n_boot: int = 500,
    random_state: int = 0,
) -> pd.DataFrame:
    """Bootstrap model ranks for overall divergence within a scenario."""
    sub = _filter_df(df, scenario=scenario).dropna(subset=[outcome]).copy()
    if sub.empty:
        return pd.DataFrame()

    clusters = sub.loc[:, ["question_number", "turn", "participant_role"]].drop_duplicates().reset_index(drop=True)
    clusters["_cluster_id"] = np.arange(len(clusters))
    clustered = sub.merge(clusters, on=["question_number", "turn", "participant_role"], how="left")

    rng = np.random.default_rng(random_state)
    rows: list[pd.DataFrame] = []
    for boot_idx in range(n_boot):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        weights = pd.Series(sampled).value_counts().rename_axis("_cluster_id").reset_index(name="_w")
        boot = clustered.merge(weights, on="_cluster_id", how="inner")
        agg = (
            boot.groupby("model", observed=True)
            .apply(lambda g: float(np.average(g[outcome], weights=g["_w"])), include_groups=False)
            .rename("boot_estimate")
            .sort_values(ascending=False)
            .reset_index()
        )
        agg["rank"] = np.arange(1, len(agg) + 1)
        agg["bootstrap"] = boot_idx
        rows.append(agg)

    boot_df = pd.concat(rows, ignore_index=True)
    return (
        boot_df.groupby("model", observed=True)
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            rank_ci_low=("rank", lambda s: float(np.quantile(s, 0.025))),
            rank_ci_high=("rank", lambda s: float(np.quantile(s, 0.975))),
            bootstrap_mean=("boot_estimate", "mean"),
        )
        .reset_index()
        .sort_values("mean_rank")
        .reset_index(drop=True)
    )


def _kmeans(
    X: np.ndarray,
    *,
    n_clusters: int,
    random_state: int = 0,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic K-means implementation."""
    if len(X) == 0:
        return np.array([], dtype=int), np.empty((0, 0))
    if n_clusters >= len(X):
        labels = np.arange(len(X), dtype=int)
        return labels, X.copy()

    rng = np.random.default_rng(random_state)
    centroids = [X[rng.integers(0, len(X))]]
    while len(centroids) < n_clusters:
        d2 = np.min(
            np.stack([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=1),
            axis=1,
        )
        next_idx = int(np.argmax(d2))
        centroids.append(X[next_idx])
    centroids_arr = np.vstack(centroids)

    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        distances = np.stack([np.sum((X - c) ** 2, axis=1) for c in centroids_arr], axis=1)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        updated = []
        for cluster_id in range(n_clusters):
            members = X[labels == cluster_id]
            updated.append(members.mean(axis=0) if len(members) else centroids_arr[cluster_id])
        centroids_arr = np.vstack(updated)
    return labels, centroids_arr


def trajectory_clusters(
    df: pd.DataFrame,
    *,
    metric: str = "gap_signed",
    scenario: str | None = None,
    n_clusters: int = 3,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster model-role trajectories over family-by-turn profiles."""
    sub = _filter_df(df, scenario=scenario).dropna(subset=[metric]).copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    turns = sorted(sub["turn"].unique())
    feature_order = [(family, turn) for family in config.FAMILIES for turn in turns]
    grouped = (
        sub.groupby(["scenario", "model", "participant_role", "question_family", "turn"], observed=True)[metric]
        .mean()
        .reset_index()
    )
    feature_matrix = (
        grouped.pivot_table(
            index=["scenario", "model", "participant_role"],
            columns=["question_family", "turn"],
            values=metric,
            aggfunc="mean",
        )
        .reindex(columns=feature_order)
        .fillna(0.0)
    )

    labels, centroids = _kmeans(
        feature_matrix.to_numpy(dtype=float),
        n_clusters=n_clusters,
        random_state=random_state,
    )
    assignments = feature_matrix.reset_index().assign(cluster=labels + 1)
    centroid_df = (
        pd.DataFrame(
            centroids,
            columns=pd.MultiIndex.from_tuples(feature_order, names=["question_family", "turn"]),
        )
        .assign(cluster=np.arange(1, len(centroids) + 1))
        .set_index("cluster")
        .stack(level=["question_family", "turn"], future_stack=True)
        .rename("centroid_value")
        .reset_index()
    )
    return assignments, centroid_df


def item_discrimination_table(
    df: pd.DataFrame,
    *,
    metric: str = "gap_abs",
    scenario: str | None = None,
) -> pd.DataFrame:
    """Rank questions by how strongly they differentiate models, roles, and scenarios."""
    sub = _filter_df(df, scenario=scenario).dropna(subset=[metric]).copy()
    if sub.empty:
        return pd.DataFrame()

    by_model = (
        sub.groupby(["question_number", "question_family", "model"], observed=True)[metric]
        .mean()
        .reset_index()
    )
    model_spread = (
        by_model.groupby(["question_number", "question_family"], observed=True)[metric]
        .agg(between_model_sd="std", between_model_range=lambda s: float(s.max() - s.min()))
        .reset_index()
    )

    by_role = (
        sub.groupby(["question_number", "question_family", "participant_role"], observed=True)[metric]
        .mean()
        .reset_index()
        .pivot_table(
            index=["question_number", "question_family"],
            columns="participant_role",
            values=metric,
        )
        .reset_index()
    )
    if "alpha" not in by_role:
        by_role["alpha"] = np.nan
    if "beta" not in by_role:
        by_role["beta"] = np.nan
    by_role["role_dif_abs"] = (by_role["alpha"] - by_role["beta"]).abs()

    by_scenario = (
        sub.groupby(["question_number", "question_family", "scenario"], observed=True)[metric]
        .mean()
        .reset_index()
    )
    scenario_spread = (
        by_scenario.groupby(["question_number", "question_family"], observed=True)[metric]
        .agg(scenario_sd="std", scenario_range=lambda s: float(s.max() - s.min()))
        .reset_index()
    )

    overall = (
        sub.groupby(["question_number", "question_family"], observed=True)[metric]
        .mean()
        .rename("mean_metric")
        .reset_index()
    )
    out = overall.merge(model_spread, on=["question_number", "question_family"], how="left")
    out = out.merge(
        by_role[["question_number", "question_family", "role_dif_abs"]],
        on=["question_number", "question_family"],
        how="left",
    )
    out = out.merge(scenario_spread, on=["question_number", "question_family"], how="left")
    for col in ("between_model_sd", "between_model_range", "role_dif_abs", "scenario_sd", "scenario_range"):
        out[col] = out[col].fillna(0.0)
    out["discrimination_index"] = (
        _zscore(out["between_model_sd"])
        + _zscore(out["between_model_range"])
        + _zscore(out["role_dif_abs"])
        + _zscore(out["scenario_sd"])
    ) / 4.0
    return out.sort_values("discrimination_index", ascending=False).reset_index(drop=True)


def public_private_coupling(
    df: pd.DataFrame,
    *,
    scenario: str | None = None,
) -> pd.DataFrame:
    """Compute contemporaneous and lagged public/private coupling by question."""
    sub = _filter_df(df, scenario=scenario).copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["scenario", "model", "participant_role", "question_family", "question_number"]
    for group_key, group in sub.groupby(group_cols, observed=True):
        group = group.sort_values("turn")
        public = group["public_score"].to_numpy(dtype=float)
        private = group["private_score"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_cols, group_key, strict=False)),
                "n_turns": int(len(group)),
                "corr_same_turn": _safe_corr(public, private),
                "corr_private_leads_public": _safe_corr(private[:-1], public[1:]) if len(group) >= 3 else float("nan"),
                "corr_public_leads_private": _safe_corr(public[:-1], private[1:]) if len(group) >= 3 else float("nan"),
                "mean_gap_signed": float(group["gap_signed"].mean()),
                "mean_gap_abs": float(group["gap_abs"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def response_style_summary(
    df: pd.DataFrame,
    *,
    score_columns: tuple[str, ...] = ("public_score", "private_score"),
) -> pd.DataFrame:
    """Summarize midpoint, extremity, and polarity tendencies."""
    rows: list[dict[str, Any]] = []
    all_scores = pd.concat([df[col] for col in score_columns], ignore_index=True)
    scale_min = float(all_scores.min())
    scale_max = float(all_scores.max())
    midpoint = (scale_min + scale_max) / 2.0
    half_range = max((scale_max - scale_min) / 2.0, 1e-9)
    extreme_cutoff = midpoint + 0.75 * half_range

    for group_key, group in df.groupby(list(_GROUP_COLUMNS), observed=True):
        meta = dict(zip(_GROUP_COLUMNS, group_key, strict=False))
        for score_col in score_columns:
            values = group[score_col].dropna().astype(float)
            if values.empty:
                continue
            deviation = (values - midpoint).abs()
            rows.append(
                {
                    **meta,
                    "score_type": score_col,
                    "n_responses": int(len(values)),
                    "mean_score": float(values.mean()),
                    "std_score": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                    "midpoint_rate": float((values == midpoint).mean()),
                    "exact_extreme_rate": float(((values == scale_min) | (values == scale_max)).mean()),
                    "near_extreme_rate": float((deviation >= abs(extreme_cutoff - midpoint)).mean()),
                    "positive_rate": float((values > midpoint).mean()),
                    "negative_rate": float((values < midpoint).mean()),
                    "mean_abs_midpoint_deviation": float(deviation.mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(list(_GROUP_COLUMNS) + ["score_type"]).reset_index(drop=True)


def strategic_concealment_index(
    df: pd.DataFrame,
    *,
    threshold: float = config.DIVERGENCE_THRESHOLD,
) -> pd.DataFrame:
    """Build a composite index of public/private divergence pressure."""
    base = (
        df.groupby(list(_GROUP_COLUMNS), observed=True)
        .agg(
            mean_gap_abs=("gap_abs", "mean"),
            peak_gap_abs=("gap_abs", "max"),
            persistence=("gap_abs", lambda s: float((s >= threshold).mean())),
            directional_consistency=("gap_signed", lambda s: float(abs(s.mean()))),
        )
        .reset_index()
    )

    coupling = (
        public_private_coupling(df)
        .groupby(list(_GROUP_COLUMNS), observed=True)
        .agg(mean_same_turn_corr=("corr_same_turn", "mean"))
        .reset_index()
    )
    coupling["coupling_inverse"] = 1.0 - coupling["mean_same_turn_corr"].clip(lower=-1.0, upper=1.0).fillna(0.0)

    style = response_style_summary(df)
    style_pivot = (
        style.pivot_table(
            index=list(_GROUP_COLUMNS),
            columns="score_type",
            values="mean_abs_midpoint_deviation",
            aggfunc="mean",
        )
        .reset_index()
    )
    if "public_score" not in style_pivot:
        style_pivot["public_score"] = np.nan
    if "private_score" not in style_pivot:
        style_pivot["private_score"] = np.nan
    style_pivot["style_shift"] = (style_pivot["public_score"] - style_pivot["private_score"]).abs()

    out = base.merge(coupling[list(_GROUP_COLUMNS) + ["coupling_inverse"]], on=list(_GROUP_COLUMNS), how="left")
    out = out.merge(style_pivot[list(_GROUP_COLUMNS) + ["style_shift"]], on=list(_GROUP_COLUMNS), how="left")
    feature_cols = [
        "mean_gap_abs",
        "peak_gap_abs",
        "persistence",
        "directional_consistency",
        "coupling_inverse",
        "style_shift",
    ]
    for col in feature_cols:
        out[col] = out[col].fillna(float(out[col].mean()) if out[col].notna().any() else 0.0)
        out[f"z_{col}"] = _zscore(out[col])
    z_cols = [f"z_{col}" for col in feature_cols]
    out["strategic_concealment_index"] = out[z_cols].mean(axis=1)
    return out.sort_values("strategic_concealment_index", ascending=False).reset_index(drop=True)


def scenario_invariance_table(
    df: pd.DataFrame,
    *,
    metric: str = "gap_abs",
) -> pd.DataFrame:
    """Estimate structural invariance across scenarios with profile correlations."""
    sub = df.dropna(subset=[metric]).copy()
    scenarios = sorted(sub["scenario"].unique())
    rows: list[dict[str, Any]] = []

    for role in sorted(sub["participant_role"].unique()):
        role_df = sub[sub["participant_role"] == role]
        for family in config.FAMILIES:
            fam_df = role_df[role_df["question_family"] == family]
            if fam_df.empty:
                continue

            model_profile = (
                fam_df.groupby(["scenario", "model"], observed=True)[metric]
                .mean()
                .unstack("model")
            )
            question_profile = (
                fam_df.groupby(["scenario", "question_number"], observed=True)[metric]
                .mean()
                .unstack("question_number")
            )
            model_question_profile = (
                fam_df.groupby(["scenario", "model", "question_number"], observed=True)[metric]
                .mean()
                .unstack(["model", "question_number"])
            )

            for left, right in combinations(scenarios, 2):
                if left not in model_profile.index or right not in model_profile.index:
                    continue
                rows.append(
                    {
                        "participant_role": role,
                        "question_family": family,
                        "scenario_left": left,
                        "scenario_right": right,
                        "model_rank_spearman": _safe_spearman(model_profile.loc[left], model_profile.loc[right]),
                        "question_profile_pearson": _safe_corr(question_profile.loc[left], question_profile.loc[right]),
                        "model_question_profile_pearson": _safe_corr(
                            model_question_profile.loc[left],
                            model_question_profile.loc[right],
                        ),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["participant_role", "question_family", "scenario_left", "scenario_right"]
    ).reset_index(drop=True)
