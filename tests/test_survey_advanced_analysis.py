import numpy as np
import pandas as pd

from survey_plotting import advanced_analysis


def _build_tidy_df() -> pd.DataFrame:
    rows = []
    family_questions = {
        "deliberative": [1, 2],
        "evaluative": [7, 8],
        "incentive": [10, 11],
    }
    scenario_offsets = {"scenario_a": 0.0, "scenario_b": 0.2}
    model_offsets = {"model_a": 0.0, "model_b": 0.35}
    role_offsets = {"alpha": 0.0, "beta": -0.15}
    family_gap = {"deliberative": 0.15, "evaluative": 0.25, "incentive": 0.5}

    for scenario, scenario_offset in scenario_offsets.items():
        for model, model_offset in model_offsets.items():
            for role, role_offset in role_offsets.items():
                for family, questions in family_questions.items():
                    for question_number in questions:
                        question_offset = (question_number % 3) * 0.05
                        for turn in (1, 2, 3, 4):
                            public = (
                                scenario_offset
                                + model_offset
                                + role_offset
                                + question_offset
                                + turn * 0.1
                            )
                            private = public - family_gap[family] - (0.03 if model == "model_b" else 0.0)
                            rows.append(
                                {
                                    "scenario": scenario,
                                    "model": model,
                                    "incentive_direction": "positive",
                                    "incentive_type": "future",
                                    "question_number": question_number,
                                    "question_family": family,
                                    "question_text": f"Question {question_number}",
                                    "participant_role": role,
                                    "turn": turn,
                                    "public_score": public,
                                    "private_score": private,
                                    "public_se": 0.01,
                                    "private_se": 0.01,
                                    "gap_signed": public - private,
                                    "gap_abs": abs(public - private),
                                    "question_order": question_number,
                                    "family_order": 1,
                                    "direction_label": "positive",
                                }
                            )
    return pd.DataFrame(rows)


def test_reliability_by_family_returns_expected_columns():
    tidy_df = _build_tidy_df()

    result = advanced_analysis.reliability_by_family(tidy_df)

    assert not result.empty
    assert {"cronbach_alpha", "avg_inter_item_corr", "score_type"}.issubset(result.columns)
    assert set(result["score_type"]) == {"public_score", "private_score", "gap_signed"}


def test_dimensionality_report_returns_scree_and_loadings():
    tidy_df = _build_tidy_df()

    scree, loadings = advanced_analysis.dimensionality_report(tidy_df, max_components=3)

    assert not scree.empty
    assert not loadings.empty
    assert set(["component", "explained_variance_ratio", "score_type"]).issubset(scree.columns)
    assert "question_number" in loadings.columns


def test_fit_gap_regression_returns_coefficients_and_interactions():
    tidy_df = _build_tidy_df()

    coeffs = advanced_analysis.fit_gap_regression(tidy_df)

    assert not coeffs.empty
    assert "Intercept" in set(coeffs["term"])
    assert any(term.startswith("turn_x_") for term in coeffs["term"])
    assert {"estimate", "std_error_hc1", "ci_low", "ci_high"}.issubset(coeffs.columns)


def test_bootstrap_group_means_and_ranks_are_nonempty():
    tidy_df = _build_tidy_df()

    summary = advanced_analysis.bootstrap_group_means(tidy_df, n_boot=40, random_state=7)
    ranks = advanced_analysis.bootstrap_model_ranks(
        tidy_df,
        scenario="scenario_a",
        n_boot=40,
        random_state=7,
    )

    assert not summary.empty
    assert {"estimate", "ci_low", "ci_high"}.issubset(summary.columns)
    assert not ranks.empty
    assert {"mean_rank", "rank_ci_low", "rank_ci_high"}.issubset(ranks.columns)


def test_trajectory_clusters_and_item_discrimination_work():
    tidy_df = _build_tidy_df()

    assignments, centroids = advanced_analysis.trajectory_clusters(
        tidy_df,
        n_clusters=3,
        random_state=3,
    )
    item_table = advanced_analysis.item_discrimination_table(tidy_df)

    assert not assignments.empty
    assert not centroids.empty
    assert "cluster" in assignments.columns
    assert "discrimination_index" in item_table.columns
    assert item_table["discrimination_index"].is_monotonic_decreasing


def test_coupling_response_style_index_and_invariance_are_nonempty():
    tidy_df = _build_tidy_df()

    coupling = advanced_analysis.public_private_coupling(tidy_df)
    response_style = advanced_analysis.response_style_summary(tidy_df)
    index_df = advanced_analysis.strategic_concealment_index(tidy_df)
    invariance = advanced_analysis.scenario_invariance_table(tidy_df)

    assert not coupling.empty
    assert {"corr_same_turn", "corr_private_leads_public"}.issubset(coupling.columns)
    assert not response_style.empty
    assert np.isclose(response_style["midpoint_rate"].min(), 0.0)
    assert not index_df.empty
    assert "strategic_concealment_index" in index_df.columns
    assert not invariance.empty
    assert {"model_rank_spearman", "question_profile_pearson"}.issubset(invariance.columns)
