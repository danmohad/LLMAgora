"""Configuration constants for survey plotting and analysis helpers."""

FAMILY_MAP: dict[int, str] = {
    **{q: "deliberative" for q in range(1, 7)},
    **{q: "evaluative" for q in range(7, 10)},
    **{q: "incentive" for q in range(10, 16)},
}

FAMILY_ORDER: dict[str, int] = {"deliberative": 1, "evaluative": 2, "incentive": 3}
FAMILIES: list[str] = ["deliberative", "evaluative", "incentive"]
DIVERGENCE_THRESHOLD: float = 0.5

