"""Debate history normalization utilities."""

from __future__ import annotations

from typing import Any


def get_structured_debate_history(
    structured_history: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Normalize canonical Agora structured history for downstream analyzers.

    Returns:
        Mapping of agent name to:
        - ``debate_turns``: list of turn records with public/private text fields
        - ``pre_interview``: optional pre-interview text
        - ``post_interview``: optional post-interview text
    """
    if "turns" not in structured_history:
        raise ValueError("Debate history must use canonical structured history format")

    agent_data: dict[str, dict[str, Any]] = {
        "Alpha": {"debate_turns": [], "pre_interview": None, "post_interview": None},
        "Beta": {"debate_turns": [], "pre_interview": None, "post_interview": None},
    }

    pre_interviews = structured_history.get("pre_interviews", {})
    post_interviews = structured_history.get("post_interviews", {})
    for slot in ("Alpha", "Beta"):
        pre_stage = pre_interviews.get(slot, {})
        post_stage = post_interviews.get(slot, {})
        agent_data[slot]["pre_interview"] = pre_stage.get("response")
        agent_data[slot]["post_interview"] = post_stage.get("response")

    for turn in structured_history.get("turns", []):
        turn_num = int(turn.get("turn_num", 0))
        for slot in ("Alpha", "Beta"):
            subturn = turn.get(slot, {})
            agent_data[slot]["debate_turns"].append(
                {
                    "turn_num": turn_num,
                    "public_speech": subturn.get("public_utterance") or "",
                    "private_reflection": subturn.get("private_utterance") or "",
                    "public_stance": "",
                }
            )

    return agent_data


__all__ = ["get_structured_debate_history"]
