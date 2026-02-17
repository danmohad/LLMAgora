"""Debate history normalization utilities.

This module converts either canonical Agora structured history payloads or
legacy memory-turn records into one normalized per-agent shape used by
analysis code.
"""

from __future__ import annotations

from typing import Any


def _get_or_create_turn(agent_turns: list[dict], turn_num: int) -> dict:
    for turn in agent_turns:
        if int(turn.get("turn_num", 0)) == int(turn_num):
            return turn
    entry = {
        "turn_num": int(turn_num),
        "public_speech": "",
        "private_reflection": "",
        "public_stance": "",
    }
    agent_turns.append(entry)
    agent_turns.sort(key=lambda item: int(item.get("turn_num", 0)))
    return entry


def get_structured_debate_history(memory_turns: Any) -> dict[str, dict[str, Any]]:
    """Normalize debate history for downstream analyzers.

    Returns:
        Mapping of agent name to:
        - ``debate_turns``: list of turn records with public/private text fields
        - ``pre_interview``: optional pre-interview text
        - ``post_interview``: optional post-interview text
    """
    # Preferred path: canonical structured history payload from Agora.
    if isinstance(memory_turns, dict) and "turns" in memory_turns:
        agent_data: dict[str, dict[str, Any]] = {
            "Alpha": {"debate_turns": [], "pre_interview": None, "post_interview": None},
            "Beta": {"debate_turns": [], "pre_interview": None, "post_interview": None},
        }

        pre_interviews = memory_turns.get("pre_interviews", {})
        post_interviews = memory_turns.get("post_interviews", {})
        for slot in ("Alpha", "Beta"):
            pre_stage = pre_interviews.get(slot, {})
            post_stage = post_interviews.get(slot, {})
            agent_data[slot]["pre_interview"] = pre_stage.get("response")
            agent_data[slot]["post_interview"] = post_stage.get("response")

        for turn in memory_turns.get("turns", []):
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

    # Fallback path: legacy list of MemoryTurn-like records.
    agent_data: dict[str, dict[str, Any]] = {}
    agent_turn_nums: dict[str, int] = {}

    for turn in memory_turns:
        speaker_name = turn.metadata.get("speaker_name", turn.speaker_id)

        if speaker_name not in agent_data:
            agent_data[speaker_name] = {
                "debate_turns": [],
                "pre_interview": None,
                "post_interview": None,
            }
            agent_turn_nums[speaker_name] = 0

        if turn.role == "pre_interview":
            agent_data[speaker_name]["pre_interview"] = turn.private_reflection
            continue

        if turn.role == "post_interview":
            agent_data[speaker_name]["post_interview"] = turn.private_reflection
            continue

        # Prefer explicit macro turn metadata when available.
        metadata_turn_num = turn.metadata.get("turn_num")
        event_type = turn.metadata.get("event_type")
        if metadata_turn_num is not None and event_type in {
            "public_utterance",
            "private_utterance",
        }:
            turn_data = _get_or_create_turn(
                agent_data[speaker_name]["debate_turns"], int(metadata_turn_num)
            )
            if event_type == "public_utterance":
                turn_data["public_speech"] = turn.public_speech or ""
            elif event_type == "private_utterance":
                turn_data["private_reflection"] = turn.private_reflection or ""
            continue

        # Legacy ordering fallback.
        if turn.role == "assistant":
            agent_turn_nums[speaker_name] += 1
            turn_num = agent_turn_nums[speaker_name]
            agent_data[speaker_name]["debate_turns"].append(
                {
                    "turn_num": turn_num,
                    "public_speech": turn.public_speech or "",
                    "private_reflection": "",
                    "public_stance": "",
                }
            )
        elif turn.role == "reflection" and agent_data[speaker_name]["debate_turns"]:
            agent_data[speaker_name]["debate_turns"][-1]["private_reflection"] = (
                turn.private_reflection or ""
            )

    return agent_data


__all__ = ["get_structured_debate_history"]
