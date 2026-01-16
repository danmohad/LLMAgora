"""Utility functions for debate evaluation and analysis."""

from typing import Any, Dict, List, Optional, Sequence

from .memory import MemoryTurn


def get_structured_debate_history(memory_turns: Sequence[MemoryTurn]) -> Dict[str, Any]:
    """
    Transform a list of MemoryTurn objects into a structured format for analysis.

    Args:
        memory_turns: List of MemoryTurn objects from Agora.history()

    Returns:
        dict: Structured debate data organized by speaker:
            {
                'speaker_name': {
                    'debate_turns': [
                        {
                            'private_reflection': str,
                            'public_speech': str,
                            'turn_id': int,
                        },
                        ...
                    ],
                    'pre_interview': str or None,
                    'post_interview': str or None,
                }
            }
    """
    # Initialize structure for each speaker
    speakers: Dict[str, Dict[str, Any]] = {}
    
    # Track pending reflections (private thoughts before public speech)
    pending_reflections: Dict[str, str] = {}
    
    for turn in memory_turns:
        speaker_name = turn.metadata.get("speaker_name", turn.speaker_id)
        
        # Initialize speaker entry if needed
        if speaker_name not in speakers:
            speakers[speaker_name] = {
                "debate_turns": [],
                "pre_interview": None,
                "post_interview": None,
            }
        
        if turn.role == "pre_interview":
            speakers[speaker_name]["pre_interview"] = turn.private_reflection
            
        elif turn.role == "post_interview":
            speakers[speaker_name]["post_interview"] = turn.private_reflection
            
        elif turn.role == "reflection":
            # Store reflection to pair with next public speech
            pending_reflections[speaker_name] = turn.private_reflection or ""
            
        elif turn.role == "assistant":
            # This is a public speech - pair with pending reflection
            debate_turn = {
                "private_reflection": pending_reflections.pop(speaker_name, ""),
                "public_speech": turn.public_speech or "",
                "turn_id": turn.turn_id,
            }
            speakers[speaker_name]["debate_turns"].append(debate_turn)
    
    return speakers


__all__ = ["get_structured_debate_history"]

