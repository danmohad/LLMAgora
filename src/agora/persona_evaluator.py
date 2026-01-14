"""Persona adherence evaluator for debate agents."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .llm import LLMClient


@dataclass
class PersonaScore:
    """Score for a single turn or cumulative turns."""
    turn_num: int
    score: int  # 1-5 scale


@dataclass
class AgentPersonaEvaluation:
    """Complete persona evaluation for one agent."""
    persona_id: str
    public_turn_scores: List[PersonaScore] = field(default_factory=list)
    private_turn_scores: List[PersonaScore] = field(default_factory=list)
    public_cumulative_scores: List[PersonaScore] = field(default_factory=list)
    private_cumulative_scores: List[PersonaScore] = field(default_factory=list)
    full_debate_public_score: Optional[int] = None
    full_debate_private_score: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "persona_id": self.persona_id,
            "public_turn_scores": [(s.turn_num, s.score) for s in self.public_turn_scores],
            "private_turn_scores": [(s.turn_num, s.score) for s in self.private_turn_scores],
            "public_cumulative_scores": [(s.turn_num, s.score) for s in self.public_cumulative_scores],
            "private_cumulative_scores": [(s.turn_num, s.score) for s in self.private_cumulative_scores],
            "full_debate_public_score": self.full_debate_public_score,
            "full_debate_private_score": self.full_debate_private_score,
        }


@dataclass
class DebatePersonaEvaluation:
    """Complete persona evaluation for both agents in a debate."""
    alpha: AgentPersonaEvaluation
    beta: AgentPersonaEvaluation
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "alpha": self.alpha.to_dict(),
            "beta": self.beta.to_dict(),
        }


class PersonaEvaluator:
    """Evaluates how well agents adhere to their assigned personas during debates."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        personas: Dict,
        model: str = "anthropic/claude-sonnet-4",
    ):
        """
        Initialize the persona evaluator.
        
        Args:
            llm_client: LLM client for making evaluation requests
            personas: Dictionary containing persona definitions (from personas.json)
            model: Model to use for evaluation
        """
        self.llm_client = llm_client
        self.personas = personas.get("personas", {})
        self.model = model
    
    def _create_evaluation_prompt(
        self,
        text: str,
        persona_id: str,
        turn_label: str = "turn",
    ) -> str:
        """
        Create a prompt for evaluating persona adherence.
        
        Args:
            text: The text to evaluate
            persona_id: ID of the persona to evaluate against
            turn_label: Label describing what's being evaluated (e.g., "turn 1", "cumulative turns 1-3")
        
        Returns:
            Evaluation prompt
        """
        if persona_id not in self.personas:
            raise ValueError(f"Unknown persona id: {persona_id}")
        
        persona = self.personas[persona_id]
        actual_persona = persona.get("actual_persona", "")
        
        prompt = f"""You are evaluating how well a debate participant adheres to their assigned persona.

ASSIGNED PERSONA:
{actual_persona}

TEXT TO EVALUATE ({turn_label}):
{text}

On a scale of 1-5, how well does this text match the assigned persona?

1 = Not at all - completely inconsistent with the persona
2 = Poorly - mostly inconsistent, with only minor alignment
3 = Moderately - some consistency, but significant deviations
4 = Well - mostly consistent, with minor deviations
5 = Perfectly - completely consistent and authentic to the persona

Consider:
- Does the language and tone match what this persona would use?
- Are the arguments and perspectives consistent with this persona's background and interests?
- Does the content reflect the constraints, obligations, and stakes mentioned in the persona?
- Is the level of detail and specificity appropriate for this persona?

Respond with ONLY a single number from 1 to 5, nothing else."""
        
        return prompt
    
    def _score_text(
        self,
        text: str,
        persona_id: str,
        turn_label: str = "turn",
    ) -> int:
        """
        Score a piece of text against a persona.
        
        Args:
            text: The text to evaluate
            persona_id: ID of the persona to evaluate against
            turn_label: Label describing what's being evaluated
        
        Returns:
            Score from 1-5
        """
        if not text or not text.strip():
            return 3  # Neutral score for empty text
        
        prompt = self._create_evaluation_prompt(text, persona_id, turn_label)
        
        try:
            response = self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
            )
            
            # Extract the score from the response
            response_text = response.strip()
            
            # Try to find a number in the response
            import re
            numbers = re.findall(r'\b[1-5]\b', response_text)
            if numbers:
                return int(numbers[0])
            
            # If no valid number found, return neutral score
            print(f"Warning: Could not parse score from response: {response_text}")
            return 3
            
        except Exception as e:
            print(f"Error scoring text: {e}")
            return 3  # Return neutral score on error
    
    def evaluate_debate(
        self,
        debate_data: Dict,
        alpha_persona_id: str,
        beta_persona_id: str,
        verbose: bool = False,
    ) -> DebatePersonaEvaluation:
        """
        Evaluate both agents' persona adherence throughout the debate.
        
        Args:
            debate_data: Structured debate data (from get_structured_debate_history)
            alpha_persona_id: Persona ID for Alpha agent
            beta_persona_id: Persona ID for Beta agent
            verbose: If True, print progress information
        
        Returns:
            Complete evaluation for both agents
        """
        # Get agent names from debate_data
        agent_names = list(debate_data.keys())
        if len(agent_names) != 2:
            raise ValueError("Expected exactly 2 agents in debate data")
        
        alpha_name, beta_name = agent_names[0], agent_names[1]
        
        # Evaluate Alpha
        if verbose:
            print(f"Evaluating {alpha_name} against persona: {alpha_persona_id}")
        alpha_eval = self._evaluate_agent(
            debate_data[alpha_name],
            alpha_persona_id,
            verbose=verbose,
        )
        
        # Evaluate Beta
        if verbose:
            print(f"\nEvaluating {beta_name} against persona: {beta_persona_id}")
        beta_eval = self._evaluate_agent(
            debate_data[beta_name],
            beta_persona_id,
            verbose=verbose,
        )
        
        return DebatePersonaEvaluation(alpha=alpha_eval, beta=beta_eval)
    
    def _evaluate_agent(
        self,
        agent_data: Dict,
        persona_id: str,
        verbose: bool = False,
    ) -> AgentPersonaEvaluation:
        """
        Evaluate a single agent's persona adherence.
        
        Args:
            agent_data: Agent's debate data
            persona_id: Persona ID to evaluate against
            verbose: If True, print progress information
        
        Returns:
            Complete evaluation for the agent
        """
        evaluation = AgentPersonaEvaluation(persona_id=persona_id)
        
        debate_turns = agent_data.get("debate_turns", [])
        if not debate_turns:
            return evaluation
        
        # Accumulate text for cumulative scoring
        public_cumulative = []
        private_cumulative = []
        
        # Score each turn individually and cumulatively
        for turn_idx, turn_data in enumerate(debate_turns):
            turn_num = turn_idx + 1  # 1-indexed for output
            
            public_speech = turn_data.get("public_speech", "")
            private_reflection = turn_data.get("private_reflection", "")
            
            # Individual turn scores
            if verbose:
                print(f"  Turn {turn_num}: Scoring individual public speech...")
            public_score = self._score_text(
                public_speech,
                persona_id,
                turn_label=f"public turn {turn_num}",
            )
            evaluation.public_turn_scores.append(
                PersonaScore(turn_num=turn_num, score=public_score)
            )
            
            if verbose:
                print(f"  Turn {turn_num}: Scoring individual private reflection...")
            private_score = self._score_text(
                private_reflection,
                persona_id,
                turn_label=f"private turn {turn_num}",
            )
            evaluation.private_turn_scores.append(
                PersonaScore(turn_num=turn_num, score=private_score)
            )
            
            # Cumulative scoring
            public_cumulative.append(public_speech)
            private_cumulative.append(private_reflection)
            
            cumulative_public_text = "\n\n---\n\n".join(public_cumulative)
            cumulative_private_text = "\n\n---\n\n".join(private_cumulative)
            
            if verbose:
                print(f"  Turn {turn_num}: Scoring cumulative public (turns 1-{turn_num})...")
            cumulative_public_score = self._score_text(
                cumulative_public_text,
                persona_id,
                turn_label=f"cumulative public turns 1-{turn_num}",
            )
            evaluation.public_cumulative_scores.append(
                PersonaScore(turn_num=turn_num, score=cumulative_public_score)
            )
            
            if verbose:
                print(f"  Turn {turn_num}: Scoring cumulative private (turns 1-{turn_num})...")
            cumulative_private_score = self._score_text(
                cumulative_private_text,
                persona_id,
                turn_label=f"cumulative private turns 1-{turn_num}",
            )
            evaluation.private_cumulative_scores.append(
                PersonaScore(turn_num=turn_num, score=cumulative_private_score)
            )
        
        # Full debate scores (should match last cumulative scores)
        if evaluation.public_cumulative_scores:
            evaluation.full_debate_public_score = evaluation.public_cumulative_scores[-1].score
        if evaluation.private_cumulative_scores:
            evaluation.full_debate_private_score = evaluation.private_cumulative_scores[-1].score
        
        return evaluation


def get_structured_debate_history(memory_turns: List) -> Dict:
    """
    Convert raw memory turns into structured debate data.
    
    This function organizes the debate history by agent and separates
    public speeches from private reflections.
    
    Args:
        memory_turns: List of MemoryTurn objects from Agora.history()
    
    Returns:
        Dictionary mapping agent names to their debate data:
        {
            'agent_name': {
                'debate_turns': [
                    {
                        'turn_num': int,
                        'public_speech': str,
                        'private_reflection': str,
                        'public_stance': str,  # extracted metadata if available
                    },
                    ...
                ],
                'pre_interview': str or None,
                'post_interview': str or None,
            },
            ...
        }
    """
    from .memory import MemoryTurn
    
    # Group turns by agent
    agent_data = {}
    
    # Track current turn number for each agent
    agent_turn_nums = {}
    
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
        
        elif turn.role == "post_interview":
            agent_data[speaker_name]["post_interview"] = turn.private_reflection
        
        elif turn.role == "assistant":
            # This is a public speech
            # Check if there's a pending private reflection for this turn
            agent_turn_nums[speaker_name] += 1
            turn_num = agent_turn_nums[speaker_name]
            
            turn_data = {
                "turn_num": turn_num,
                "public_speech": turn.public_speech or "",
                "private_reflection": "",  # Will be filled by next reflection
                "public_stance": "",  # Can extract from public_speech if needed
            }
            agent_data[speaker_name]["debate_turns"].append(turn_data)
        
        elif turn.role == "reflection":
            # This is a private reflection - add to the most recent turn
            if agent_data[speaker_name]["debate_turns"]:
                # Add to next turn (reflection comes before public speech)
                agent_data[speaker_name]["debate_turns"][-1]["private_reflection"] = turn.private_reflection or ""
            else:
                # First reflection before any public speech - will be associated with next turn
                pass
    
    return agent_data


__all__ = [
    "PersonaEvaluator",
    "PersonaScore",
    "AgentPersonaEvaluation",
    "DebatePersonaEvaluation",
    "get_structured_debate_history",
]
