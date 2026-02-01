"""Refactored Persona adherence evaluator for debate agents."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re
import numpy as np


@dataclass
class PersonaScore:
    """Score for a single turn or cumulative turns with multiple evaluations."""
    turn_num: int
    scores_raw: List[int]  # All raw scores from multiple evaluations
    
    @property
    def score_mean(self) -> float:
        """Mean of all evaluations."""
        return float(np.mean(self.scores_raw))
    
    @property
    def score_std(self) -> float:
        """Standard deviation of all evaluations."""
        return float(np.std(self.scores_raw))


@dataclass
class AgentPersonaEvaluation:
    """Complete persona evaluation for one agent."""
    persona_id: str
    public_turn_scores: List[PersonaScore] = field(default_factory=list)
    private_turn_scores: List[PersonaScore] = field(default_factory=list)
    public_cumulative_scores: List[PersonaScore] = field(default_factory=list)
    private_cumulative_scores: List[PersonaScore] = field(default_factory=list)
    full_debate_public_score: Optional[Tuple[float, float]] = None  # (mean, std)
    full_debate_private_score: Optional[Tuple[float, float]] = None  # (mean, std)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format with separate turns and scores."""
        def scores_to_dict(scores: List[PersonaScore]) -> Dict:
            """Convert list of PersonaScore to dict format."""
            return {
                "turns": [s.turn_num for s in scores],
                "scores": {
                    "mean": [s.score_mean for s in scores],
                    "std": [s.score_std for s in scores],
                    "raw": [s.scores_raw for s in scores],
                }
            }
        
        return {
            "persona_id": self.persona_id,
            "public_turn_scores": scores_to_dict(self.public_turn_scores),
            "private_turn_scores": scores_to_dict(self.private_turn_scores),
            "public_cumulative_scores": scores_to_dict(self.public_cumulative_scores),
            "private_cumulative_scores": scores_to_dict(self.private_cumulative_scores),
            "full_debate_public_score": {
                "mean": self.full_debate_public_score[0] if self.full_debate_public_score else None,
                "std": self.full_debate_public_score[1] if self.full_debate_public_score else None,
            },
            "full_debate_private_score": {
                "mean": self.full_debate_private_score[0] if self.full_debate_private_score else None,
                "std": self.full_debate_private_score[1] if self.full_debate_private_score else None,
            },
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
        llm_client,
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
        # Handle both nested and flat persona structures
        if "personas" in personas:
            self.personas = personas["personas"]
        else:
            self.personas = personas
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
        n_samples: int = 1,
    ) -> List[int]:
        """
        Score a piece of text against a persona multiple times.
        
        Args:
            text: The text to evaluate
            persona_id: ID of the persona to evaluate against
            turn_label: Label describing what's being evaluated
            n_samples: Number of times to evaluate (for mean/std)
        
        Returns:
            List of scores from 1-5 (length = n_samples)
        """
        if not text or not text.strip():
            return [3] * n_samples  # Neutral score for empty text
        
        prompt = self._create_evaluation_prompt(text, persona_id, turn_label)
        
        scores = []
        for i in range(n_samples):
            try:
                response = self.llm_client.complete(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                )
                
                # Extract the score from the response
                response_text = response.strip()
                
                # Try to find a number in the response
                numbers = re.findall(r'\b[1-5]\b', response_text)
                if numbers:
                    scores.append(int(numbers[0]))
                else:
                    # If no valid number found, return neutral score
                    if i == 0:  # Only print warning once
                        print(f"Warning: Could not parse score from response: {response_text}")
                    scores.append(3)
                
            except Exception as e:
                if i == 0:  # Only print error once
                    print(f"Error scoring text: {e}")
                scores.append(3)  # Return neutral score on error
        
        return scores
    
    def evaluate_debate_from_history(
        self,
        memory_turns: List,
        alpha_persona_id: str,
        beta_persona_id: str,
        verbose: bool = False,
        n_samples: int = 1,
    ) -> DebatePersonaEvaluation:
        """
        Evaluate both agents' persona adherence from Agora memory turns.
        
        Args:
            memory_turns: List of MemoryTurn objects from Agora.history()
            alpha_persona_id: Persona ID for Alpha agent
            beta_persona_id: Persona ID for Beta agent
            verbose: If True, print progress information
            n_samples: Number of times to evaluate each text (for mean/std)
        
        Returns:
            Complete evaluation for both agents
        """
        # Convert memory turns to structured debate data
        debate_data = get_structured_debate_history(memory_turns)
        
        # Get agent names from debate_data
        agent_names = list(debate_data.keys())
        if len(agent_names) != 2:
            raise ValueError(f"Expected exactly 2 agents in debate data, got {len(agent_names)}")
        
        alpha_name, beta_name = agent_names[0], agent_names[1]
        
        # Evaluate Alpha
        if verbose:
            print(f"Evaluating {alpha_name} against persona: {alpha_persona_id}")
        alpha_eval = self._evaluate_agent(
            debate_data[alpha_name],
            alpha_persona_id,
            verbose=verbose,
            n_samples=n_samples,
        )
        
        # Evaluate Beta
        if verbose:
            print(f"\nEvaluating {beta_name} against persona: {beta_persona_id}")
        beta_eval = self._evaluate_agent(
            debate_data[beta_name],
            beta_persona_id,
            verbose=verbose,
            n_samples=n_samples,
        )
        
        return DebatePersonaEvaluation(alpha=alpha_eval, beta=beta_eval)
    
    def _evaluate_agent(
        self,
        agent_data: Dict,
        persona_id: str,
        verbose: bool = False,
        n_samples: int = 1,
    ) -> AgentPersonaEvaluation:
        """
        Evaluate a single agent's persona adherence.
        
        Args:
            agent_data: Agent's debate data
            persona_id: Persona ID to evaluate against
            verbose: If True, print progress information
            n_samples: Number of times to evaluate each text (for mean/std)
        
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
            public_scores = self._score_text(
                public_speech,
                persona_id,
                turn_label=f"public turn {turn_num}",
                n_samples=n_samples,
            )
            evaluation.public_turn_scores.append(
                PersonaScore(turn_num=turn_num, scores_raw=public_scores)
            )
            
            if verbose:
                print(f"  Turn {turn_num}: Scoring individual private reflection...")
            private_scores = self._score_text(
                private_reflection,
                persona_id,
                turn_label=f"private turn {turn_num}",
                n_samples=n_samples,
            )
            evaluation.private_turn_scores.append(
                PersonaScore(turn_num=turn_num, scores_raw=private_scores)
            )
            
            # Cumulative scoring
            public_cumulative.append(public_speech)
            private_cumulative.append(private_reflection)
            
            cumulative_public_text = "\n\n---\n\n".join(public_cumulative)
            cumulative_private_text = "\n\n---\n\n".join(private_cumulative)
            
            if verbose:
                print(f"  Turn {turn_num}: Scoring cumulative public (turns 1-{turn_num})...")
            cumulative_public_scores = self._score_text(
                cumulative_public_text,
                persona_id,
                turn_label=f"cumulative public turns 1-{turn_num}",
                n_samples=n_samples,
            )
            evaluation.public_cumulative_scores.append(
                PersonaScore(turn_num=turn_num, scores_raw=cumulative_public_scores)
            )
            
            if verbose:
                print(f"  Turn {turn_num}: Scoring cumulative private (turns 1-{turn_num})...")
            cumulative_private_scores = self._score_text(
                cumulative_private_text,
                persona_id,
                turn_label=f"cumulative private turns 1-{turn_num}",
                n_samples=n_samples,
            )
            evaluation.private_cumulative_scores.append(
                PersonaScore(turn_num=turn_num, scores_raw=cumulative_private_scores)
            )
        
        # Full debate scores (should match last cumulative scores)
        if evaluation.public_cumulative_scores:
            last_public = evaluation.public_cumulative_scores[-1]
            evaluation.full_debate_public_score = (last_public.score_mean, last_public.score_std)
        if evaluation.private_cumulative_scores:
            last_private = evaluation.private_cumulative_scores[-1]
            evaluation.full_debate_private_score = (last_private.score_mean, last_private.score_std)
        
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
            agent_turn_nums[speaker_name] += 1
            turn_num = agent_turn_nums[speaker_name]
            
            turn_data = {
                "turn_num": turn_num,
                "public_speech": turn.public_speech or "",
                "private_reflection": "",  # Will be filled by corresponding reflection
                "public_stance": "",
            }
            agent_data[speaker_name]["debate_turns"].append(turn_data)
        
        elif turn.role == "reflection":
            # This is a private reflection - add to the most recent turn
            if agent_data[speaker_name]["debate_turns"]:
                # Add to the last turn (reflection corresponds to the last public speech)
                agent_data[speaker_name]["debate_turns"][-1]["private_reflection"] = turn.private_reflection or ""
    
    return agent_data


def plot_persona_adherence(
    eval_dict: Dict,
    alpha_persona_name: str,
    beta_persona_name: str,
    save_path: str = None,
    show_plot: bool = True,
):
    """
    Plot persona adherence scores over time with error bars.
    
    Args:
        eval_dict: Dictionary from DebatePersonaEvaluation.to_dict()
        alpha_persona_name: Display name for alpha persona
        beta_persona_name: Display name for beta persona
        save_path: If provided, save plot to this path
        show_plot: If True, display the plot
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Persona Adherence Scores Over Time', fontsize=16)
    
    # Extract data
    alpha_data = eval_dict['alpha']
    beta_data = eval_dict['beta']
    
    # Define colors for each agent
    alpha_color = 'tab:blue'
    beta_color = 'tab:orange'
    
    # Left panel: Individual Turn Scores
    ax = axes[0]
    
    # Alpha Public (solid)
    alpha_pub_ind = alpha_data['public_turn_scores']
    ax.errorbar(
        alpha_pub_ind['turns'], 
        alpha_pub_ind['scores']['mean'],
        yerr=alpha_pub_ind['scores']['std'],
        marker='o', label=f'{alpha_persona_name} - Public', 
        linewidth=2, capsize=5, alpha=0.8,
        color=alpha_color, linestyle='-'
    )
    
    # Alpha Private (dashed)
    alpha_priv_ind = alpha_data['private_turn_scores']
    ax.errorbar(
        alpha_priv_ind['turns'], 
        alpha_priv_ind['scores']['mean'],
        yerr=alpha_priv_ind['scores']['std'],
        marker='o', label=f'{alpha_persona_name} - Private', 
        linewidth=2, capsize=5, alpha=0.8,
        color=alpha_color, linestyle='--'
    )
    
    # Beta Public (solid)
    beta_pub_ind = beta_data['public_turn_scores']
    ax.errorbar(
        beta_pub_ind['turns'], 
        beta_pub_ind['scores']['mean'],
        yerr=beta_pub_ind['scores']['std'],
        marker='s', label=f'{beta_persona_name} - Public', 
        linewidth=2, capsize=5, alpha=0.8,
        color=beta_color, linestyle='-'
    )
    
    # Beta Private (dashed)
    beta_priv_ind = beta_data['private_turn_scores']
    ax.errorbar(
        beta_priv_ind['turns'], 
        beta_priv_ind['scores']['mean'],
        yerr=beta_priv_ind['scores']['std'],
        marker='s', label=f'{beta_persona_name} - Private', 
        linewidth=2, capsize=5, alpha=0.8,
        color=beta_color, linestyle='--'
    )
    
    ax.set_title('Individual Turn Scores')
    ax.set_xlabel('Turn Number')
    ax.set_ylabel('Score (1-5)')
    ax.set_ylim(0.5, 5.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Right panel: Cumulative Scores
    ax = axes[1]
    
    # Alpha Public (solid)
    alpha_pub_cum = alpha_data['public_cumulative_scores']
    ax.errorbar(
        alpha_pub_cum['turns'], 
        alpha_pub_cum['scores']['mean'],
        yerr=alpha_pub_cum['scores']['std'],
        marker='o', label=f'{alpha_persona_name} - Public', 
        linewidth=2, capsize=5, alpha=0.8,
        color=alpha_color, linestyle='-'
    )
    
    # Alpha Private (dashed)
    alpha_priv_cum = alpha_data['private_cumulative_scores']
    ax.errorbar(
        alpha_priv_cum['turns'], 
        alpha_priv_cum['scores']['mean'],
        yerr=alpha_priv_cum['scores']['std'],
        marker='o', label=f'{alpha_persona_name} - Private', 
        linewidth=2, capsize=5, alpha=0.8,
        color=alpha_color, linestyle='--'
    )
    
    # Beta Public (solid)
    beta_pub_cum = beta_data['public_cumulative_scores']
    ax.errorbar(
        beta_pub_cum['turns'], 
        beta_pub_cum['scores']['mean'],
        yerr=beta_pub_cum['scores']['std'],
        marker='s', label=f'{beta_persona_name} - Public', 
        linewidth=2, capsize=5, alpha=0.8,
        color=beta_color, linestyle='-'
    )
    
    # Beta Private (dashed)
    beta_priv_cum = beta_data['private_cumulative_scores']
    ax.errorbar(
        beta_priv_cum['turns'], 
        beta_priv_cum['scores']['mean'],
        yerr=beta_priv_cum['scores']['std'],
        marker='s', label=f'{beta_persona_name} - Private', 
        linewidth=2, capsize=5, alpha=0.8,
        color=beta_color, linestyle='--'
    )
    
    ax.set_title('Cumulative Scores')
    ax.set_xlabel('Turn Number')
    ax.set_ylabel('Score (1-5)')
    ax.set_ylim(0.5, 5.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


__all__ = [
    "PersonaEvaluator",
    "PersonaScore",
    "AgentPersonaEvaluation",
    "DebatePersonaEvaluation",
    "get_structured_debate_history",
    "plot_persona_adherence",
]