from typing import Any, Optional


class DebateAnalyzer:
    """Analyzer for debate metrics and visualizations."""
    
    def __init__(self, memory_turns, model_name='all-mpnet-base-v2'):
        """
        Initialize analyzer with debate history.
        
        Args:
            memory_turns: List of MemoryTurn objects or structured debate_data dict
            model_name: SentenceTransformer model name
        """
        # Handle both raw memory_turns and pre-structured debate_data
        if isinstance(memory_turns, dict):
            self.debate_data = memory_turns
        else:
            from agora.persona_evaluator import get_structured_debate_history
            self.debate_data = get_structured_debate_history(memory_turns)
        
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._intra_agent_honesty = None
        self._inter_agent_alignment = {}
        self._util: Optional[Any] = None
    
    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer, util
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentence-transformers is required for DebateAnalyzer. "
                "Install it to compute similarity metrics."
            ) from exc
        self._util = util
        print(f"Loading model: {self.model_name}...")
        return SentenceTransformer(self.model_name)
    
    def calculate_similarity(self, text1, text2):
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            float: Cosine similarity score
        """
        embedding1 = self.model.encode(text1, convert_to_tensor=True)
        embedding2 = self.model.encode(text2, convert_to_tensor=True)
        if self._util is None:
            self._model = self._load_model()
        cosine_score = self._util.cos_sim(embedding1, embedding2)
        return cosine_score.item()
    
    def compute_intra_agent_honesty(self, force_recompute=False):
        """
        Compute internal-external honesty (private vs public) for each agent.
        
        Args:
            force_recompute: If True, recompute even if cached
            
        Returns:
            dict: Per-agent honesty scores over turns
                {
                    'agent_name': {
                        'turns': [0, 1, 2, ...],
                        'scores': [0.85, 0.72, ...]
                    }
                }
        """
        if self._intra_agent_honesty is not None and not force_recompute:
            return self._intra_agent_honesty
        
        self._intra_agent_honesty = {
            speaker_name: {
                "turns": list(range(len(speaker_data['debate_turns']))),
                "scores": [
                    self.calculate_similarity(
                        turn['private_reflection'],
                        turn['public_speech'],
                    )
                    for turn in speaker_data['debate_turns']
                ],
            }
            for speaker_name, speaker_data in self.debate_data.items()
        }
        return self._intra_agent_honesty
    
    def compute_inter_agent_alignment(
        self,
        agent_a_narrative="public_speech",
        agent_b_narrative="public_speech",
        force_recompute=False,
    ):
        """
        Compute alignment between two agents' narratives.
        
        Args:
            agent_a_narrative: Narrative field for agent A ('public_speech' or 'private_reflection')
            agent_b_narrative: Narrative field for agent B ('public_speech' or 'private_reflection')
            force_recompute: If True, recompute even if cached
            
        Returns:
            dict: Alignment scores over turns
                {
                    'turns': [0, 1, 2, ...],
                    'scores': [0.45, 0.32, ...]
                }
        """
        cache_key = (agent_a_narrative, agent_b_narrative)
        
        if cache_key in self._inter_agent_alignment and not force_recompute:
            return self._inter_agent_alignment[cache_key]
        
        agent_ids = list(self.debate_data.keys())
        if len(agent_ids) < 2:
            raise ValueError("Requires at least two agents")
        
        agent_a, agent_b = agent_ids[:2]
        turns_a = self.debate_data[agent_a]["debate_turns"]
        turns_b = self.debate_data[agent_b]["debate_turns"]
        
        num_turns = min(len(turns_a), len(turns_b))
        turns = list(range(num_turns))
        
        scores = [
            self.calculate_similarity(
                turns_a[t][agent_a_narrative],
                turns_b[t][agent_b_narrative],
            )
            for t in turns
        ]
        
        result = {"turns": turns, "scores": scores}
        self._inter_agent_alignment[cache_key] = result
        return result
    
    def get_turn_content(self, agent_name, turn_idx, field='public_speech'):
        """
        Get specific turn content for an agent.
        
        Args:
            agent_name: Name of the agent
            turn_idx: Turn index (0-based)
            field: Field to retrieve ('public_speech', 'private_reflection', 'public_stance')
            
        Returns:
            Content of the specified field
        """
        return self.debate_data[agent_name]['debate_turns'][turn_idx][field]
    
    def get_agent_names(self):
        """Get list of agent names in the debate."""
        return list(self.debate_data.keys())
    
    def get_num_turns(self, agent_name=None):
        """
        Get number of turns.
        
        Args:
            agent_name: If specified, returns turns for that agent. 
                       If None, returns min turns across all agents.
        
        Returns:
            int: Number of turns
        """
        if agent_name:
            return len(self.debate_data[agent_name]['debate_turns'])
        else:
            return min(len(data['debate_turns']) for data in self.debate_data.values())
    
    def summary(self):
        """Print summary statistics of the debate."""
        print("=" * 60)
        print("DEBATE ANALYSIS SUMMARY")
        print("=" * 60)
        
        for agent_name, agent_data in self.debate_data.items():
            print(f"\n{agent_name}:")
            print(f"  Total turns: {len(agent_data['debate_turns'])}")
            print(f"  Has pre-interview: {agent_data['pre_interview'] is not None}")
            print(f"  Has post-interview: {agent_data['post_interview'] is not None}")
        
        if self._intra_agent_honesty:
            print("\n" + "-" * 60)
            print("INTRA-AGENT HONESTY (Private vs Public)")
            print("-" * 60)
            for agent_name, data in self._intra_agent_honesty.items():
                scores = data["scores"]
                print(f"{agent_name}:")
                print(f"  Mean: {sum(scores)/len(scores):.4f}")
                print(f"  Min:  {min(scores):.4f}")
                print(f"  Max:  {max(scores):.4f}")
        
        if self._inter_agent_alignment:
            print("\n" + "-" * 60)
            print("INTER-AGENT ALIGNMENT")
            print("-" * 60)
            for (narrative_a, narrative_b), data in self._inter_agent_alignment.items():
                scores = data["scores"]
                print(f"{narrative_a} vs {narrative_b}:")
                print(f"  Mean: {sum(scores)/len(scores):.4f}")
                print(f"  Min:  {min(scores):.4f}")
                print(f"  Max:  {max(scores):.4f}")
        
        print("\n" + "=" * 60)
