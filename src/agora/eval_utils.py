import torch
from sentence_transformers import SentenceTransformer, util

def get_structured_debate_history(memory_turns):
    """Parse a list of MemoryTurn objects into a structured debate format organized by agent."""
    
    def extract_public_stance(text):
        """Extract public stance metrics from text."""
        if not text:
            return text, None
        
        lines = text.strip().split('\n')
        stance_metrics = {}
        content_lines = []
        
        for line in lines:
            if line.strip().startswith('Public_'):
                # Parse the metric
                parts = line.strip().split('=')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    try:
                        stance_metrics[key] = int(value)
                    except ValueError:
                        stance_metrics[key] = value
            else:
                content_lines.append(line)
        
        cleaned_text = '\n'.join(content_lines).strip()
        return cleaned_text, stance_metrics if stance_metrics else None
    
    debate_data = {}
    
    # Get unique agents
    agents = {}
    for turn in memory_turns:
        speaker_name = turn.metadata.get('speaker_name', turn.speaker_id)
        if speaker_name not in agents:
            agents[speaker_name] = turn.speaker_id
    
    # Initialize structure for each agent
    for speaker_name in agents:
        debate_data[speaker_name] = {
            'pre_interview': None,
            'debate_turns': [],
            'post_interview': None
        }
    
    # Process pre-interview
    for turn in memory_turns:
        if turn.role == 'pre_interview':
            speaker_name = turn.metadata.get('speaker_name', turn.speaker_id)
            debate_data[speaker_name]['pre_interview'] = {
                'private_reflection': turn.private_reflection
            }
    
    # Process post-interview
    for turn in memory_turns:
        if turn.role == 'post_interview':
            speaker_name = turn.metadata.get('speaker_name', turn.speaker_id)
            debate_data[speaker_name]['post_interview'] = {
                'private_reflection': turn.private_reflection
            }
    
    # Process core debate turns
    core_turns = [t for t in memory_turns if t.role in ['assistant', 'reflection']]
    
    # Track current reflection for each speaker
    current_reflections = {}
    
    for turn in core_turns:
        speaker_name = turn.metadata.get('speaker_name', turn.speaker_id)
        
        if turn.role == 'reflection':
            # Store reflection, waiting for the corresponding speech
            current_reflections[speaker_name] = turn.private_reflection
        
        elif turn.role == 'assistant':
            # Extract public stance from public speech
            cleaned_speech, public_stance = extract_public_stance(turn.public_speech)
            
            # Create turn entry with reflection (if exists) and speech
            turn_entry = {
                'private_reflection': current_reflections.get(speaker_name),
                'public_speech': cleaned_speech,
                'public_stance': public_stance
            }
            debate_data[speaker_name]['debate_turns'].append(turn_entry)
            
            # Clear the used reflection
            if speaker_name in current_reflections:
                del current_reflections[speaker_name]

    # Set the first turn's reflection from pre-interview if missing
    for speaker_name in debate_data:
        if debate_data[speaker_name]['debate_turns'][0]['private_reflection'] is None:
            debate_data[speaker_name]['debate_turns'][0]['private_reflection'] = debate_data[speaker_name]['pre_interview']['private_reflection']
    
    return debate_data


def calculate_narrative_similarity(text1, text2, model=None):
    """
    Calculate cosine similarity between two pieces of text using sentence embeddings.
    
    Args:
        text1: First text string
        text2: Second text string
        model: Optional SentenceTransformer model. If None, loads 'all-mpnet-base-v2'
    
    Returns:
        float: Cosine similarity score between the two texts
    """
    # Load model if not provided
    if model is None:
        model = SentenceTransformer('all-mpnet-base-v2')
    
    # Encode the texts to get their embeddings
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_score = util.cos_sim(embedding1, embedding2)
    
    # Extract the float value from the tensor
    score_val = cosine_score.item()
    
    return score_val


def compute_intra_agent_honesty(
    debate_data,
    similarity_fn=calculate_narrative_similarity,
):
    """
    Returns per-agent dict with turn indices and scores.
    """
    return {
        speaker_name: {
            "turns": list(range(len(speaker_data['debate_turns']))),
            "scores": [
                similarity_fn(
                    turn['private_reflection'],
                    turn['public_speech'],
                )
                for turn in speaker_data['debate_turns']
            ],
        }
        for speaker_name, speaker_data in debate_data.items()
    }


def compute_inter_agent_alignment(
    debate_data,
    agent_a_narrative="public_speech",
    agent_b_narrative="public_speech",
    similarity_fn=calculate_narrative_similarity,
):
    """
    Computes turn-aligned similarity between the first two agents in debate_data,
    using specified narrative fields for each agent.

    Args:
        debate_data (dict): Debate data containing exactly two agents.
        agent_a_narrative (str): Narrative field for agent A.
        agent_b_narrative (str): Narrative field for agent B.
        similarity_fn (callable): Similarity function.

    Returns:
        dict with:
            - 'turns': list[int]
            - 'scores': list[float]
    """
    agent_ids = list(debate_data.keys())
    if len(agent_ids) < 2:
        raise ValueError("compute_inter_agent_alignment requires at least two agents")

    agent_a, agent_b = agent_ids[:2]

    turns_a = debate_data[agent_a]["debate_turns"]
    turns_b = debate_data[agent_b]["debate_turns"]

    num_turns = min(len(turns_a), len(turns_b))
    turns = list(range(num_turns))

    scores = [
        similarity_fn(
            turns_a[t][agent_a_narrative],
            turns_b[t][agent_b_narrative],
        )
        for t in turns
    ]

    return {
        "turns": turns,
        "scores": scores,
    }
