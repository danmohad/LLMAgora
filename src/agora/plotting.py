"""Plotting utilities for visualizing debate metrics."""

import re
from typing import Dict, List, Any, Optional, Tuple

import matplotlib.pyplot as plt


def extract_metrics(text: str) -> Dict[str, int]:
    """Extract metrics from a response text. Returns dict with metric values."""
    if not text:
        return {}
    
    metrics = {}
    # Patterns for each metric - handles formats like:
    # Public_STANCE_SHIFT={{+1}}, {{-1}}, or =-1
    patterns = {
        'STANCE_SHIFT': r'(?:Public_|Off_Record_|Private_)STANCE_SHIFT\s*=\s*\{?\{?([+-]?\d+)\}?\}?',
        'CONFIDENCE': r'(?:Public_|Off_Record_|Private_)CONFIDENCE\s*=\s*\{?\{?(\d+)\}?\}?',
        'RESPECT': r'(?:Public_|Off_Record_|Private_)RESPECT\s*=\s*\{?\{?(\d+)\}?\}?',
        'INTEREST': r'(?:Public_|Off_Record_|Private_)INTEREST_IN_OPPONENT_RESPONSE\s*=\s*\{?\{?(\d+)\}?\}?',
        'TENSION': r'(?:Public_|Off_Record_|Private_|Public_Record_)TENSION(?:_WITH_OPPONENT_RESPONSE)?\s*=\s*\{?\{?(\d+)\}?\}?',
    }
    
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[metric_name] = int(match.group(1))
    
    return metrics


def collect_agent_metrics(agora) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Collect metrics from the Agora's full turn log.
    Uses the complete history including private reflections.
    
    Returns: dict[agent_name] -> {'public': [...], 'off_record': [...]}
    """
    agent_metrics = {}
    
    # Get all turns from the agora's turn log
    all_turns = agora.history()
    
    # Group turns by agent
    agent_names = set()
    for turn in all_turns:
        speaker = turn.metadata.get('speaker_name', turn.speaker_id)
        agent_names.add(speaker)
    
    for agent_name in agent_names:
        agent_metrics[agent_name] = {
            'public': [],
            'off_record': []
        }
        
        # Get this agent's turns in order
        agent_turns = [t for t in all_turns if t.metadata.get('speaker_name', t.speaker_id) == agent_name]
        
        # Pair reflections with their following public responses
        round_num = 0
        pending_reflection = None
        
        for turn in agent_turns:
            if turn.role == 'reflection' and turn.private_reflection:
                pending_reflection = turn
            elif turn.role == 'assistant' and turn.public_speech:
                round_num += 1
                
                # Extract public metrics
                public_metrics = extract_metrics(turn.public_speech)
                if public_metrics:
                    public_metrics['round'] = round_num
                    agent_metrics[agent_name]['public'].append(public_metrics)
                
                # Extract off_record metrics from the preceding reflection
                if pending_reflection:
                    off_record_metrics = extract_metrics(pending_reflection.private_reflection)
                    if off_record_metrics:
                        off_record_metrics['round'] = round_num
                        agent_metrics[agent_name]['off_record'].append(off_record_metrics)
                    pending_reflection = None
    
    return agent_metrics


def plot_metrics(
    agent_metrics: Dict[str, Dict[str, List[Dict[str, Any]]]],
    agent_to_persona: Optional[Dict[str, str]] = None,
    persona_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (16, 10),
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a 2x3 subplot - one plot per metric.
    Each plot has 4 lines: 2 agents × (public solid + off_record dashed).
    
    Args:
        agent_metrics: Dict from collect_agent_metrics()
        agent_to_persona: Optional mapping of agent names to persona IDs for labels
        persona_colors: Optional mapping of persona IDs to colors
        figsize: Figure size tuple
    
    Returns:
        Tuple of (figure, axes)
    """
    metric_names = ['STANCE_SHIFT', 'CONFIDENCE', 'RESPECT', 'INTEREST', 'TENSION']
    metric_labels = ['Stance Shift', 'Confidence', 'Respect', 'Interest in Opponent', 'Tension']
    
    # Default mappings if not provided
    if agent_to_persona is None:
        agent_to_persona = {}
    if persona_colors is None:
        persona_colors = {}
    
    # Build color mapping: agent_name -> color
    default_colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    agent_colors = {}
    for i, agent_name in enumerate(agent_metrics.keys()):
        persona_id = agent_to_persona.get(agent_name, agent_name)
        if persona_id in persona_colors:
            agent_colors[agent_name] = persona_colors[persona_id]
        else:
            agent_colors[agent_name] = default_colors[i % len(default_colors)]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Find max rounds across all agents
    max_rounds = 0
    for agent_name, data in agent_metrics.items():
        for entry in data['public']:
            max_rounds = max(max_rounds, entry.get('round', 0))
    
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        
        for agent_name, data in agent_metrics.items():
            color = agent_colors.get(agent_name, '#3498db')
            display_name = agent_to_persona.get(agent_name, agent_name)
            
            # Plot public metrics (solid line with circles)
            public_data = data['public']
            if public_data:
                rounds = [d['round'] for d in public_data if metric in d]
                values = [d[metric] for d in public_data if metric in d]
                if rounds:
                    ax.plot(rounds, values, 'o-', color=color, linewidth=2, 
                            label=f'{display_name} (Public)', markersize=7)
            
            # Plot off-record metrics (dashed line with squares)
            off_record_data = data['off_record']
            if off_record_data:
                rounds = [d['round'] for d in off_record_data if metric in d]
                values = [d[metric] for d in off_record_data if metric in d]
                if rounds:
                    ax.plot(rounds, values, 's--', color=color, linewidth=2, 
                            label=f'{display_name} (Off-Record)', markersize=7, alpha=0.7)
        
        ax.set_xlabel('Round')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Over Rounds')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        if max_rounds > 0:
            ax.set_xticks(range(1, max_rounds + 1))
        
        # Set y-axis limits based on metric type
        if metric == 'STANCE_SHIFT':
            ax.set_ylim(-5, 5)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        else:
            ax.set_ylim(0, 105)
    
    # Hide the 6th subplot (empty)
    axes[5].set_visible(False)
    
    plt.tight_layout()
    return fig, axes
