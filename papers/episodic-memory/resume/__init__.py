"""
ReSuME: Representational Surprise-based Memory for LLM Agents
"""

from .agent import ReSuMEAgent
from .surprise_detector import SurpriseDetector
from .episodic_memory import EpisodicMemory
from .sae_loader import load_pretrained_sae

__version__ = "0.1.0"

__all__ = [
    "ReSuMEAgent",
    "SurpriseDetector",
    "EpisodicMemory",
    "load_pretrained_sae",
]
