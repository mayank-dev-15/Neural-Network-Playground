"""
Neural Network Playground - Core runtime and graph engine.
"""
from .graph import ComputationGraph, LayerNode

try:
    from .engine import TrainingEngine
except ImportError:
    pass

__all__ = ["ComputationGraph", "LayerNode", "TrainingEngine"]
