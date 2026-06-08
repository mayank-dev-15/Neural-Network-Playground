"""
Neural Network Playground - Core runtime and graph engine.
"""
from .graph import ComputationGraph, LayerNode
from .engine import TrainingEngine

__all__ = ["ComputationGraph", "LayerNode", "TrainingEngine"]
