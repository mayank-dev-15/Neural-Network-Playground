"""
Browser-based visualizations for neural network architecture and training.
"""
import math
import json
from typing import Dict, List, Optional, Any


class NetworkVisualizer:
    """Generate visualization data for neural network architecture rendering."""

    @staticmethod
    def compute_layout(nodes: List[Dict], width: int = 1200, height: int = 800) -> List[Dict]:
        """Compute positions for each node in the graph using layered layout."""
        layers = {}
        for node in nodes:
            layer_idx = node.get("layer", 0)
            if layer_idx not in layers:
                layers[layer_idx] = []
            layers[layer_idx].append(node)

        num_layers = len(layers)
        layer_width = width / (num_layers + 1) if num_layers > 0 else width

        layout = []
        for idx in sorted(layers.keys()):
            layer_nodes = layers[idx]
            x = (idx + 1) * layer_width
            spacing = height / (len(layer_nodes) + 1)
            for j, node in enumerate(layer_nodes):
                y = (j + 1) * spacing
                layout.append({
                    "id": node["id"],
                    "x": x,
                    "y": y,
                    "label": node.get("name", node["id"]),
                    "type": node["type"],
                    "activation": node.get("activation", ""),
                })
        return layout

    @staticmethod
    def generate_weight_heatmap(weights: List[List[float]]) -> Dict:
        """Generate heatmap data from weight matrix."""
        w_array = weights
        w_min = min(min(row) for row in w_array) if w_array else 0
        w_max = max(max(row) for row in w_array) if w_array else 0
        return {
            "data": w_array,
            "min": w_min,
            "max": w_max,
            "width": len(w_array[0]) if w_array else 0,
            "height": len(w_array),
        }

    @staticmethod
    def generate_metrics_chart(metrics: List[Dict]) -> Dict:
        """Generate chart data for training metrics."""
        return {
            "train_loss": [m.get("train_loss", 0) for m in metrics],
            "val_loss": [m.get("val_loss", 0) for m in metrics],
            "train_accuracy": [m.get("train_accuracy", 0) for m in metrics],
            "val_accuracy": [m.get("val_accuracy", 0) for m in metrics],
            "epochs": list(range(1, len(metrics) + 1)),
        }

    @staticmethod
    def gradient_flow(weights: List[List[float]], grads: List[List[float]]) -> Dict:
        """Create gradient flow visualization data."""
        flow = []
        for i in range(len(weights)):
            row = []
            for j in range(len(weights[i])):
                magnitude = abs(grads[i][j]) if grads and i < len(grads) and j < len(grads[i]) else 0
                direction = 1 if grads[i][j] > 0 else -1 if grads[i][j] < 0 else 0
                row.append({"w": weights[i][j], "g": magnitude, "d": direction})
            flow.append(row)
        return {
            "gradients": flow,
            "max_gradient": max(
                abs(g) for r in grads for g in r
            ) if grads else 0,
        }
