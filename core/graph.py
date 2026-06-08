"""
Computation graph for building and visualizing neural networks.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json


@dataclass
class LayerConfig:
    """Configuration for a single neural network layer."""
    layer_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    activation: Optional[str] = None
    name: Optional[str] = None


class LayerNode:
    """Node in the computation graph representing one layer."""

    def __init__(self, config: LayerConfig, node_id: Optional[str] = None):
        self.id = node_id or f"layer_{id(self)}"
        self.config = config
        self.inputs: List[str] = []
        self.outputs: List[str] = []

    def add_input(self, node_id: str):
        if node_id not in self.inputs:
            self.inputs.append(node_id)

    def add_output(self, node_id: str):
        if node_id not in self.outputs:
            self.outputs.append(node_id)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.config.layer_type,
            "params": self.config.params,
            "activation": self.config.activation,
            "name": self.config.name or self.id,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


class ComputationGraph:
    """Directed acyclic graph representing a neural network architecture."""

    def __init__(self):
        self.nodes: Dict[str, LayerNode] = {}
        self.input_node: Optional[str] = None
        self.output_node: Optional[str] = None

    def add_node(self, node: LayerNode) -> str:
        self.nodes[node.id] = node
        return node.id

    def connect(self, from_id: str, to_id: str):
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError(f"Node not found: {from_id} or {to_id}")
        self.nodes[from_id].add_output(to_id)
        self.nodes[to_id].add_input(from_id)

    def set_input(self, node_id: str):
        if node_id not in self.nodes:
            raise ValueError(f"Node not found: {node_id}")
        self.input_node = node_id

    def set_output(self, node_id: str):
        if node_id not in self.nodes:
            raise ValueError(f"Node not found: {node_id}")
        self.output_node = node_id

    def remove_node(self, node_id: str):
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        for inp in node.inputs:
            if inp in self.nodes:
                self.nodes[inp].outputs.remove(node_id)
        for out in node.outputs:
            if out in self.nodes:
                self.nodes[out].inputs.remove(node_id)
        del self.nodes[node_id]

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order (forward pass order)."""
        in_degree = {nid: len(node.inputs) for nid, node in self.nodes.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        ordered = []
        while queue:
            nid = queue.pop(0)
            ordered.append(nid)
            for out_nid in self.nodes[nid].outputs:
                in_degree[out_nid] -= 1
                if in_degree[out_nid] == 0:
                    queue.append(out_nid)
        return ordered

    def get_param_count(self) -> int:
        """Estimate total trainable parameters."""
        total = 0
        for node in self.nodes.values():
            cfg = node.config
            if cfg.layer_type == "Dense":
                in_f = cfg.params.get("input_dim", 0)
                out_f = cfg.params.get("units", 0)
                total += in_f * out_f + out_f
            elif cfg.layer_type in ("Conv2D", "Conv1D"):
                in_ch = cfg.params.get("input_channels", 0)
                out_ch = cfg.params.get("filters", 0)
                k = cfg.params.get("kernel_size", 1)
                total += in_ch * out_ch * k + out_ch
        return total

    def to_json(self) -> str:
        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "input_node": self.input_node,
            "output_node": self.output_node,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ComputationGraph":
        data = json.loads(json_str)
        graph = cls()
        for nd in data["nodes"]:
            config = LayerConfig(
                layer_type=nd["type"],
                params=nd.get("params", {}),
                activation=nd.get("activation"),
                name=nd.get("name"),
            )
            node = LayerNode(config, nd["id"])
            node.inputs = nd.get("inputs", [])
            node.outputs = nd.get("outputs", [])
            graph.nodes[nd["id"]] = node
        graph.input_node = data.get("input_node")
        graph.output_node = data.get("output_node")
        return graph

    @classmethod
    def mlp(cls, input_dim: int, hidden_dims: List[int], output_dim: int) -> "ComputationGraph":
        """Create a standard MLP architecture."""
        graph = cls()
        prev = None
        for i, h_dim in enumerate(hidden_dims):
            config = LayerConfig(
                layer_type="Dense",
                params={"input_dim": input_dim if i == 0 else hidden_dims[i - 1], "units": h_dim},
                activation="relu",
                name=f"Hidden_{i + 1}",
            )
            node = LayerNode(config, f"hidden_{i + 1}")
            graph.add_node(node)
            if prev:
                graph.connect(prev, node.id)
            else:
                graph.set_input(node.id)
            prev = node.id

        out_config = LayerConfig(
            layer_type="Dense",
            params={"input_dim": hidden_dims[-1] if hidden_dims else input_dim, "units": output_dim},
            activation="softmax",
            name="Output",
        )
        out_node = LayerNode(out_config, "output")
        graph.add_node(out_node)
        graph.set_output(out_node.id)
        if prev:
            graph.connect(prev, out_node.id)
        return graph
