import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.graph import ComputationGraph, LayerNode, LayerConfig


def test_empty_graph():
    g = ComputationGraph()
    assert len(g.nodes) == 0
    assert g.input_node is None
    assert g.output_node is None


def test_add_node():
    g = ComputationGraph()
    config = LayerConfig(layer_type="input", params={"shape": [10]})
    node = LayerNode(config, "input")
    g.add_node(node)
    assert len(g.nodes) == 1
    assert "input" in g.nodes


def test_connect():
    g = ComputationGraph()
    a = LayerNode(LayerConfig(layer_type="input"), "a")
    b = LayerNode(LayerConfig(layer_type="Dense"), "b")
    g.add_node(a)
    g.add_node(b)
    g.connect("a", "b")
    assert "b" in g.nodes["a"].outputs
    assert "a" in g.nodes["b"].inputs


def test_topological_sort():
    g = ComputationGraph()
    a = LayerNode(LayerConfig(layer_type="input"), "a")
    b = LayerNode(LayerConfig(layer_type="Dense"), "b")
    c = LayerNode(LayerConfig(layer_type="Dense"), "c")
    g.add_node(a)
    g.add_node(b)
    g.add_node(c)
    g.connect("a", "b")
    g.connect("b", "c")
    g.set_input("a")
    g.set_output("c")
    order = g.topological_sort()
    assert order.index("a") < order.index("b") < order.index("c")


def test_mlp_factory():
    g = ComputationGraph.mlp(5, [10, 10], 2)
    assert len(g.nodes) == 4  # hidden_1, hidden_2, output + implicit
    assert g.input_node is not None
    assert g.output_node is not None


def test_serialization():
    g = ComputationGraph.mlp(3, [8, 4], 1)
    json_str = g.to_json()
    g2 = ComputationGraph.from_json(json_str)
    assert len(g2.nodes) == len(g.nodes)
    assert g2.input_node == g.input_node
    assert g2.output_node == g.output_node


def test_remove_node():
    g = ComputationGraph()
    a = LayerNode(LayerConfig(layer_type="input"), "a")
    b = LayerNode(LayerConfig(layer_type="Dense"), "b")
    g.add_node(a)
    g.add_node(b)
    g.connect("a", "b")
    g.remove_node("b")
    assert "b" not in g.nodes
    assert "b" not in g.nodes["a"].outputs


if __name__ == '__main__':
    test_empty_graph()
    test_add_node()
    test_connect()
    test_topological_sort()
    test_mlp_factory()
    test_serialization()
    test_remove_node()
    print("All graph tests passed.")
