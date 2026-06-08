import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.graph import ComputationGraph


def test_empty_graph():
    g = ComputationGraph()
    assert len(g.nodes) == 0
    assert len(g.edges) == 0


def test_add_node():
    g = ComputationGraph()
    g.add_node('input', 'input', {'shape': [10]})
    assert len(g.nodes) == 1
    assert g.nodes[0].id == 'input'


def test_add_edge():
    g = ComputationGraph()
    g.add_node('a', 'input', {})
    g.add_node('b', 'linear', {'in': 10, 'out': 5})
    g.add_edge('a', 'b')
    assert ('a', 'b') in g.edges


def test_topological_sort():
    g = ComputationGraph()
    g.add_node('a', 'input', {})
    g.add_node('b', 'linear', {'in': 10, 'out': 5})
    g.add_node('c', 'output', {})
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')
    g.topological_sort()
    assert g.nodes[0].id == 'a'
    assert g.nodes[2].id == 'c'


def test_mlp_factory():
    g = ComputationGraph.create_mlp(5, [10, 10], 2)
    assert len(g.nodes) == 6
    assert g.param_count() > 0


def test_serialization():
    g = ComputationGraph.create_mlp(3, [8, 4], 1)
    d = g.to_dict()
    assert len(d['nodes']) == 6
    assert len(d['edges']) == 5
    g2 = ComputationGraph.from_dict(d)
    assert len(g2.nodes) == len(g.nodes)


if __name__ == '__main__':
    test_empty_graph()
    test_add_node()
    test_add_edge()
    test_topological_sort()
    test_mlp_factory()
    test_serialization()
    print("All graph tests passed.")
