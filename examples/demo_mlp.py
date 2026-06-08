import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.architectures import MLPBuilder
from core.graph import ComputationGraph
from core.visualizer import NetworkVisualizer


def generate_data(n_samples=1000):
    X = torch.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] * X[:, 2] - X[:, 5] > 0).float()
    return X, y


def main():
    X, y = generate_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MLPBuilder.build(input_dim=10, hidden_dims=[64, 32], output_dim=1, activation='relu', dropout=0.2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training MLP classifier...")
    for epoch in range(10):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx).squeeze()
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc = ((model(bx).squeeze() > 0).float() == by).float().mean().item()
        print(f"  Epoch {epoch+1:2d}  loss={total_loss/len(loader):.4f}  acc={acc:.4f}")

    print("\nBuilding computation graph visualization...")
    graph = ComputationGraph()
    graph.add_node('input', 'input', {'shape': [10]})
    graph.add_node('fc1', 'linear', {'in': 10, 'out': 64})
    graph.add_node('relu1', 'activation', {'fn': 'relu'})
    graph.add_node('fc2', 'linear', {'in': 64, 'out': 32})
    graph.add_node('relu2', 'activation', {'fn': 'relu'})
    graph.add_node('fc3', 'linear', {'in': 32, 'out': 1})
    graph.add_node('output', 'output', {})

    graph.add_edge('input', 'fc1')
    graph.add_edge('fc1', 'relu1')
    graph.add_edge('relu1', 'fc2')
    graph.add_edge('fc2', 'relu2')
    graph.add_edge('relu2', 'fc3')
    graph.add_edge('fc3', 'output')

    graph.topological_sort()

    print(f"\nGraph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"Parameters: {graph.param_count():,}")

    viz = NetworkVisualizer(graph)
    layout = viz.layered_layout()
    print(f"Layout layers: {len(layout)}")

    torch.save(model.state_dict(), 'mlp_demo.pt')
    print("\nModel saved to mlp_demo.pt")
    print("Demo complete!")


if __name__ == '__main__':
    main()
