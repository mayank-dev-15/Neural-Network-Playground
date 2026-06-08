import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.engine import TrainingEngine
from core.graph import ComputationGraph


def test_engine_init():
    engine = TrainingEngine(ComputationGraph())
    assert engine is not None


def test_engine_config():
    config = {'epochs': 5, 'lr': 0.01}
    engine = TrainingEngine(ComputationGraph(), **config)
    assert engine.config['epochs'] == 5


if __name__ == '__main__':
    test_engine_init()
    test_engine_config()
    print("All engine tests passed.")
