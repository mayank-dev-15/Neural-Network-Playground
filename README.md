# Neural Network Playground

Visual deep learning framework for education and experimentation. Build, train, and visualize neural networks in the browser with real-time feedback.

## Features

- **Visual Network Builder**: Drag-and-drop layers, connect nodes, configure parameters
- **Real-time Training Visualization**: Loss/accuracy curves, weight distributions, gradient flow
- **Interactive Forward/Backward Pass**: Step through computation graph, inspect tensors
- **Pre-built Architectures**: MLP, CNN, RNN/LSTM, Transformer, Autoencoder, GAN
- **Dataset Gallery**: MNIST, CIFAR-10, Fashion-MNIST, synthetic datasets
- **Export Models**: ONNX, PyTorch, TensorFlow Lite, Keras H5
- **Code View**: Auto-generated PyTorch/TensorFlow/JAX code
- **Collaboration**: Share models via URL, fork others' experiments

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Web UI         │────▶│  WASM Engine │◀───▶│  WebGPU/     │
│  (React +       │     │  (ONNX       │     │  WebGL       │
│  Cytoscape)     │     │  Runtime)    │     │  Backend     │
└─────────────────┘     └──────────────┘     └──────────────┘
         │                                              │
         │              ┌──────────────┐                │
         └─────────────▶│  Python API  │◀───────────────┘
                        │  (Training,  │
                        │  Export)     │
                        └──────────────┘
```

## Quick Start

### Web Only (No Backend)
```bash
cd web
npm install
npm run dev
# Open http://localhost:5173
```

### With Python Backend (Training + Export)
```bash
pip install -r requirements.txt
python -m api.server --host 0.0.0.0 --port 8080

# In another terminal
cd web && npm run dev
```

## Supported Layers

| Category | Layers |
|----------|--------|
| Core | Dense, Dropout, BatchNorm, LayerNorm |
| Convolutional | Conv1D/2D/3D, DepthwiseConv, TransposedConv |
| Pooling | MaxPool, AvgPool, GlobalPool, AdaptivePool |
| Recurrent | LSTM, GRU, RNN, Bidirectional |
| Attention | MultiHeadAttention, SelfAttention, CrossAttention |
| Activation | ReLU, GELU, Swish, SiLU, Mish, Tanh, Sigmoid |
| Normalization | BatchNorm, LayerNorm, GroupNorm, InstanceNorm |

## Requirements

- Node.js 18+ (web)
- Python 3.10+ (backend): torch, onnx, fastapi
- Modern browser with WebGPU/WebGL2 support

## License

Apache-2.0

---

## ⚠️ Attribution & Credit Notice

This project is created and maintained by **Mayank Basena** ([@mayank-dev-15](https://github.com/mayank-dev-15)).

If you fork, use, modify, or derive work from this repository, **you must give proper credit** to the original author. This includes:

- Keeping this attribution section intact in any fork or derivative work
- Crediting **Mayank Basena** in your project's README or documentation
- Linking back to the original repository

**Failure to provide proper credit is a violation of the spirit of open source and may result in a DMCA takedown request.**

> *"No AI. No Shortcuts."* — Mayank Basena
