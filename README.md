Sonnet

Sonnet is a deep learning framework being built completely from scratch in pure Python.

The project focuses on understanding how modern AI systems work internally by implementing every major component manually instead of relying on frameworks like PyTorch or TensorFlow.

Sonnet is currently developing:

tensor systems

automatic differentiation

computational graphs

neural network layers

optimizers

training systems


Future goals include:

transformer architectures

text generation

image generation

audio generation

video generation

multimodal AI systems



---

Philosophy

Sonnet is not just a machine learning project.

It is an AI systems engineering project focused on:

first-principles understanding

mathematical clarity

modular architecture

framework engineering

deep learning internals


Every feature includes:

implementation code

detailed explanations

markdown documentation

engineering reasoning


The objective is to understand how real AI frameworks are designed internally.


---

Current Development Stage

Completed

Tensor Engine

[x] Tensor abstraction

[x] Scalar tensor operations

[x] Operator overloading

[x] Computational graph tracking

[x] Gradient storage

[x] Parent tensor tracking

[x] Operation tracking


Automatic Differentiation

[x] Backward propagation system

[x] Chain rule implementation

[x] Gradient propagation


Neural Network Core

[x] Basic neural network layers

[x] Neuron abstraction

[x] XOR training example


Tensor Expansion

[x] ND tensor foundations


Optimization

[x] Optimizer foundations


Data Systems

[x] Initial data pipeline work



---

Project Structure

sonnet/
│
├── main.py
│
├── sonnet_core/
│   ├── __init__.py
│   └── tensor.py
│
├── docs/
│   ├── 01_tensor.md
│   ├── 02_tensor_operations.md
│   ├── 03_autograd_graph.md
│   ├── BACKWARD_PROPAGATION.md
│   ├── NN_LAYER.md
│   ├── XOR_AND_TRAINING.md
│   ├── NDTENSOR.md
│   ├── OPTIMIZERS.md
│   └── DATA_PIPELINE.md
│
├── tests/
│
└── examples/


---

Documentation

Core Tensor System

Document	Description

01 Tensor System	Introduction to Sonnet tensors
02 Tensor Operations	Tensor arithmetic and operator overloading
03 Autograd Graph	Computational graph construction



---

Automatic Differentiation

Document	Description

Backward Propagation	Chain rule and backward gradient propagation



---

Neural Networks

Document	Description

Neural Network Layers	Basic neuron and layer abstractions
XOR And Training	First neural network training workflow



---

Tensor Expansion

Document	Description

ND Tensor	Multi-dimensional tensor foundations



---

Optimization Systems

Document	Description

Optimizers	Optimization and parameter update systems



---

Data Systems

Document	Description

Data Pipeline	Dataset and data loading foundations



---

Installation

Clone Repository

git clone https://github.com/Dion-Fred/sonnet.git


---

Enter Project

cd sonnet


---

Run Sonnet

python main.py


---

Example

from sonnet_core.tensor import Tensor

x = Tensor(2)
y = Tensor(3)

z = x * y

z.backward()

print(x.grad)
print(y.grad)

Expected behavior:

x.grad = 3
y.grad = 2


---

Engineering Goals

Sonnet 1.0

Focus:

understanding deep learning internals

building training systems from scratch

implementing transformers manually

creating a small language model



---

Long-Term Vision

Sonnet aims to eventually support:

Deep Learning Infrastructure

autograd engine

neural network systems

optimizers

GPU acceleration

distributed training


Language Models

tokenization

embeddings

transformers

attention mechanisms

autoregressive generation


Generative AI

text generation

image generation

audio generation

video generation

multimodal architectures



---

Why Build From Scratch?

Building systems manually provides understanding of:

tensor computation

gradient propagation

computational graphs

optimization

neural network internals

framework architecture


This project prioritizes understanding the engineering behind modern AI systems.


---

Roadmap

Phase 1 — Tensor Systems

[x] Tensor abstraction

[x] Tensor arithmetic

[x] Graph tracking

[x] Backpropagation

[x] Gradient propagation


Phase 2 — Neural Networks

[x] Neuron abstraction

[x] Layer systems

[x] XOR training

[ ] Activation functions

[ ] Loss functions


Phase 3 — Training Infrastructure

[ ] Dataset systems

[ ] Data batching

[ ] Model serialization

[ ] Evaluation systems


Phase 4 — Transformers

[ ] Embeddings

[ ] Attention

[ ] Positional encoding

[ ] Transformer blocks


Phase 5 — Generative AI

[ ] Text generation

[ ] Image generation

[ ] Audio generation

[ ] Video generation



---

Repository

GitHub Repository:

https://github.com/Dion-Fred/sonnet


---

License

MIT License (recommended)


---

Author

Built by Dion Fred.

Focused on learning AI systems engineering from first principles.