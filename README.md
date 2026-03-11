# Building GPT-2 124M from Scratch

This repository contains a step-by-step implementation of the GPT-2 (124 Million parameter) model built from scratch using PyTorch. The project is broken down into 5 incremental Jupyter Notebooks (`model1.ipynb` to `model5.ipynb`), demonstrating the evolution of the architecture from a simple linear model to a complete Transformer network.

## Repository Walkthrough

The project builds the architecture progressively:

### 1. `model1.ipynb`
This is the foundational model. It implements the most basic components:
- **Token Embedding (`nn.Embedding`)**
- **GELU Activation (`nn.GELU`)**
- **Linear Projection (`nn.Linear`)**
*(See `model1_architecture.jpg` for a visual representation)*

### 2. `model2.ipynb`
Builds upon Model 1 by adding positional awareness and normalization:
- **Token Embedding** + **Positional Embedding** (added together)
- **Layer Normalization (`nn.LayerNorm`)**
- **GELU Activation**
- **Linear Projection with Tied Weights** (Output layer shares weights with the Token Embedding layer for efficiency).
*(See `model2_architecture.jpeg` for a visual representation)*

### 3. `model3.ipynb`
Introduces the core concept of Transformers:
- Implements the foundational **Attention Mechanism**.
- Adds **Layer Normalization** specifically before the attention block.
- Starts to structure the forward pass sequence for contextual token relationships.
*(See `model3_architecture.jpeg` for a visual representation)*

### 4. `model4.ipynb`
Combines components into a single structural unit:
- Completes a full **Transformer Block**.
- A single Transformer block here consists of the **Attention Layer** and an **MLP Sub-layer** (GELU), both enveloped with Layer Normalization and skip/residual connections.
*(See `model4_architecture.jpeg` for a visual representation)*

### 5. `model5.ipynb`
The final, complete architecture for the GPT-2 124M model:
- Stacks **12 complete Transformer Blocks** sequentially.
- Implements full **Multi-Head Attention** across the blocks.
- Combines all components to achieve the exact layout and parameter count (124M) of the original OpenAI GPT-2 small model.
*(See `model5_architecture.png` for a visual representation)*

## Requirements
*   Python 3.x
*   PyTorch
*   Tiktoken (for BPE tokenization)
*   Matplotlib (for visualizations)
