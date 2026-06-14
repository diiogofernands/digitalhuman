# Episodic Memory from Compression Boundaries in Latent Representation Space

Official implementation for the ReSuME paper at the ICLR 2026 Workshop on Memory for LLM-Based Agentic Systems.

For more information, see the paper on [OpenReview](https://openreview.net/forum?id=En9aRT4uz8).

Project page: [GitHub Pages](https://diiogofernands.github.io/digitalhuman/episodic-memory/).

ReSuME uses **SAE reconstruction error** as a surprise signal to decide when an LLM interaction should be written to episodic memory. Instead of heuristic write rules, the memory gate is driven by representational deviation from routine activation patterns.

## Overview

ReSuME pipeline:

1. Extract hidden activations from an LLM.
2. Reconstruct activations with a Sparse Autoencoder (SAE).
3. Compute surprise from reconstruction error (optionally z-normalized).
4. Write only surprising states into a fixed-size episodic memory bank.

This repository includes:

- core ReSuME package (`resume/`)
- optional example entry points under `examples/`
- optional dataset-preparation utilities under `dataset/`
- a project page published from the repository root `docs/`

## Environment Setup

```bash
git clone <your-repo-url>
cd ICLR-workshop

python -m venv .venv
. .venv/bin/activate
# on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start (Core API)

```python
import torch
from resume import ReSuMEAgent
from resume.sae_loader import SparseAutoencoder

sae = SparseAutoencoder(input_dim=4096, hidden_dim=16384)
agent = ReSuMEAgent(sae=sae, surprise_threshold=2.0, memory_size=100)

state = torch.randn(4096)
result = agent.process(state, context="user turn")
print(result["surprise_metrics"].surprise_score, result["stored_in_memory"])
```

## Main Entry Points

- `examples/synthetic_pipeline.py`

## Repository Structure

```text
.
├── examples/                    # Optional example entry points
├── resume/                      # Core package (agent, surprise detector, memory, SAE loader)
├── dataset/                     # Dataset generation utilities
├── requirements.txt
└── .gitignore
```

## Acknowledgements

This codebase builds on open-source tooling from PyTorch, Hugging Face Transformers, and the broader mechanistic interpretability / sparse autoencoder ecosystem.
