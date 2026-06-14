# AIrtist: VR Framework for Interactive Engagement via Stroke-Based Neural Rendering

Minimal implementation utilities for the AIrtist paper at IEEE AIxVR 2026.

For more information, see the paper on [IEEE Xplore](https://doi.org/10.1109/AIxVR67263.2026.00073).

Project page: [GitHub Pages](https://diiogofernands.github.io/digitalhuman/airtist/).

AIrtist is an immersive VR framework on Meta Quest 3 that bridges art appreciation and interactive engagement. The system combines LLM-driven narration (ElevenLabs TTS), a gamified painting phase grounded in Deep Reinforcement Learning ("Learning to Paint"), and a stroke analysis engine (PCA + connected component labeling) to guide users through reconstructing Van Gogh masterpieces.

## Overview

AIrtist pipeline:

1. Immerse the user in a virtual Van Gogh studio (Unity, Meta Quest 3).
2. Deliver contextual art history and technique explanations via LLM narration and TTS.
3. Decompose target paintings into stroke sequences with a DRL painting model.
4. Translate algorithmic strokes into human-interpretable guides via PCA and differential frame masks.
5. Let users reconstruct the painting interactively in VR.

This directory includes Python utilities for stroke decomposition and the painting API used by the VR client.

## Environment Setup

Clone [Learning to Paint](https://github.com/hzwer/ICCV2019-LearningToPaint) next to the scripts you run (the code expects a sibling folder named `ICCV2019-LearningToPaint`):

```bash
git clone https://github.com/hzwer/ICCV2019-LearningToPaint
```

Download the pretrained `renderer.pkl` and `actor.pkl` into that repository as described in the upstream README.

Install `pngquant` (used to compress generated stroke frames):

```bash
# Ubuntu / Debian
sudo apt install pngquant

# macOS
brew install pngquant
```

Install Python dependencies (PyTorch, OpenCV, Pillow, FastAPI, etc.) for the scripts you plan to run.

## Main Entry Points

- `pincel_inference.py` — run stroke decomposition on a single painting image; writes a `.zip` with masks and intermediate frames
- `pincel_api.py` — FastAPI service for painting lookup and stroke generation
- `create_dataset_artist.py` — build chunked WikiArt artist metadata for retrieval
- `create_embeddings.py` — build FAISS index and metadata for painting search

Example:

```bash
cd papers/airtist
python3 pincel_inference.py /path/to/painting.png
```

## Repository Structure

```text
.
├── pincel_inference.py
├── pincel_api.py
├── create_dataset_artist.py
├── create_embeddings.py
└── README.md
```

## Acknowledgements

This codebase builds on [Learning to Paint](https://github.com/hzwer/ICCV2019-LearningToPaint) (Huang et al., ICCV 2019), Unity, Meta Quest SDK, and the broader neural rendering / reinforcement learning painting ecosystem.
