# MathVR: Personality-Based Agents for Mathematics Education in Virtual Reality

Project page for the MathVR paper at IEEE AIxVR 2026.

For more information, see the paper on [IEEE Xplore](https://doi.org/10.1109/AIxVR67263.2026.00036).

Project page: [GitHub Pages](https://diiogofernands.github.io/digitalhuman/mathvr/).

MathVR is an extended reality tutoring system with an LLM-driven tutor on Meta Quest 3. Learners interact through speech and handwritten sketches within a shared virtual workspace. The tutor projects synchronized visual explanations onto a virtual whiteboard and uses guided discovery rather than direct solutions.

## Overview

MathVR pipeline:

1. Enter a shared VR workspace with a personality-based LLM tutor (Unreal Engine, Meta Quest 3).
2. Ask questions and submit handwritten sketches via multimodal input.
3. Receive conversational guidance with scaffolding and incremental hints.
4. View synchronized visual explanations projected onto a virtual whiteboard via a tagging mechanism.

The Unreal Engine client project from the paper is not included in this repository. Additional components can be published in separate updates.

## Acknowledgements

This work builds on open-source tooling from Unreal Engine, Meta Quest SDK, OpenAI, and ElevenLabs.
