# Fine-Grained Spatial Scene Reconstruction for Adaptive Memory in Virtual Reality

Project page for the spatial memory paper at the IEEE VR 2026 Workshop on Spatial Memory in XR (XRMemory).

For more information, see the paper on [IEEE Xplore](https://doi.org/10.1109/VRW70859.2026.00186).

Project page: [GitHub Pages](https://diiogofernands.github.io/digitalhuman/spatial-memory/).

Workshop: [XRMemory'26](https://sites.google.com/view/xrmemory26-ieee-vr2026/main).

This VR system frames spatial memory recall as interactive scene reconstruction through explicit editing operations. A remembered environment is treated as an editable spatial artifact, combining a 360° panorama scaffold with manipulable 3D objects and localized panorama editing.

## Overview

Spatial memory reconstruction pipeline:

1. Start from a 360° panoramic scaffold as an egocentric contextual anchor (Unity, Meta Quest 3).
2. Insert, move, clone, and delete scene elements as reconstruction operations.
3. Adjust fine-grained spatial layout with manipulable 3D objects.
4. Correct or suppress background elements through localized panorama editing.
5. Generate new objects via voice-driven 3D creation (Hunyuan3D).

## Acknowledgements

This work builds on open-source tooling from Unity, Meta Quest SDK, and generative 3D model ecosystems.
