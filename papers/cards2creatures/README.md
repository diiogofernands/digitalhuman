# From Cards to Creatures: Interactive 3D Generation using GenAI

Project page for the From Cards to Creatures paper at SVR 2025 (27th Symposium on Virtual and Augmented Reality).

For more information, see the paper on [IEEE Xplore](https://doi.org/10.1109/SVR67689.2025.00052) or the [SBC proceedings](https://sol.sbc.org.br/index.php/svr/article/view/40676).

Project page: [GitHub Pages](https://diiogofernands.github.io/digitalhuman/cards2creatures/).

From Cards to Creatures is a mobile AR system that turns physical trading cards into interactive 3D models. YOLOv11 detects cards in real time; pre-generated Hunyuan3D-2.1 meshes are retrieved from the Poke3D dataset and rendered with spatial alignment.

## Overview

Cards to Creatures pipeline:

1. Capture the physical trading card with the mobile camera (Unity AR).
2. Detect and identify the card with a YOLOv11-based detector.
3. Retrieve the corresponding 3D mesh and texture from the pre-compiled Poke3D dataset.
4. Render and spatially align the 3D model with the detected card in the AR scene.

This directory includes the card-recognition and 3D-generation pipelines. The Unity AR client project from the paper is not included in this repository. Additional components can be published in separate updates.

## Acknowledgements

This work builds on open-source tooling from Unity AR Foundation, Ultralytics YOLOv11, and the Hunyuan3D generative 3D ecosystem.
