# From Cards to Creatures — Setup

Scripts to reproduce the card-recognition and 3D-model pipelines.

## Pipeline

```
get_cards/     fetch metadata → download images → crop art → create labels → prepare dataset
bg_removal/    remove backgrounds from cropped art
train/         train YOLO detector
inference/     card identification API (YOLO → OCR)
generate_3d/   image directory → textured GLB models (requires Hunyuan3D clone)
```

Install Python deps: `pip install -r requirements.txt` (TCGdex: `pip install tcgdex-sdk`)

---

## 1. Card data (`get_cards/`)

Run from `get_cards/`:

```bash
npm install   # optional; only needed if using the Node TCGdex SDK

# For each set (repeat with --set sv03, sv04, …):
python fetch_metadata.py --set sv10
python download_images.py --sets sv10

python crop_card.py --sets sv10
python create_yolo_labels.py --sets sv10
python prepare_dataset.py --sets sv10
```

`prepare_dataset.py` writes `../train/dataset/train/` and `../train/dataset/valid/`.

Optional evolution-line cleanup:

```bash
python remove_evo_overlay.py sv10/sv10-162_cropped_art.png pixels.json
```

---

## 2. Background removal (`bg_removal/`)

```bash
python remove_background.py ../get_cards/sv10 -o transparent_sv10
# → *_transparent.png
```

---

## 3. Train YOLO (`train/`)

```bash
cd train
python train_yolo.py
cp runs/detect/train/weights/best.pt ../inference/final_models/best.pt
```

Classes: `card_num` (0), `card_set` (1).

---

## 4. Card identification API (`inference/`)

```bash
cd inference
mkdir -p final_models
cp ../train/runs/detect/train/weights/best.pt final_models/
python inference_api.py    # POST /inference with base64 image
```

---

## 5. 3D model generation (`generate_3d/`)

Requires a separate [Hunyuan3D-2.1](https://github.com/Tencent/Hunyuan3D-2.1) clone (~30 GB VRAM).

```bash
cd generate_3d
git clone https://github.com/Tencent/Hunyuan3D-2.1.git && cd Hunyuan3D-2.1

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd hy3dpaint/custom_rasterizer && pip install -e . && cd ../..
cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh && cd ../..
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt

cp ../generate_models.py .
cp ../torchvision_fix.py .
cp ../torchvision_fix.py hy3dpaint/utils/

python hy3dpaint/utils/torchvision_fix.py
python generate_models.py /path/to/transparent/images -o output/
```

Output: `output/{name}/{name}_textured.glb`
