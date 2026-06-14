#!/usr/bin/env python3
"""Generate textured GLB models from a directory of images using Hunyuan3D-2.1.

Run from the root of a Hunyuan3D-2.1 clone (where hy3dshape/ and hy3dpaint/ exist).
Copy this script and torchvision_fix.py into that clone before running.

Usage:
    python generate_models.py /path/to/images -o output/
"""

import argparse
import os
import sys

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix not found — run hy3dpaint/utils/torchvision_fix.py if imports fail")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-generate textured GLB models from images")
    parser.add_argument("input_dir", help="Directory of PNG/JPG images (searched recursively)")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory (default: output)")
    return parser.parse_args()


def load_pipelines():
    sys.path.insert(0, "./hy3dshape")
    sys.path.insert(0, "./hy3dpaint")

    from PIL import Image
    from hy3dshape.rembg import BackgroundRemover
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    model_path = "tencent/Hunyuan3D-2.1"
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    rembg = BackgroundRemover()

    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    return shape_pipeline, rembg, paint_pipeline


def process_image(image_path, output_dir, shape_pipeline, rembg, paint_pipeline):
    from PIL import Image

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(output_dir, image_name)
    os.makedirs(out_dir, exist_ok=True)

    textured_glb = os.path.join(out_dir, f"{image_name}_textured.glb")
    if os.path.exists(textured_glb):
        print(f"Skipping {image_path} — already processed")
        return

    print(f"Processing {image_path}")
    image = Image.open(image_path)
    if image.mode != "RGBA":
        image = image.convert("RGB")
        image = rembg(image)
    else:
        image = image.convert("RGBA")

    mesh = shape_pipeline(image=image)[0]
    mesh_glb = os.path.join(out_dir, f"{image_name}.glb")
    mesh.export(mesh_glb)

    paint_pipeline(
        mesh_path=mesh_glb,
        image_path=image_path,
        output_mesh_path=textured_glb,
    )
    print(f"Saved {textured_glb}")


def iter_images(input_dir):
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                yield os.path.join(root, name)


def main():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        sys.exit(f"Input directory not found: {args.input_dir}")

    shape_pipeline, rembg, paint_pipeline = load_pipelines()

    for image_path in iter_images(args.input_dir):
        process_image(image_path, args.output_dir, shape_pipeline, rembg, paint_pipeline)


if __name__ == "__main__":
    main()
