import argparse
import os
from PIL import Image
from transparent_background import Remover


def parse_args():
    parser = argparse.ArgumentParser(description="Remove backgrounds from cropped card art")
    parser.add_argument("input_dir", help="Directory of PNG/JPG images (searched recursively)")
    parser.add_argument("-o", "--output-dir", help="Output directory (default: same as input)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)
    remover = Remover(device="cuda:0")

    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            if file.endswith("_transparent.png"):
                continue

            path = os.path.join(root, file)
            rel = os.path.relpath(path, args.input_dir)
            out_path = os.path.join(output_dir, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            name, _ = os.path.splitext(out_path)
            out_path = f"{name}_transparent.png"
            if os.path.exists(out_path):
                print(f"Skipping {path}")
                continue

            img = Image.open(path).convert("RGB")
            out = remover.process(img)
            out.save(out_path)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
