"""Copy card images and YOLO labels into train/dataset/ for YOLO training."""

import argparse
import glob
import os
import random
import re
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare YOLO train/valid split")
    parser.add_argument(
        "--sets",
        nargs="+",
        default=[f"sv{i:02d}" for i in range(3, 11)],
        help="Set directories to include (default: sv03–sv10)",
    )
    parser.add_argument(
        "--output",
        default="../train/dataset",
        help="Output dataset root (default: ../train/dataset)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_pairs(sets):
    pairs = []
    card_pattern = re.compile(r"^sv\d{2}-\d{3}\.png$")

    for set_name in sets:
        for image_path in glob.glob(f"{set_name}/{set_name}-*.png"):
            filename = os.path.basename(image_path)
            if not card_pattern.match(filename):
                continue

            card_id = filename[:-4]
            label_path = f"bbox_{set_name}/{card_id}.txt"
            if not os.path.exists(label_path):
                print(f"Warning: missing label for {card_id}, skipping")
                continue

            pairs.append((image_path, label_path))

    return pairs


def copy_split(pairs, split_name, output_root):
    image_dir = os.path.join(output_root, split_name, "images")
    label_dir = os.path.join(output_root, split_name, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for image_path, label_path in pairs:
        filename = os.path.basename(image_path)
        card_id = filename[:-4]
        shutil.copy2(image_path, os.path.join(image_dir, filename))
        shutil.copy2(label_path, os.path.join(label_dir, f"{card_id}.txt"))


def main():
    args = parse_args()
    pairs = collect_pairs(args.sets)
    if not pairs:
        raise SystemExit(
            "No image/label pairs found. Run create_yolo_labels.py first "
            "and ensure card PNGs exist under svXX/."
        )

    random.seed(args.seed)
    random.shuffle(pairs)

    val_count = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    for split in ("train", "valid"):
        shutil.rmtree(os.path.join(args.output, split), ignore_errors=True)

    copy_split(train_pairs, "train", args.output)
    copy_split(val_pairs, "valid", args.output)

    print(f"Prepared {len(train_pairs)} train and {len(val_pairs)} valid samples in {args.output}")


if __name__ == "__main__":
    main()
