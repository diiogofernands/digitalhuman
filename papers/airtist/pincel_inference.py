from __future__ import annotations

import sys
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def compress_pngs_in_place(painting_dir: Path):
    pngs = list(painting_dir.glob("*.png"))
    if not pngs:
        return

    result = subprocess.run(
        [
            "pngquant",
            "--speed",
            "10",
            "--strip",
            "--skip-if-larger",
            "--quality",
            "90-100",
            *map(str, pngs),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode not in (0, 99):
        raise RuntimeError(f"pngquant failed with code {result.returncode}")

    for compressed in painting_dir.glob("*-fs8.png"):
        original = compressed.with_name(compressed.name.replace("-fs8.png", ".png"))
        if original.exists():
            original.unlink()
        compressed.rename(original)


def get_process(image: Image.Image, original_name: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="painting_"))
    stem = Path(original_name).stem

    img_path = temp_dir / f"{stem}.png"
    image.save(img_path)

    out_dir = temp_dir / f"{stem}_full_painting"
    out_dir.mkdir()

    repo = Path("ICCV2019-LearningToPaint").resolve()
    subprocess.run(
        [
            "python3",
            str(repo / "baseline/test.py"),
            "--max_step=80",
            f"--actor={repo / 'actor.pkl'}",
            f"--renderer={repo / 'renderer.pkl'}",
            f"--img={img_path}",
            "--divide=4",
            f"--outdir={out_dir}",
        ],
        check=True,
    )

    image.save(out_dir / f"{stem}.png")
    Image.new("RGB", image.size).save(out_dir / "generated000.png")
    return out_dir


def generate_red_masks_from_model_output(input_dir: Path, out_root: Path):
    mask_dir = out_root / "mask"
    painting_dir = out_root / "painting"
    mask_dir.mkdir(parents=True, exist_ok=True)
    painting_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("generated*.png"))
    if len(paths) < 2:
        raise RuntimeError("Need at least two generated images")

    for i in range(len(paths) - 1):
        a = cv2.imread(str(paths[i]))
        b = cv2.imread(str(paths[i + 1]))
        if a is None or b is None:
            raise RuntimeError("Failed to read generated frames")

        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        mask = diff.max(axis=2) > 10

        out = np.zeros((*mask.shape, 3), dtype=np.uint8)
        out[mask] = (0, 0, 255)

        name = f"frame{i+1:03d}.png"
        cv2.imwrite(str(mask_dir / name), out)
        cv2.imwrite(str(painting_dir / name), a)

    compress_pngs_in_place(painting_dir)


def compound_masks_after_stroke(
    mask_dir: Path,
    painting_dir: Path,
    start_stroke: int = 20,
    max_compound: int = 10,
):
    def load_mask(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read {path}")
        return img[:, :, 2] > 0

    def save_mask(binary: np.ndarray, path: Path):
        out = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        out[binary] = (0, 0, 255)
        cv2.imwrite(str(path), out)

    i = start_stroke - 1

    while True:
        masks = sorted(mask_dir.glob("frame*.png"))
        paintings = sorted(painting_dir.glob("frame*.png"))

        if len(masks) != len(paintings):
            raise RuntimeError("Mask / painting count mismatch")

        if i >= len(masks) - 1:
            break

        acc = load_mask(masks[i])
        count = 1
        j = i + 1

        while j < len(masks) and count < max_compound:
            nxt = load_mask(masks[j])

            if np.any(acc & nxt):
                break

            acc |= nxt
            count += 1
            j += 1

        if count == 1:
            i += 1
            continue

        save_mask(acc, masks[i])

        for k in range(i + 1, j):
            masks[k].unlink()
            paintings[k].unlink()

        masks = sorted(mask_dir.glob("frame*.png"))
        paintings = sorted(painting_dir.glob("frame*.png"))

        for idx, (m, p) in enumerate(zip(masks, paintings)):
            new_name = f"frame{idx+1:03d}.png"
            if m.name != new_name:
                m.rename(mask_dir / new_name)
            if p.name != new_name:
                p.rename(painting_dir / new_name)

        i += 1


def remove_small_conglomerates_in_frame(
    painting_dir: Path,
    mask_dir: Path,
    min_pixels: int = 40,
):
    def load_mask(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read mask {path}")
        return img

    masks = sorted(mask_dir.glob("frame*.png"))
    paintings = sorted(painting_dir.glob("frame*.png"))

    if len(masks) != len(paintings):
        raise RuntimeError("Mask / painting count mismatch")

    to_delete = []

    for idx, mask_path in enumerate(masks):
        img = load_mask(mask_path)
        red = img[:, :, 2] > 0

        if not red.any():
            to_delete.append(idx)
            continue

        mask_uint8 = red.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8,
        )

        cleaned = np.zeros_like(red)

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_pixels:
                cleaned |= (labels == label)

        if not cleaned.any():
            to_delete.append(idx)
            continue

        out = np.zeros_like(img)
        out[cleaned] = (0, 0, 255)
        cv2.imwrite(str(mask_path), out)

    for idx in reversed(to_delete):
        masks[idx].unlink()
        paintings[idx].unlink()

    remaining_masks = sorted(mask_dir.glob("frame*.png"))
    remaining_paintings = sorted(painting_dir.glob("frame*.png"))

    for i, (m, p) in enumerate(zip(remaining_masks, remaining_paintings)):
        new_name = f"frame{i+1:03d}.png"
        if m.name != new_name:
            m.rename(mask_dir / new_name)
        if p.name != new_name:
            p.rename(painting_dir / new_name)


def run(input_path: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Accept PNG/JPG/JPEG (case-insensitive)
    if input_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Input must be a .png, .jpg, or .jpeg file")

    image = Image.open(input_path).convert("RGB")

    out_dir = get_process(image, input_path.name)

    analysis = out_dir.parent / "analysis"
    generate_red_masks_from_model_output(out_dir, analysis)

    remove_small_conglomerates_in_frame(
        painting_dir=analysis / "painting",
        mask_dir=analysis / "mask",
        min_pixels=40,
    )

    compound_masks_after_stroke(
        mask_dir=analysis / "mask",
        painting_dir=analysis / "painting",
        start_stroke=20,
        max_compound=10,
    )

    zip_path = Path.cwd() / f"{input_path.stem}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in analysis.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(analysis))

    return zip_path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 pincel_inference.py /path/to/painting.png")

    input_path = Path(sys.argv[1]).expanduser().resolve()
    zip_path = run(input_path)
    print(zip_path)


if __name__ == "__main__":
    main()