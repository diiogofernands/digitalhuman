from __future__ import annotations

import os
import subprocess
import tempfile
import zipfile
import pickle
import uuid
import threading
from pathlib import Path
from typing import Dict, Any, Tuple

import faiss
import torch
import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from datasets import load_dataset

app = FastAPI(
    title="AIrtist",
    description="AIrtist painting API (threaded jobs: /paint returns job_id+txtInfos immediately; poll /jobs/{id}; download /jobs/{id}/download)",
)

device = "cuda"

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    device=device,
    model_kwargs={
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.float16,
    },
    tokenizer_kwargs={"padding_side": "left"},
)

index = faiss.read_index("wikiart.index")

with open("wikiart_meta.pkl", "rb") as f:
    metadata = pickle.load(f)

ds = load_dataset("asahi417/wikiart-all", split="test", keep_in_memory=False)
id_to_row = {ds[i]["id"]: i for i in range(len(ds))}

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, object]] = {}

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_ERROR = "error"


def _job_set(job_id: str, **fields):
    with _jobs_lock:
        job = _jobs.setdefault(job_id, {})
        job.update(fields)


def _job_get(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def search(query: str, k: int = 1):
    emb = model.encode(
        [query],
        prompt_name="query",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    _, indices = index.search(emb, k)
    return [metadata[idx] for idx in indices[0]]


def get_painting(text: str) -> Tuple[Image.Image, str, Dict[str, Any]]:
    top = search(text, 1)[0]
    rec = ds[id_to_row[top["id"]]]
    return rec["image"], rec["url"], top


def compress_pngs_in_place(painting_dir: Path):
    pngs = list(painting_dir.glob("*.png"))
    if not pngs:
        return

    result = subprocess.run(
        [
            "pngquant",
            "--speed", "10",
            "--strip",
            "--skip-if-larger",
            "--quality", "90-100",
            *map(str, pngs),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode not in (0, 99):
        raise RuntimeError(f"pngquant failed with code {result.returncode}")

    for compressed in painting_dir.glob("*-fs8.png"):
        original = compressed.with_name(
            compressed.name.replace("-fs8.png", ".png")
        )

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
        return img[:, :, 2] > 0  # red channel binary

    def save_mask(binary: np.ndarray, path: Path):
        out = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        out[binary] = (0, 0, 255)
        cv2.imwrite(str(path), out)

    i = start_stroke - 1  # zero based index

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

        # nothing to compound
        if count == 1:
            i += 1
            continue

        # overwrite starting mask
        save_mask(acc, masks[i])

        # delete intermediate frames
        for k in range(i + 1, j):
            masks[k].unlink()
            paintings[k].unlink()

        # renumber everything safely
        masks = sorted(mask_dir.glob("frame*.png"))
        paintings = sorted(painting_dir.glob("frame*.png"))

        for idx, (m, p) in enumerate(zip(masks, paintings)):
            new_name = f"frame{idx+1:03d}.png"

            if m.name != new_name:
                m.rename(mask_dir / new_name)
            if p.name != new_name:
                p.rename(painting_dir / new_name)

        # move to next frame after the compounded one
        i += 1


def remove_small_conglomerate_masks(
    painting_dir: Path,
    mask_dir: Path,
    min_pixels: int = 6,
):
    def load_mask_bool(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read mask {path}")
        return img[:, :, 2] > 0  # red channel

    masks = sorted(mask_dir.glob("frame*.png"))
    paintings = sorted(painting_dir.glob("frame*.png"))

    if len(masks) != len(paintings):
        raise RuntimeError("Mask / painting count mismatch")

    to_delete = []

    for idx, mask_path in enumerate(masks):
        mask_bool = load_mask_bool(mask_path)

        if not mask_bool.any():
            # empty mask, treat as small
            to_delete.append(idx)
            continue

        # connected components, 8-connectivity
        mask_uint8 = mask_bool.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8,
        )

        # skip background label 0
        component_sizes = stats[1:, cv2.CC_STAT_AREA]

        if not any(size >= min_pixels for size in component_sizes):
            to_delete.append(idx)

    # delete marked frames
    for idx in reversed(to_delete):
        masks[idx].unlink()
        paintings[idx].unlink()

    # renumber remaining frames
    remaining_masks = sorted(mask_dir.glob("frame*.png"))
    remaining_paintings = sorted(painting_dir.glob("frame*.png"))

    for i, (m, p) in enumerate(zip(remaining_masks, remaining_paintings)):
        new_name = f"frame{i+1:03d}.png"

        if m.name != new_name:
            m.rename(mask_dir / new_name)
        if p.name != new_name:
            p.rename(painting_dir / new_name)


def remove_small_conglomerates_in_frame(
    painting_dir: Path,
    mask_dir: Path,
    min_pixels: int = 6,
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

        # write cleaned mask back
        out = np.zeros_like(img)
        out[cleaned] = (0, 0, 255)
        cv2.imwrite(str(mask_path), out)

    # delete fully black frames
    for idx in reversed(to_delete):
        masks[idx].unlink()
        paintings[idx].unlink()

    # renumber remaining frames
    remaining_masks = sorted(mask_dir.glob("frame*.png"))
    remaining_paintings = sorted(painting_dir.glob("frame*.png"))

    for i, (m, p) in enumerate(zip(remaining_masks, remaining_paintings)):
        new_name = f"frame{i+1:03d}.png"
        if m.name != new_name:
            m.rename(mask_dir / new_name)
        if p.name != new_name:
            p.rename(painting_dir / new_name)


# API models
class PaintRequest(BaseModel):
    transcription: str


class PaintResponse(BaseModel):
    job_id: str
    txtInfos: str


class JobStatus(BaseModel):
    status: str


def _run_job(job_id: str, image: Image.Image, original_name: str):
    try:
        _job_set(job_id, status=JOB_RUNNING)

        out_dir = get_process(image, original_name)

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

        zip_path = out_dir.parent / f"{job_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in analysis.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(analysis))

        _job_set(job_id, status=JOB_DONE, result_zip=str(zip_path))

    except Exception as e:
        print(f"[JOB {job_id}] ERROR:", e, flush=True)
        _job_set(job_id, status=JOB_ERROR, error=str(e))


@app.post("/paint", response_model=PaintResponse)
def paint(req: PaintRequest):
    job_id = uuid.uuid4().hex

    img, original_name, top = get_painting(req.transcription)

    dim_line = f"{img.width}x{img.height}"
    txt_infos = "\n".join(
        [
            str(top.get("artistName", "")),
            str(top.get("title", "")),
            "0",
            dim_line,
        ]
    )

    _job_set(job_id, status=JOB_QUEUED, txtInfos=txt_infos)

    t = threading.Thread(target=_run_job, args=(job_id, img, original_name), daemon=True)
    t.start()

    return PaintResponse(job_id=job_id, txtInfos=txt_infos)


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str):
    job = _job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(status=str(job.get("status", JOB_ERROR)))


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    job = _job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get("status")
    if status != JOB_DONE:
        raise HTTPException(status_code=409, detail=f"Job not finished (status={status})")

    zip_path = job.get("result_zip")
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=500, detail="Result ZIP missing")

    return StreamingResponse(
        open(zip_path, "rb"),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )


# Run example:
# uvicorn pincel_api:app --host 0.0.0.0 --port 8081 --workers 1 --log-level trace --timeout-keep-alive 180 --access-log


