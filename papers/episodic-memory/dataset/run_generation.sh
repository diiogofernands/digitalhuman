#!/bin/bash
# Runs dataset generation in Docker.

set -e

# Optional environment variables:
#   HF_TOKEN      Hugging Face token, if required by the selected model
#   DOCKER_IMAGE  Docker image to use
#   HF_CACHE_DIR  Host cache directory for Hugging Face artifacts
#   GPU_DEVICE    GPU selection, e.g. "all" or "device=0"

HF_TOKEN="${HF_TOKEN:-}"
DOCKER_IMAGE="${DOCKER_IMAGE:-pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
GPU_DEVICE="${GPU_DEVICE:-all}"

mkdir -p "$HF_CACHE_DIR"

echo "[INFO] Starting Docker container in background"
echo "[INFO] Image: $DOCKER_IMAGE"
echo "[INFO] Hugging Face cache: $HF_CACHE_DIR"

if [ "$GPU_DEVICE" = "all" ]; then
    docker run -d \
      --gpus all \
      -e HF_TOKEN="$HF_TOKEN" \
      -e HF_HOME="/workspace/.cache/huggingface" \
      -v "$PWD":/workspace \
      -v "$HF_CACHE_DIR":/workspace/.cache/huggingface \
      -w /workspace \
      --name qwen_dataset_iclr \
      "$DOCKER_IMAGE" \
      bash /workspace/dataset/docker_entrypoint.sh
else
    docker run -d \
      --gpus "$GPU_DEVICE" \
      -e HF_TOKEN="$HF_TOKEN" \
      -e HF_HOME="/workspace/.cache/huggingface" \
      -v "$PWD":/workspace \
      -v "$HF_CACHE_DIR":/workspace/.cache/huggingface \
      -w /workspace \
      --name qwen_dataset_iclr \
      "$DOCKER_IMAGE" \
      bash /workspace/dataset/docker_entrypoint.sh
fi

if [ $? -eq 0 ]; then
    echo "[OK] Container started"
    echo "[INFO] Useful commands:"
    echo "  Live logs:     docker logs -f qwen_dataset_iclr"
    echo "  Stop:          docker stop qwen_dataset_iclr"
    echo "  Status:        docker ps | grep qwen_dataset_iclr"
    echo "  Output file:   dataset_qwen_sae.jsonl"
    echo "[INFO] Checking initial container state..."
    sleep 5

    if docker ps | grep -q qwen_dataset_iclr; then
        echo "[OK] Container is running"
        echo "[INFO] Recent log lines:"
        docker logs qwen_dataset_iclr 2>&1 | tail -10
    else
        echo "[ERROR] Container stopped unexpectedly. Full logs:"
        docker logs qwen_dataset_iclr 2>&1
    fi
else
    echo "[ERROR] Failed to start container"
    exit 1
fi
