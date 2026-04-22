#!/bin/bash
set -e

echo "[INFO] Installing dependencies..."
pip install -q transformers accelerate sentencepiece protobuf safetensors einops --upgrade

echo "[INFO] Installing Flash Attention 2..."
pip install -q flash-attn --no-build-isolation 2>&1 | grep -v 'Requirement already satisfied' || true

echo ""
echo "[INFO] Starting dataset generation..."
echo "Timestamp: $(date)"
echo ""

python dataset/generate_sae_dataset_qwen.py

echo ""
echo "[OK] Dataset generation completed"
echo "Timestamp: $(date)"
echo ""

if [ -f dataset_qwen_sae.jsonl ]; then
    echo "[INFO] Generated file statistics:"
    wc -l dataset_qwen_sae.jsonl
    echo ""
    echo "First 3 conversations:"
    head -3 dataset_qwen_sae.jsonl | python -m json.tool --compact 2>/dev/null || head -3 dataset_qwen_sae.jsonl
fi
