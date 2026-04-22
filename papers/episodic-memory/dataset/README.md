# Dataset Utilities

This directory stores dataset generation utilities and expected evaluation inputs.

These scripts are optional and mainly intended for data preparation. They may require Docker,
GPU access, and external model dependencies.

The main evaluation scripts expect the following files (not distributed in this repository):

- `dataset/routine_chat.json`
- `dataset/critical_chat.json`
- `dataset/ood_chat.json`
- `dataset/more_data.jsonl`

Expected format:

- `*.json`: list of conversations; each conversation is a list of turns with `role` and `content`
- `*.jsonl`: one JSON object per line, with labels and turn content

Generation helpers kept in this folder:

- `generate_sae_dataset_qwen.py`
- `run_generation.sh`
- `docker_entrypoint.sh`
