# YouTube Thumbnails Training Pipeline

This repository contains the training pipeline for the YouTube Thumbnails MLOps project.

## Overview
- **Input**: DVC-versioned dataset from `youtube-thumbnails-dataset`.
- **Framework**: PyTorch.
- **Task**: **Classification** (Predicting video category from thumbnail).
- **Compute**: RunPod (GPU).
- **Tracking**: Weights & Biases.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure credentials (env vars):
   - `WANDB_API_KEY`
   - `DVC_*` (if pulling data)

## CI/CD
This repository uses GitHub Actions to:
1. **Build & Push**: Automatically builds the Docker image and pushes to Docker Hub on every commit to `main`.
2. **Monitor & Train**:Periodically checks for new data in `youtube-thumbnails-dataset` and triggers training on RunPod.

## Usage
```bash
python train.py
```
