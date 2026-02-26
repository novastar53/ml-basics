#!/bin/bash
# run_vae_training.sh - Orchestrate CLIP extraction on GPU, training on CPU
#
# Usage: ./run_vae_training.sh <ssh-host>
#
# This script:
# 1. Runs CLIP embedding extraction on remote GPU
# 2. Downloads the embeddings to local machine
# 3. Runs VAE training on local CPU using cached embeddings

set -e

SSH_HOST=${1:-}

if [ -z "$SSH_HOST" ]; then
    echo "Usage: $0 <ssh-host>"
    echo ""
    echo "Example:"
    echo "  $0 gpu-server"
    exit 1
fi

echo "========================================"
echo "CLIP-Conditioned VAE Training Pipeline"
echo "========================================"
echo ""
echo "Step 1: Extract CLIP embeddings on GPU ($SSH_HOST)"
echo "Step 2: Download embeddings to local"
echo "Step 3: Train VAE on local CPU"
echo ""

# Step 1: Run CLIP extraction on GPU
echo "========================================"
echo "Step 1: Extracting CLIP embeddings"
echo "========================================"

./remote_run.sh -d -n clip_extract "$SSH_HOST" diffusion_zero_to_hero/extract_clip_embeddings.py --split train --output data/celeba_clip_train.npz

echo ""
echo "CLIP extraction started in background."
echo "Waiting for completion..."
echo ""

# Poll for completion
while true; do
    sleep 30
    
    # Check status
    if ./remote_run.sh "$SSH_HOST" status 2>/dev/null | grep -q "NOT RUNNING"; then
        echo "Extraction completed!"
        break
    fi
    
    echo "Still extracting... ($(date))"
done

# Step 2: Download embeddings
echo ""
echo "========================================"
echo "Step 2: Downloading embeddings"
echo "========================================"

mkdir -p data
scp "$SSH_HOST:~/ml-basics/data/celeba_clip_train.npz" data/

echo "Embeddings downloaded to data/celeba_clip_train.npz"

# Step 3: Train VAE on CPU
echo ""
echo "========================================"
echo "Step 3: Training VAE on CPU"
echo "========================================"

cd diffusion_zero_to_hero
uv run python train_vae_cached.py --epochs 10 --batch-size 32

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo ""
echo "Results saved to: diffusion_zero_to_hero/trained_vae_samples.png"
