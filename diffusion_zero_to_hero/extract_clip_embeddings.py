"""
Extract CLIP embeddings for CelebA dataset.

This script pre-computes CLIP embeddings for all images in the CelebA dataset
and saves them to disk. This allows the VAE training to run efficiently on CPU
without the CLIP inference bottleneck.

Usage:
    python extract_clip_embeddings.py --split train --output data/celeba_clip_train.npz
    python extract_clip_embeddings.py --split test --output data/celeba_clip_test.npz
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from transformers import CLIPProcessor, CLIPModel
from jax_flow.datasets.celeb_a import DataConfig, make_dataloader


def extract_embeddings(split: str, output_path: str, batch_size: int = 64):
    """
    Extract CLIP embeddings for a dataset split.

    Args:
        split: 'train' or 'test'
        output_path: Path to save the .npz file
        batch_size: Batch size for CLIP inference
    """
    print(f"Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    print(f"Loading {split} dataset...")
    cfg = DataConfig(batch_size=batch_size, num_epochs=1, shuffle=False)
    dataloader = make_dataloader(split, cfg)

    all_embeddings = []
    all_labels = []

    print(f"Extracting embeddings...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            # Convert JAX/NumPy images to PIL
            # images are (B, C, H, W) in CHW format, values in [0, 1]
            images_np = np.array(images)
            images_hwc = images_np.transpose(0, 2, 3, 1)  # CHW -> HWC

            # Convert to PIL images (0-255 range)
            pil_images = [
                Image.fromarray((img * 255).astype(np.uint8))
                for img in images_hwc
            ]

            # Process with CLIP
            inputs = processor(images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get image features (use projection to common space)
            embeddings = model.get_image_features(**inputs)

            # Normalize embeddings
            embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)

            # Store
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(np.array(labels))

    # Concatenate all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"Extracted {len(all_embeddings)} embeddings")
    print(f"Embedding shape: {all_embeddings.shape}")

    # Save to disk
    print(f"Saving to {output_path}...")
    np.savez_compressed(output_path,
                        embeddings=all_embeddings,
                        labels=all_labels)
    print(f"Saved!")

    return all_embeddings, all_labels


def main():
    parser = argparse.ArgumentParser(description='Extract CLIP embeddings for CelebA')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='Dataset split to process')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: data/celeba_clip_{split}.npz)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for CLIP inference')

    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/celeba_clip_{args.split}.npz"

    extract_embeddings(args.split, args.output, args.batch_size)


if __name__ == "__main__":
    main()
