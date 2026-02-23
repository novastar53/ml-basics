"""Compare spectral analysis of 16x16 patches from CelebA and ImageNet datasets.

Plots both eigenvalue distributions on the same axes for direct comparison.
"""

import os
import sys

# Add jax_fusion to path for importing jax_flow.datasets
sys.path.insert(0, '/Users/vikram/dev/jax_fusion/src')

import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

from jax_flow.datasets.celeb_a import DataConfig as CelebAConfig, make_dataloader as make_celeba_loader
from jax_flow.datasets.imagenet import DataConfig as ImageNetConfig, make_dataloader as make_imagenet_loader

def extract_patches(images, patch_size, patches_per_image, key):
    """Extract random patches from images."""
    patches = []
    for img in images:
        img_h, img_w = img.shape[0], img.shape[1]
        key, k1, k2 = random.split(key, 3)
        y_positions = random.randint(k1, (patches_per_image,), 0, img_h - patch_size)
        x_positions = random.randint(k2, (patches_per_image,), 0, img_w - patch_size)

        for y, x in zip(np.array(y_positions), np.array(x_positions)):
            patch = img[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch.flatten())
    return jnp.array(patches)

def get_cov(data):
    """Compute covariance matrix of centered data."""
    centered = data - jnp.mean(data, axis=0)
    return jnp.dot(centered.T, centered) / (data.shape[0] - 1)

# Parameters
key = random.PRNGKey(42)
patch_size = 16
channels = 3
n_features = patch_size * patch_size * channels  # 768
n_images = 1000
patches_per_image = 20

print("=" * 60)
print("Loading CelebA dataset...")
print("=" * 60)

os.environ['CELEBA_BACKEND'] = 'torchvision'
celeba_cfg = CelebAConfig(
    batch_size=n_images,
    num_epochs=1,
    shuffle=True,
    as_chw=False,
    sample_size=n_images,
    image_size=(128, 128),
    seed=42
)

celeba_it = make_celeba_loader('train', celeba_cfg)
celeba_images, _ = next(celeba_it)
celeba_images = np.array(celeba_images)
print(f"CelebA images shape: {celeba_images.shape}")

print("\nExtracting patches from CelebA...")
key, k1 = random.split(key)
celeba_patches = extract_patches(celeba_images, patch_size, patches_per_image, k1)
print(f"CelebA patches shape: {celeba_patches.shape}")

print("\nComputing CelebA covariance...")
celeba_cov = get_cov(celeba_patches)
celeba_eigvals = jnp.linalg.eigvalsh(celeba_cov)[::-1]

print("=" * 60)
print("Loading ImageNet dataset...")
print("=" * 60)

os.environ['IMAGENET_BACKEND'] = 'huggingface'
imagenet_cfg = ImageNetConfig(
    batch_size=n_images,
    num_epochs=1,
    shuffle=True,
    as_chw=False,
    sample_size=n_images,
    image_size=(128, 128),
    data_dir='./data/imagenet',
    seed=42
)

# Load ImageNet - will raise if it fails
imagenet_it = make_imagenet_loader('train', imagenet_cfg)
imagenet_images, _ = next(imagenet_it)
imagenet_images = np.array(imagenet_images)
print(f"ImageNet images shape: {imagenet_images.shape}")

print("\nExtracting patches from ImageNet...")
key, k2 = random.split(key)
imagenet_patches = extract_patches(imagenet_images, patch_size, patches_per_image, k2)
print(f"ImageNet patches shape: {imagenet_patches.shape}")

print("\nComputing ImageNet covariance...")
imagenet_cov = get_cov(imagenet_patches)
imagenet_eigvals = jnp.linalg.eigvalsh(imagenet_cov)[::-1]

print("=" * 60)
print("Generating white noise patches...")
print("=" * 60)

key, k3 = random.split(key)
# White noise: each pixel is independent Gaussian
noise_patches = random.normal(k3, (n_images * patches_per_image, n_features)) * 0.3
print(f"Noise patches shape: {noise_patches.shape}")

print("\nComputing white noise covariance...")
noise_cov = get_cov(noise_patches)
noise_eigvals = jnp.linalg.eigvalsh(noise_cov)[::-1]

# Plot comparison
print("\n" + "=" * 60)
print("Generating comparison plot...")
print("=" * 60)

plt.figure(figsize=(10, 6))
plt.plot(celeba_eigvals, color='blue', linewidth=2, label='CelebA', alpha=0.8)
plt.plot(imagenet_eigvals, color='green', linewidth=2, label='ImageNet', alpha=0.8)
plt.plot(noise_eigvals, color='red', linewidth=2, label='White Noise', alpha=0.8, linestyle='--')

plt.yscale('log')
plt.title("Spectral Decay Comparison: 16x16 Patches")
plt.xlabel("Eigenvalue Rank (Dimension)")
plt.ylabel("Variance (Log Scale)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_file = 'patch_spectral_comparison.png'
plt.savefig(output_file, dpi=150)
print(f"\nComparison plot saved to: {output_file}")

# Print summary statistics
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

print(f"\nCelebA:")
print(f"  Top eigenvalue: {celeba_eigvals[0]:.4f}")
print(f"  90% variance: top {jnp.searchsorted(jnp.cumsum(celeba_eigvals) / jnp.sum(celeba_eigvals), 0.9)} dimensions")
print(f"  95% variance: top {jnp.searchsorted(jnp.cumsum(celeba_eigvals) / jnp.sum(celeba_eigvals), 0.95)} dimensions")
print(f"  99% variance: top {jnp.searchsorted(jnp.cumsum(celeba_eigvals) / jnp.sum(celeba_eigvals), 0.99)} dimensions")
print(f"  99.9% variance: top {jnp.searchsorted(jnp.cumsum(celeba_eigvals) / jnp.sum(celeba_eigvals), 0.999)} dimensions")

print(f"\nImageNet:")
print(f"  Top eigenvalue: {imagenet_eigvals[0]:.4f}")
print(f"  90% variance: top {jnp.searchsorted(jnp.cumsum(imagenet_eigvals) / jnp.sum(imagenet_eigvals), 0.9)} dimensions")
print(f"  95% variance: top {jnp.searchsorted(jnp.cumsum(imagenet_eigvals) / jnp.sum(imagenet_eigvals), 0.95)} dimensions")
print(f"  99% variance: top {jnp.searchsorted(jnp.cumsum(imagenet_eigvals) / jnp.sum(imagenet_eigvals), 0.99)} dimensions")
print(f"  99.9% variance: top {jnp.searchsorted(jnp.cumsum(imagenet_eigvals) / jnp.sum(imagenet_eigvals), 0.999)} dimensions")

print(f"\nWhite Noise:")
print(f"  Top eigenvalue: {noise_eigvals[0]:.4f}")
print(f"  90% variance: top {jnp.searchsorted(jnp.cumsum(noise_eigvals) / jnp.sum(noise_eigvals), 0.9)} dimensions")
print(f"  95% variance: top {jnp.searchsorted(jnp.cumsum(noise_eigvals) / jnp.sum(noise_eigvals), 0.95)} dimensions")
print(f"  99% variance: top {jnp.searchsorted(jnp.cumsum(noise_eigvals) / jnp.sum(noise_eigvals), 0.99)} dimensions")
print(f"  99.9% variance: top {jnp.searchsorted(jnp.cumsum(noise_eigvals) / jnp.sum(noise_eigvals), 0.999)} dimensions")
print(f"  (Flat spectrum - all dimensions carry equal variance)")

plt.show()
