import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def svd_compress(channel, k):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    compressed = U_k @ np.diag(S_k) @ Vt_k
    return compressed, U_k, S_k, Vt_k


def compress_image_rgb(image, k):
    compressed_channels = []
    storage = 0
    for c in range(3):
        comp, U_k, S_k, Vt_k = svd_compress(image[:, :, c], k)
        compressed_channels.append(comp)
        storage += U_k.size + S_k.size + Vt_k.size
    compressed = np.stack(compressed_channels, axis=2)
    compressed = np.clip(compressed, 0, 1)
    return compressed, storage


def main():
    img_path = "data/celeba/img_align_celeba/000001.jpg"
    img = Image.open(img_path)
    img_np = np.array(img) / 255.0
    h, w, c = img_np.shape

    k_values = [5, 10, 20, 50, 100, 200, 500]
    original_storage = h * w * c

    compressed_images = []
    mse_values = []
    storage_values = []

    for k in k_values:
        comp, storage = compress_image_rgb(img_np, k)
        compressed_images.append(comp)
        mse = np.mean((img_np - comp) ** 2)
        mse_values.append(mse)
        storage_values.append(storage)

    # Singular value decay for the red channel
    U_r, S_r, Vt_r = np.linalg.svd(img_np[:, :, 0], full_matrices=False)
    S_r_normalized = S_r / S_r[0]

    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.2)

    # Original image
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img_np)
    ax_orig.set_title(f"Original\nStorage: {original_storage:,}", fontsize=10)
    ax_orig.axis("off")

    # Compressed images
    positions = [
        (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1),
    ]
    for idx, ((row, col), (k, comp, storage)) in enumerate(zip(positions, zip(k_values, compressed_images, storage_values))):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(comp)
        ratio = original_storage / storage
        ax.set_title(f"K = {k}\nMSE: {mse_values[idx]:.6f}\nRatio: {ratio:.1f}x", fontsize=9)
        ax.axis("off")

    # Singular value decay
    ax_svd = fig.add_subplot(gs[2, 2])
    ax_svd.semilogy(S_r_normalized, linewidth=1.5)
    ax_svd.set_xlabel("Singular Value Index", fontsize=9)
    ax_svd.set_ylabel("Normalized Singular Value", fontsize=9)
    ax_svd.set_title("Singular Value Decay (Red Channel)", fontsize=10)
    ax_svd.grid(True, alpha=0.3)
    for k in k_values:
        if k < len(S_r):
            ax_svd.axvline(x=k, color='r', linestyle='--', alpha=0.3, linewidth=0.8)

    plt.suptitle("SVD Image Compression on CelebA", fontsize=14, y=0.98)
    plt.savefig("svd_compression_grid.png", dpi=150, bbox_inches="tight")
    plt.show()

    # MSE vs K plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(k_values, mse_values, marker='o', linewidth=2, markersize=6)
    ax2.set_xlabel("K (Rank)", fontsize=11)
    ax2.set_ylabel("MSE", fontsize=11)
    ax2.set_title("Reconstruction Error vs Rank", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig("svd_compression_mse.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Image shape: {h} x {w} x {c}")
    print(f"Original storage: {original_storage:,} values")
    for k, mse, storage in zip(k_values, mse_values, storage_values):
        ratio = original_storage / storage
        print(f"K={k:3d}: MSE={mse:.6f}, Storage={storage:,}, Compression={ratio:.1f}x")


if __name__ == "__main__":
    main()
