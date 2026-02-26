import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

# 1. Setup Parameters
key = random.PRNGKey(42)
patch_dim = 16
channels = 3
n_features = patch_dim * patch_dim * channels  # 768
n_patches = 2000  # Number of samples to build the covariance matrix

# 2. Generate a "Structured" Signal (The Manifold)
# Real image patches are low-rank. We simulate this with a 30-dim basis.
rank = 30 
key, k1, k2 = random.split(key, 3)
basis = random.normal(k1, (n_features, rank))
coeffs = random.normal(k2, (rank, n_patches))
signal = jnp.dot(basis, coeffs).T  # (n_patches, 768)

# 3. Add High-Dimensional Gaussian Noise
key, k3 = random.split(key)
noise = random.normal(k3, (n_patches, n_features)) * 2.0
noisy_signal = signal + noise

# 4. Calculate Covariance (768 x 768)
def get_cov(data):
    centered = data - jnp.mean(data, axis=0)
    return jnp.dot(centered.T, centered) / (n_patches - 1)

cov_signal = get_cov(signal)
cov_noisy = get_cov(noisy_signal)

# 5. Spectral Analysis (Eigenvalues)
# eigh is efficient for symmetric matrices like Covariance
eigvals_sig = jnp.linalg.eigvalsh(cov_signal)[::-1]
eigvals_noisy = jnp.linalg.eigvalsh(cov_noisy)[::-1]

# 6. Visualize the "Bottleneck Effect"
plt.figure(figsize=(10, 6))
plt.plot(eigvals_sig[:300], label='Clean Manifold (Signal)', color='blue')
plt.plot(eigvals_noisy[:300], label='Noisy Input (Signal + Noise)', color='orange', alpha=0.6)
plt.axvline(x=128, color='red', linestyle='--', label='JiT Bottleneck Boundary')
plt.yscale('log')
plt.title("The Spectral Gap in 768-D Patch Space")
plt.xlabel("Eigenvalue Rank (Dimension)")
plt.ylabel("Variance (Log Scale)")
plt.legend()
plt.show()
