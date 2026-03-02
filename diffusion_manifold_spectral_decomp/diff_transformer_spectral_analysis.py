import numpy as np
import matplotlib.pyplot as plt

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def get_spectral_decay(matrix):
    s = np.linalg.svd(matrix, compute_uv=False)
    return s / s[0] # Normalize to top singular value

# Parameters
seq_len = 128
dim = 64
head_dim = dim // 2
lam = 0.8 # Differential coefficient

# Generate dummy Q, K (simulating structural bias)
np.random.seed(42)
Q = np.random.randn(seq_len, dim) * 0.5
K = np.random.randn(seq_len, dim) * 0.5

# 1. Standard Softmax Attention
scores_std = (Q @ K.T) / np.sqrt(dim)
A_std = softmax(scores_std)

# 2. Differential Attention (V2 logic)
Q1, Q2 = Q[:, :head_dim], Q[:, head_dim:]
K1, K2 = K[:, :head_dim], K[:, head_dim:]
scores1 = (Q1 @ K1.T) / np.sqrt(head_dim)
scores2 = (Q2 @ K2.T) / np.sqrt(head_dim)
A_diff = softmax(scores1) - lam * softmax(scores2)

# Spectral Decay Data
decay_std = get_spectral_decay(A_std)
decay_diff = get_spectral_decay(A_diff)

# Plotting the Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Spectral Decay Plot
ax1.plot(decay_std, label='Standard Softmax', color='#1f77b4', linewidth=2)
ax1.plot(decay_diff, label=f'Differential (λ={lam})', color='#d62728', linestyle='--', linewidth=2)
ax1.set_yscale('log')
ax1.set_title('Attention Spectral Decay', fontsize=14)
ax1.set_xlabel('Singular Value Index')
ax1.set_ylabel('Normalized Magnitude (log)')
ax1.legend()

# Visual Heatmap (Diff Attention)
im = ax2.imshow(A_diff[:48, :48], cmap='RdBu_r', vmin=-0.02, vmax=0.02)
ax2.set_title('Differential Attention Map (Detail)', fontsize=14)
fig.colorbar(im, ax=ax2)

plt.tight_layout()
plt.show()
