import jax
import jax.numpy as jnp
import jax.dlpack
import torch.utils.dlpack
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

model_id = "Qwen/Qwen3-Embedding-0.6B"
weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
state_dict = load_file(weights_path)
pt_embeddings = state_dict['embed_tokens.weight']
jax_embeddings = jnp.array(pt_embeddings.float().numpy())
print(f"Manifold Shape: {jax_embeddings.shape}") 

import matplotlib.pyplot as plt

def plot_spectral_decay(jax_embeddings, title="Spectral Decay of Qwen3 Embeddings"):
    """
    Computes SVD and plots the spectral decay on a log-log scale.
    jax_embeddings: jnp.array of shape (Vocab_Size, Embedding_Dim)
    """
    
    # 1. Center the embeddings (Essential for Manifold Analysis)
    # Subtract the mean of each feature dimension
    centered_embeddings = jax_embeddings - jnp.mean(jax_embeddings, axis=0)
    
    # 2. Compute Singular Values (SVD)
    # We use compute_uv=False because we only need the values (s), not the vectors (U, V)
    # This is much faster and memory-efficient.
    s = jnp.linalg.svd(centered_embeddings, compute_uv=False)
    
    # 3. Normalize values by the largest singular value
    # This allows us to see the RELATIVE importance of each dimension
    s_normalized = s / s[0]
    
    # 4. Create the Log-Log Plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Plotting on log-log scale to reveal the Power Law
    ranks = jnp.arange(1, len(s_normalized) + 1)
    plt.loglog(ranks, s_normalized, label='Observed Decay', color='#1f77b4', linewidth=2)
    
    # 5. Add a reference Line (Ideal Power Law alpha=1.0)
    # Most natural language manifolds follow Zipf-like/Power-law scaling
    plt.loglog(ranks, 1/ranks, linestyle='--', color='gray', alpha=0.5, label='Power Law')
    
    # Formatting
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Singular Value Rank (log scale)", fontsize=12)
    plt.ylabel("Relative Magnitude", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    
    # Annotate the "Kink" or Noise Floor
    # Typically where the curve flattens out
    plt.annotate('Semantic Core', xy=(5, 0.5), xytext=(20, 0.8),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    plt.tight_layout()
    plt.show()

plot_spectral_decay(jax_embeddings)
