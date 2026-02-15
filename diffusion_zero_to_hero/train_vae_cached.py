"""Train CLIP-conditioned VAE using pre-extracted CLIP embeddings."""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
from tqdm import tqdm

from vae_clip import CLIPConditionedVAE, Config, step_fn, generate_from_text, visualize_generation


def train_with_cached_embeddings(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3
):
    """Train VAE using pre-extracted CLIP embeddings."""
    
    print("Loading pre-extracted CLIP embeddings...")
    data = np.load('data/celeba_clip_train.npz')
    all_clip_emb = data['embeddings']
    all_labels = data['labels']
    print(f"Loaded {len(all_clip_emb)} training embeddings, shape: {all_clip_emb.shape}")
    
    # Load all images into memory
    print("Loading images into memory...")
    from jax_flow.datasets.celeb_a import DataConfig, make_dataloader
    
    cfg = DataConfig(batch_size=1, num_epochs=1, shuffle=False)
    dataloader = make_dataloader('train', cfg)
    
    all_images = []
    for images, _ in tqdm(dataloader, desc="Loading images"):
        all_images.append(np.array(images[0]))  # Remove batch dim
    
    all_images = np.stack(all_images)
    print(f"Loaded {len(all_images)} images, shape: {all_images.shape}")
    
    assert len(all_images) == len(all_clip_emb), \
        f"Mismatch: {len(all_images)} images vs {len(all_clip_emb)} embeddings"
    
    # Initialize model
    print("Initializing model...")
    rngs = nnx.Rngs(default=0)
    config = Config()
    model = CLIPConditionedVAE(config, rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    
    # Training loop
    key = jax.random.PRNGKey(42)
    num_samples = len(all_images)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Shuffle indices
        indices = np.random.permutation(num_samples)
        epoch_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        pbar = tqdm(range(0, num_samples, batch_size), desc="Training")
        for i in pbar:
            batch_indices = indices[i:min(i + batch_size, num_samples)]
            
            # Get batch data
            batch_images = jnp.array(all_images[batch_indices])
            batch_clip = jnp.array(all_clip_emb[batch_indices])
            
            # Training step
            (loss, (_, key)), grads = step_fn(model, batch_images, batch_clip, key)
            optimizer.update(model, grads)
            
            epoch_loss += float(loss)
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")
    
    print("\nTraining complete!")
    return model, config, key


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    model, config, key = train_with_cached_embeddings(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Generate samples
    print("\nGenerating samples...")
    prompts = ['a smiling person', 'a person with glasses', 
               'a person with curly hair', 'a serious looking person']
    
    # Load CLIP wrapper for text encoding
    from vae_clip import CLIPWrapper
    clip_wrapper = CLIPWrapper()
    
    images = generate_from_text(model, clip_wrapper, prompts, key, config)
    visualize_generation(images, titles=prompts, save_path='trained_vae_samples.png')
    print('Saved samples to trained_vae_samples.png')
