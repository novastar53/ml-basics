"""Train CLIP-conditioned VAE using pre-extracted CLIP embeddings."""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
from tqdm import tqdm
import os

from vae_clip import CLIPConditionedVAE, Config, step_fn, generate_from_text, visualize_generation


def train_with_cached_embeddings(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    kl_warmup_epochs: int = 5,
    sample_dir: str = "training_samples"
):
    """Train VAE using pre-extracted CLIP embeddings."""

    # Create sample directory
    os.makedirs(sample_dir, exist_ok=True)

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

    # Handle mismatch by using minimum of both
    num_samples = min(len(all_images), len(all_clip_emb))
    all_images = all_images[:num_samples]
    all_clip_emb = all_clip_emb[:num_samples]
    print(f"Using {num_samples} matched samples")

    # Initialize model
    print("Initializing model...")
    rngs = nnx.Rngs(default=0)
    config = Config()
    model = CLIPConditionedVAE(config, rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    # Initialize CLIP wrapper for text generation
    print("Initializing CLIP wrapper for sample generation...")
    from vae_clip import CLIPWrapper
    clip_wrapper = CLIPWrapper()

    # Test prompts for generation
    test_prompts = [
        'a smiling person',
        'a person with glasses',
        'a person with curly hair',
        'a serious looking person'
    ]

    # Training loop
    key = jax.random.PRNGKey(42)
    step = 0
    steps_per_epoch = (num_samples + batch_size - 1) // batch_size

    def get_kl_weight(step, warmup_epochs, steps_per_epoch):
        """Linear KL annealing: 0 -> 1 over warmup_epochs, then stays at 1."""
        total_warmup_steps = warmup_epochs * steps_per_epoch
        if step < total_warmup_steps:
            return step / total_warmup_steps
        return 1.0

    print(f"KL Annealing: Linear β=0 -> 1 over {kl_warmup_epochs} epochs, then β=1")

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

            # Linear KL annealing
            kl_weight = get_kl_weight(step, kl_warmup_epochs, steps_per_epoch)

            # Training step
            (loss, (_, key, recon_loss, kl_loss)), grads = step_fn(model, batch_images, batch_clip, key, kl_weight)
            optimizer.update(model, grads)
            step += 1

            epoch_loss += float(loss)
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")

        # Generate and save samples after each epoch
        print(f"Generating samples for epoch {epoch + 1}...")

        # Text-conditioned samples
        key, subkey = jax.random.split(key)
        images = generate_from_text(model, clip_wrapper, test_prompts, subkey, config)
        save_path = os.path.join(sample_dir, f'epoch_{epoch + 1:02d}_text_conditioned.png')
        visualize_generation(images, titles=test_prompts, save_path=save_path)
        print(f"  Saved text-conditioned samples to {save_path}")

        # Unconditional samples (zero CLIP embedding)
        key, subkey = jax.random.split(key)
        batch_size_uncond = 4
        clip_emb_zero = jnp.zeros((batch_size_uncond, config.clip_dim))
        z = jax.random.normal(subkey, (batch_size_uncond, config.hidden_size))
        images_uncond = model.decode(z, clip_emb_zero)
        save_path_uncond = os.path.join(sample_dir, f'epoch_{epoch + 1:02d}_unconditional.png')
        visualize_generation(
            np.array(images_uncond),
            titles=[f"Unconditional {i+1}" for i in range(batch_size_uncond)],
            save_path=save_path_uncond
        )
        print(f"  Saved unconditional samples to {save_path_uncond}")

    print("\nTraining complete!")
    return model, config, key


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--kl-warmup-epochs', type=int, default=2, help='Linearly increase KL weight from 0 to 1 over this many epochs')
    parser.add_argument('--sample-dir', type=str, default='training_samples', help='Directory to save generated samples')
    args = parser.parse_args()

    model, config, key = train_with_cached_embeddings(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        kl_warmup_epochs=args.kl_warmup_epochs,
        sample_dir=args.sample_dir
    )

    print(f"\nAll training samples saved to {args.sample_dir}/")
