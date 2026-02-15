"""Basic Diffusion Model (DDPM) implementation in JAX/Flax NNX.

Follows the same patterns as vae.py:
- Uses CelebA dataset (56x56 RGB)
- Simple U-Net denoiser
- Variance-preserving forward process
"""

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import optax
from tqdm import tqdm

from jax_flow.datasets.celeb_a import DataConfig, make_dataloader


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion model."""
    # Model architecture
    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4)  # 64, 128, 256 channels
    num_res_blocks: int = 2
    time_emb_dim: int = 256

    # Diffusion parameters
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Training
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 2e-4


class SinusoidalTimeEmbedding(nnx.Module):
    """Sinusoidal time embeddings for diffusion timesteps."""

    def __init__(self, emb_dim: int, rngs: nnx.Rngs):
        self.emb_dim = emb_dim

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Create sinusoidal embeddings for timesteps.

        Args:
            t: Timesteps, shape (batch_size,)

        Returns:
            Embeddings, shape (batch_size, emb_dim)
        """
        half_dim = self.emb_dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class TimeMLP(nnx.Module):
    """MLP to process time embeddings."""

    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_dim, out_dim, rngs=rngs)
        self.linear2 = nnx.Linear(out_dim, out_dim, rngs=rngs)

    def __call__(self, t_emb: jnp.ndarray) -> jnp.ndarray:
        x = nnx.silu(self.linear1(t_emb))
        x = self.linear2(x)
        return x


class ResBlock(nnx.Module):
    """Residual block with time conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.time_mlp = nnx.Linear(time_emb_dim, out_channels, rngs=rngs)

        if in_channels != out_channels:
            self.shortcut = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), rngs=rngs)
        else:
            self.shortcut = None

    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input features (B, H, W, C)
            t_emb: Time embedding (B, time_emb_dim)

        Returns:
            Output features (B, H, W, out_channels)
        """
        h = nnx.silu(self.conv1(x))

        # Add time embedding via broadcasting
        t = self.time_mlp(nnx.silu(t_emb))
        h = h + t[:, None, None, :]

        h = self.conv2(nnx.silu(h))

        if self.shortcut is not None:
            x = self.shortcut(x)

        return x + h


class UNet(nnx.Module):
    """Simple U-Net denoiser for diffusion.

    Architecture similar to the VAE decoder but with:
    - Time conditioning at each block
    - Skip connections between encoder and decoder
    - No latent bottleneck (operates at full resolution)
    """

    def __init__(self, config: DiffusionConfig, rngs: nnx.Rngs):
        self.config = config

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(config.time_emb_dim, rngs=rngs)
        self.time_mlp = TimeMLP(config.time_emb_dim, config.time_emb_dim, rngs=rngs)

        # Input projection (3 channels RGB -> base_channels)
        self.input_conv = nnx.Conv(3, config.base_channels, kernel_size=(3, 3), padding='SAME', rngs=rngs)

        # Encoder (downsampling path) - store blocks per level for clarity
        encoder_block_lists = []  # List of lists, will convert to nnx.List later
        downsample_convs = []

        channels = config.base_channels
        for i, mult in enumerate(config.channel_mults):
            out_channels = config.base_channels * mult

            # Res blocks at this resolution
            level_blocks = []
            for _ in range(config.num_res_blocks):
                level_blocks.append(ResBlock(channels, out_channels, config.time_emb_dim, rngs=rngs))
                channels = out_channels
            encoder_block_lists.append(nnx.List(level_blocks))

            # Downsample (except at the last level)
            if i < len(config.channel_mults) - 1:
                downsample_convs.append(
                    nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
                )
            else:
                downsample_convs.append(None)

        self.encoder_block_lists = nnx.List(encoder_block_lists)
        self.downsample_convs = nnx.List(downsample_convs)

        # Middle (bottleneck)
        mid_channels = config.base_channels * config.channel_mults[-1]
        self.mid_block1 = ResBlock(mid_channels, mid_channels, config.time_emb_dim, rngs=rngs)
        self.mid_block2 = ResBlock(mid_channels, mid_channels, config.time_emb_dim, rngs=rngs)

        # Decoder (upsampling path) - separate lists for upsamples and blocks
        decoder_upsamples = []
        decoder_block_lists = []

        for i, mult in enumerate(reversed(config.channel_mults)):
            out_channels = config.base_channels * mult

            # Upsample (except at the first level)
            if i > 0:
                up_conv = nnx.ConvTranspose(channels, out_channels, kernel_size=(4, 4), strides=(2, 2), padding='SAME', rngs=rngs)
            else:
                up_conv = None
            decoder_upsamples.append(up_conv)

            # Res blocks at this resolution
            level_blocks = []
            for j in range(config.num_res_blocks):
                if j == 0:
                    # First block: concatenation of skip and current features
                    in_ch = out_channels + out_channels if i > 0 else channels + out_channels
                else:
                    # Subsequent blocks: input is output of previous block
                    in_ch = out_channels
                level_blocks.append(ResBlock(in_ch, out_channels, config.time_emb_dim, rngs=rngs))
            decoder_block_lists.append(nnx.List(level_blocks))
            channels = out_channels

        self.decoder_upsamples = nnx.List(decoder_upsamples)
        self.decoder_block_lists = nnx.List(decoder_block_lists)

        # Output projection
        self.output_conv = nnx.Conv(channels, 3, kernel_size=(3, 3), padding='SAME', rngs=rngs)

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass predicting noise.

        Args:
            x: Noisy images (B, C, H, W) in CHW format
            t: Timesteps (B,)

        Returns:
            Predicted noise (B, C, H, W) in CHW format
        """
        # Convert to HWC for convolutions
        x = x.transpose(0, 2, 3, 1)  # (B, H, W, C)

        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        # Input projection
        h = self.input_conv(x)

        # Encoder with skip connections
        skips_per_level = []  # List of lists, one per level
        for level_idx, (level_blocks, down_conv) in enumerate(zip(self.encoder_block_lists, self.downsample_convs)):
            level_skips = []
            # Res blocks
            for block in level_blocks:
                h = block(h, t_emb)
                level_skips.append(h)
            skips_per_level.append(level_skips)

            # Downsample
            if down_conv is not None:
                h = down_conv(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # Decoder with skip connections
        for level_idx, (up_conv, level_blocks) in enumerate(zip(self.decoder_upsamples, self.decoder_block_lists)):
            # Upsample
            if up_conv is not None:
                h = up_conv(h)

            # Get skips for this level (in reverse order of encoder)
            level_skips = skips_per_level[-(level_idx + 1)]

            # Res blocks with skip connections
            # First block gets concatenated skip + features
            # Subsequent blocks get just the previous block's output
            for block_idx, block in enumerate(level_blocks):
                if block_idx == 0:
                    # First block: concatenate skip with current features
                    skip = level_skips[block_idx]
                    h = jnp.concatenate([h, skip], axis=-1)
                elif block_idx < len(level_skips):
                    # Subsequent blocks that have skips: just add skip? No, standard U-Net uses
                    # concatenation only at the beginning of each level
                    pass
                h = block(h, t_emb)

        # Output
        h = self.output_conv(h)

        # Convert back to CHW
        h = h.transpose(0, 3, 1, 2)
        return h


class DiffusionModel:
    """DDPM Diffusion Model wrapper.

    Handles forward process (noising), training, and sampling.
    """

    def __init__(self, config: DiffusionConfig, model: UNet):
        self.config = config
        self.model = model

        # Pre-compute diffusion schedule
        self.betas = jnp.linspace(config.beta_start, config.beta_end, config.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.alphas_cumprod_prev = jnp.concatenate([jnp.array([1.0]), self.alphas_cumprod[:-1]])

        # Pre-compute values for sampling
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = jnp.sqrt(1.0 / self.alphas)

        # Posterior variance for sampling
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x0: jnp.ndarray, t: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward diffusion process: add noise to images.

        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)

        Args:
            x0: Clean images (B, C, H, W)
            t: Timesteps (B,)
            key: RNG key for noise

        Returns:
            (x_t, noise): Noisy images and the noise that was added
        """
        noise = jax.random.normal(key, x0.shape)

        # Get schedule values for each sample in batch
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        x_t = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

        return x_t, noise

    def p_sample(self, x_t: jnp.ndarray, t: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Single reverse diffusion step.

        Samples x_{t-1} from p(x_{t-1} | x_t) using the model's noise prediction.

        Args:
            x_t: Current noisy images (B, C, H, W)
            t: Current timestep (scalar)
            key: RNG key for sampling

        Returns:
            x_{t-1}: Less noisy images
        """
        batch_size = x_t.shape[0]
        t_batch = jnp.full((batch_size,), t)

        # Predict noise
        noise_pred = self.model(x_t, t_batch)

        # Get schedule values
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod_prev[t]
        beta = self.betas[t]

        # Compute mean of p(x_{t-1} | x_t)
        # x_0 prediction from noise
        x_0_pred = (x_t - jnp.sqrt(1 - alpha_cumprod) * noise_pred) / jnp.sqrt(alpha_cumprod)
        x_0_pred = jnp.clip(x_0_pred, -1.0, 1.0)  # Clip for stability

        # Compute posterior mean
        coef1 = jnp.sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod)
        coef2 = jnp.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
        mean = coef1 * x_0_pred + coef2 * x_t

        if t == 0:
            return mean
        else:
            # Add noise scaled by posterior variance
            variance = self.posterior_variance[t]
            noise = jax.random.normal(key, x_t.shape)
            return mean + jnp.sqrt(variance) * noise

    def sample(self, batch_size: int, key: jax.random.PRNGKey, return_trajectory: bool = False) -> jnp.ndarray:
        """Generate images by running reverse diffusion.

        Args:
            batch_size: Number of images to generate
            key: RNG key
            return_trajectory: If True, return intermediate steps

        Returns:
            Generated images (B, C, H, W), optionally with trajectory
        """
        # Start from pure noise
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (batch_size, 3, 56, 56))

        trajectory = [x] if return_trajectory else None

        # Reverse diffusion loop
        for t in tqdm(reversed(range(self.config.num_timesteps)), desc="Sampling", total=self.config.num_timesteps):
            key, subkey = jax.random.split(key)
            x = self.p_sample(x, t, subkey)

            if return_trajectory and (t % 100 == 0 or t == 0):
                trajectory.append(x)

        if return_trajectory:
            return x, trajectory
        return x


@nnx.jit
def train_step(model: UNet, x0: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray, optimizer: nnx.Optimizer):
    """Single training step.

    Args:
        model: UNet model
        x0: Clean images (B, C, H, W)
        t: Timesteps (B,)
        noise: Target noise (B, C, H, W)
        optimizer: NNX optimizer

    Returns:
        loss: MSE loss
    """
    def loss_fn(model):
        noise_pred = model(x0, t)
        loss = jnp.mean((noise_pred - noise) ** 2)
        return loss, noise_pred

    (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss


def train_diffusion(config: DiffusionConfig = None, data_dir: str = "./data"):
    """Train the diffusion model on CelebA.

    Follows the same training pattern as vae.py.
    """
    if config is None:
        config = DiffusionConfig()

    print("Initializing model...")
    rngs = nnx.Rngs(default=0)
    model = UNet(config, rngs)
    diffusion = DiffusionModel(config, model)

    optimizer = nnx.Optimizer(model, optax.adam(config.learning_rate), wrt=nnx.Param)

    # Create dataloader matching VAE settings
    data_cfg = DataConfig(
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        shuffle=True,
        as_chw=True,
        image_size=(56, 56)
    )
    train_it = make_dataloader("train", data_cfg)

    print(f"Training for {config.num_epochs} epochs...")
    key = jax.random.PRNGKey(42)

    step = 0
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_it, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_idx, (x0, _) in enumerate(pbar):
            batch_size = x0.shape[0]

            # Sample random timesteps
            key, t_key, noise_key = jax.random.split(key, 3)
            t = jax.random.randint(t_key, (batch_size,), 0, config.num_timesteps)

            # Forward diffusion: add noise
            x_t, noise = diffusion.q_sample(x0, t, noise_key)

            # Train step: predict noise
            loss = train_step(model, x_t, t, noise, optimizer)

            epoch_loss += float(loss)
            num_batches += 1
            step += 1

            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

    print("Training complete!")
    return model, diffusion, key


def visualize_samples(images: np.ndarray, save_path: str = None, title: str = "Generated Samples"):
    """Visualize generated images in a grid."""
    import matplotlib.pyplot as plt

    # Convert to HWC if CHW
    if images.ndim == 4 and images.shape[1] in (1, 3):
        images = images.transpose(0, 2, 3, 1)

    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        img = np.clip(images[i], 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def visualize_trajectory(trajectory: list, save_path: str = None):
    """Visualize the denoising trajectory."""
    import matplotlib.pyplot as plt

    # Show first image's trajectory
    n_steps = len(trajectory)
    cols = min(n_steps, 8)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 2, 2))

    indices = np.linspace(0, n_steps - 1, cols, dtype=int)

    for idx, ax in zip(indices, axes):
        img = trajectory[idx][0]  # First image in batch
        if img.shape[0] in (1, 3):  # CHW
            img = img.transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')
        timestep = 1000 - idx * 100 if idx < n_steps - 1 else 0
        ax.set_title(f't={timestep}', fontsize=8)

    plt.suptitle('Denoising Trajectory', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved trajectory to {save_path}")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a basic diffusion model")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--generate', action='store_true', help='Generate samples after training')
    parser.add_argument('--n-samples', type=int, default=16, help='Number of samples to generate')
    args = parser.parse_args()

    config = DiffusionConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_timesteps=args.timesteps
    )

    # Train
    model, diffusion, key = train_diffusion(config)

    # Generate samples
    if args.generate:
        print(f"\nGenerating {args.n_samples} samples...")
        key, subkey = jax.random.split(key)
        samples, trajectory = diffusion.sample(args.n_samples, subkey, return_trajectory=True)

        visualize_samples(np.array(samples), save_path="diffusion_samples.png")
        visualize_trajectory(trajectory, save_path="diffusion_trajectory.png")
