"""
CLIP-Conditioned VAE with FiLM (Feature-wise Linear Modulation)

This implements a conditional VAE where the decoder is conditioned on CLIP embeddings
via FiLM layers. CLIP provides a joint text-image embedding space, allowing generation
from both text prompts and reference images.

Architecture:
- Encoder: Same as unconditional VAE (conv layers -> latent mu/log_var)
- Decoder: Deconv layers with FiLM conditioning at each layer
  - FiLM: x_out = scale(clip) * x + shift(clip)

Training:
- CLIP model is frozen (pre-trained)
- VAE learns to reconstruct images conditioned on their CLIP embeddings
- At inference: can use CLIP text embeddings for text-to-image generation
"""

from dataclasses import dataclass
from typing import Optional, Callable
import urllib.request
import os

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import optax
import torch

# CLIP imports - using Hugging Face transformers with JAX
# Install: pip install transformers jax jaxlib
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Import data loader from existing codebase
from jax_flow.datasets.celeb_a import DataConfig, make_dataloader, visualize_batch


@dataclass
class Config:
    """Configuration for CLIP-Conditioned VAE"""
    hidden_size: int = 16  # Latent dimension
    clip_dim: int = 512    # CLIP projection dimension (both text and image project to 512-dim)
    film_hidden_dim: int = 128  # Hidden dim for FiLM MLPs


class FiLM(nnx.Module):
    """
    Feature-wise Linear Modulation layer.

    Transforms a conditioning vector (CLIP embedding) into scale and shift
    parameters for feature modulation:
        output = scale(cond) * x + shift(cond)

    This allows the conditioning signal to adaptively scale and translate
    feature maps in the decoder.
    """
    def __init__(self, cond_dim: int, num_features: int, rngs: nnx.Rngs):
        """
        Args:
            cond_dim: Dimension of conditioning vector (CLIP embedding)
            num_features: Number of features in the feature map being modulated
            rngs: NNX RNGs for parameter initialization
        """
        # MLP to produce scale parameter
        self.scale_mlp = nnx.Linear(cond_dim, num_features, rngs=rngs)
        # MLP to produce shift parameter
        self.shift_mlp = nnx.Linear(cond_dim, num_features, rngs=rngs)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """
        Apply FiLM modulation.

        Args:
            x: Feature map of shape (B, H, W, C) or (B, C)
            cond: Conditioning vector of shape (B, cond_dim)

        Returns:
            Modulated features with same shape as x
        """
        # Compute scale and shift from conditioning
        scale = self.scale_mlp(cond)  # (B, num_features)
        shift = self.shift_mlp(cond)  # (B, num_features)

        # Reshape for broadcasting based on input dimensions
        if x.ndim == 4:  # (B, H, W, C)
            scale = scale.reshape(x.shape[0], 1, 1, -1)
            shift = shift.reshape(x.shape[0], 1, 1, -1)
        elif x.ndim == 2:  # (B, C)
            pass  # Already (B, C)

        # Apply modulation: scale * x + shift
        return scale * x + shift


class CLIPConditionedVAE(nnx.Module):
    """
    VAE with CLIP conditioning via FiLM layers.

    The encoder maps images to latent distributions. The decoder generates
    images from latents, conditioned on CLIP embeddings via FiLM.
    """

    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config

        # ========== Encoder (same as unconditional VAE) ==========
        # Conv layers progressively reduce spatial dimensions while increasing channels
        self.conv1 = nnx.Conv(in_features=3, out_features=16, kernel_size=(3, 3),
                              strides=(2, 2), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(in_features=16, out_features=32, kernel_size=(3, 3),
                              strides=(2, 2), padding='SAME', rngs=rngs)
        self.conv3 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3),
                              strides=(1, 1), padding='SAME', rngs=rngs)

        # Project flattened features to latent parameters (mu, log_var)
        # Output is 2 * hidden_size because we split into mu and log_var
        self.linear1 = nnx.Linear(14 * 14 * 64, 2 * config.hidden_size, rngs=rngs)

        # ========== Decoder with FiLM conditioning ==========
        # Project latent to spatial features
        self.linear2 = nnx.Linear(config.hidden_size, 14 * 14 * 64, rngs=rngs)

        # Deconv layers with FiLM conditioning at each layer
        # FiLM allows CLIP embeddings to modulate the generation process

        # Layer 1: 64 channels -> FiLM -> deconv -> 32 channels
        self.film1 = FiLM(config.clip_dim, num_features=64, rngs=rngs)
        self.deconv1 = nnx.ConvTranspose(in_features=64, out_features=32,
                                         kernel_size=(3, 3), strides=(2, 2),
                                         padding='SAME', rngs=rngs)

        # Layer 2: 32 channels -> FiLM -> deconv -> 16 channels
        self.film2 = FiLM(config.clip_dim, num_features=32, rngs=rngs)
        self.deconv2 = nnx.ConvTranspose(in_features=32, out_features=16,
                                         kernel_size=(3, 3), strides=(2, 2),
                                         padding='SAME', rngs=rngs)

        # Layer 3: 16 channels -> FiLM -> deconv -> 3 channels (RGB)
        self.film3 = FiLM(config.clip_dim, num_features=16, rngs=rngs)
        self.deconv3 = nnx.ConvTranspose(in_features=16, out_features=3,
                                         kernel_size=(3, 3), strides=(1, 1),
                                         padding='SAME', rngs=rngs)

    def encode(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode images to latent distribution parameters.

        Args:
            x: Input images of shape (B, C, H, W) in CHW format

        Returns:
            mu: Mean of latent distribution (B, hidden_size)
            log_var: Log variance of latent distribution (B, hidden_size)
        """
        # Convert CHW -> HWC for convolution
        x = x.transpose(0, 2, 3, 1)  # (B, H, W, C)

        # Convolutional feature extraction
        x = jax.nn.relu(self.conv1(x))  # (B, H/2, W/2, 16)
        x = jax.nn.relu(self.conv2(x))  # (B, H/4, W/4, 32)
        x = jax.nn.relu(self.conv3(x))  # (B, H/4, W/4, 64)

        # Flatten and project to latent
        B = x.shape[0]
        x = x.reshape(B, -1)  # (B, 14*14*64)
        x = self.linear1(x)   # (B, 2 * hidden_size)

        # Split into mean and log variance
        mu, log_var = jnp.split(x, 2, axis=1)
        return mu, log_var

    def reparameterize(self, mu: jnp.ndarray, log_var: jnp.ndarray,
                       key: jnp.ndarray) -> jnp.ndarray:
        """
        Reparameterization trick: sample from N(mu, sigma^2) via noise.

        z = mu + sigma * epsilon, where epsilon ~ N(0, I)

        This allows backpropagation through the sampling operation.
        """
        epsilon = jax.random.normal(key, log_var.shape)
        return mu + jnp.exp(0.5 * log_var) * epsilon

    def decode(self, z: jnp.ndarray, clip_emb: jnp.ndarray) -> jnp.ndarray:
        """
        Decode latent codes to images, conditioned on CLIP embeddings.

        Args:
            z: Latent codes of shape (B, hidden_size)
            clip_emb: CLIP embeddings of shape (B, clip_dim)

        Returns:
            Reconstructed images of shape (B, C, H, W) in CHW format
        """
        # Project latent to spatial features
        x = self.linear2(z)  # (B, 14*14*64)
        B = x.shape[0]
        x = x.reshape(B, 14, 14, 64)  # (B, 14, 14, 64)

        # Decoder layer 1 with FiLM conditioning
        x = self.film1(x, clip_emb)  # Apply FiLM modulation
        x = jax.nn.relu(self.deconv1(x))  # (B, 28, 28, 32)

        # Decoder layer 2 with FiLM conditioning
        x = self.film2(x, clip_emb)  # Apply FiLM modulation
        x = jax.nn.relu(self.deconv2(x))  # (B, 56, 56, 16)

        # Decoder layer 3 with FiLM conditioning
        x = self.film3(x, clip_emb)  # Apply FiLM modulation
        x = jax.nn.sigmoid(self.deconv3(x))  # (B, 56, 56, 3), sigmoid for [0,1]

        # Convert HWC -> CHW for output
        x = x.transpose(0, 3, 1, 2)  # (B, 3, H, W)
        return x

    def __call__(self, x: jnp.ndarray, clip_emb: jnp.ndarray,
                 key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Full forward pass: encode -> sample -> decode.

        Args:
            x: Input images (B, C, H, W)
            clip_emb: CLIP embeddings (B, clip_dim)
            key: JAX random key for sampling

        Returns:
            y: Reconstructed images (B, C, H, W)
            mu: Latent means (B, hidden_size)
            log_var: Latent log variances (B, hidden_size)
            key: Updated random key
        """
        # Encode to latent distribution
        mu, log_var = self.encode(x)

        # Sample latent code
        key, subkey = jax.random.split(key)
        z = self.reparameterize(mu, log_var, subkey)

        # Decode with conditioning
        y = self.decode(z, clip_emb)

        return y, mu, log_var, key


class CLIPWrapper:
    """
    Wrapper for Hugging Face CLIP model to extract embeddings.

    CLIP provides a joint embedding space for text and images, allowing:
    - Text-to-image: "smiling face" -> embedding -> generate face
    - Image-to-image: reference.jpg -> embedding -> generate variation
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model and processor.

        Args:
            model_name: Hugging Face model identifier for CLIP
        """
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode (frozen)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """
        Encode text prompts to CLIP embeddings.

        Args:
            texts: List of text strings

        Returns:
            Text embeddings as numpy array (len(texts), clip_dim)
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        # Use get_text_features() which applies projection to common space
        text_features = self.model.get_text_features(**inputs)
        # Normalize embeddings (CLIP embeddings are typically normalized)
        text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)
        return text_features.detach().numpy()

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """
        Encode images to CLIP embeddings.

        Args:
            images: List of PIL Images

        Returns:
            Image embeddings as numpy array (len(images), clip_dim)
        """
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        # Use get_image_features() which applies projection to common space
        image_features = self.model.get_image_features(**inputs)
        # Normalize embeddings
        image_features = image_features / torch.norm(image_features, dim=-1, keepdim=True)
        return image_features.detach().numpy()


@nnx.jit
def step_fn(model: CLIPConditionedVAE, x: jnp.ndarray,
            clip_emb: jnp.ndarray, key: jnp.ndarray):
    """
    Single training step with conditional ELBO loss.

    The loss consists of:
    1. Reconstruction loss: ||x - decode(z, clip_emb)||^2
    2. KL divergence: D_KL(q(z|x) || p(z))

    Args:
        model: The VAE model
        x: Batch of images (B, C, H, W)
        clip_emb: Batch of CLIP embeddings (B, clip_dim)
        key: JAX random key

    Returns:
        loss: Scalar loss value
        (y, key): Tuple of reconstructed images and updated key
        grads: Gradients for optimization
    """
    def loss_fn(model: CLIPConditionedVAE):
        y, mu, log_var, key_out = model(x, clip_emb, key)

        # Reconstruction loss (MSE)
        recon_loss = jnp.sum((y - x) ** 2)

        # KL divergence to standard normal prior
        # KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(sigma^2 + mu^2 - log(sigma^2) - 1)
        kl_loss = 0.5 * jnp.sum(jnp.exp(log_var) + mu ** 2 - log_var - 1)

        # Total ELBO loss (negative ELBO, so we minimize)
        loss = (recon_loss + kl_loss) / x.shape[0]

        return loss, (y, key_out)

    return nnx.value_and_grad(loss_fn, has_aux=True)(model)


def generate_from_text(
    model: CLIPConditionedVAE,
    clip_wrapper: CLIPWrapper,
    text_prompts: list[str],
    key: jnp.ndarray,
    config: Config
) -> np.ndarray:
    """
    Generate images from text prompts using CLIP conditioning.

    Args:
        model: Trained conditional VAE
        clip_wrapper: CLIP model wrapper
        text_prompts: List of text descriptions
        key: JAX random key
        config: Model configuration

    Returns:
        Generated images as numpy array (len(prompts), C, H, W)
    """
    # Get CLIP text embeddings
    clip_emb = clip_wrapper.encode_text(text_prompts)
    clip_emb = jnp.array(clip_emb)

    # Sample random latents from prior N(0, I)
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (len(text_prompts), config.hidden_size))

    # Decode with text conditioning
    images = model.decode(z, clip_emb)

    return np.array(images)


def generate_from_image(
    model: CLIPConditionedVAE,
    clip_wrapper: CLIPWrapper,
    reference_images: list[Image.Image],
    key: jnp.ndarray,
    config: Config
) -> np.ndarray:
    """
    Generate variations of reference images using CLIP conditioning.

    Args:
        model: Trained conditional VAE
        clip_wrapper: CLIP model wrapper
        reference_images: List of PIL reference images
        key: JAX random key
        config: Model configuration

    Returns:
        Generated image variations as numpy array
    """
    # Get CLIP image embeddings
    clip_emb = clip_wrapper.encode_images(reference_images)
    clip_emb = jnp.array(clip_emb)

    # Sample random latents
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (len(reference_images), config.hidden_size))

    # Decode with image conditioning
    images = model.decode(z, clip_emb)

    return np.array(images)


def visualize_generation(
    images: np.ndarray,
    titles: list[str] = None,
    save_path: str = None
):
    """
    Visualize generated images in a grid.

    Args:
        images: Array of images (N, C, H, W) in CHW format
        titles: Optional list of titles for each image
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt

    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i, img in enumerate(images):
        row, col = i // cols, i % cols
        ax = axes[row][col]

        # Ensure numpy array (convert from JAX if needed)
        img = np.array(img)

        # Convert CHW to HWC for display
        if img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)

        ax.imshow(np.clip(img, 0, 1))
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)

    # Hide unused subplots
    for i in range(n, rows * cols):
        row, col = i // cols, i % cols
        axes[row][col].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def train_clip_vae(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_dir: str = "checkpoints"
):
    """
    Train the CLIP-conditioned VAE on CelebA dataset.

    Training process:
    1. For each batch, extract CLIP image embeddings (frozen CLIP)
    2. Pass images and CLIP embeddings to VAE
    3. Compute conditional ELBO loss
    4. Update VAE parameters via gradient descent

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        save_dir: Directory to save checkpoints
    """
    # Initialize model, optimizer, and CLIP
    rngs = nnx.Rngs(default=0)
    config = Config()
    model = CLIPConditionedVAE(config, rngs)
    clip_wrapper = CLIPWrapper()

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    # Create data loader
    data_config = DataConfig(batch_size=batch_size, num_epochs=num_epochs)
    train_it = make_dataloader("train", data_config)

    # Training loop
    key = jax.random.PRNGKey(42)
    step = 0

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (x, _) in enumerate(train_it):
            # Convert JAX array to numpy for CLIP processing
            x_np = np.array(x)

            # Convert CHW -> HWC for PIL, then to list of images
            x_hwc = x_np.transpose(0, 2, 3, 1)  # (B, H, W, C)
            pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in x_hwc]

            # Extract CLIP embeddings (frozen, no gradients)
            clip_emb_np = clip_wrapper.encode_images(pil_images)
            clip_emb = jnp.array(clip_emb_np)

            # Training step
            (loss, (_, key)), grads = step_fn(model, x, clip_emb, key)
            optimizer.update(model, grads)

            epoch_loss += float(loss)
            num_batches += 1
            step += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss:.4f}")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

        # Save checkpoint
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"vae_clip_epoch_{epoch+1}.pkl")
        # Note: In practice, use orbax or flax checkpointing
        # nnx.save_checkpoint(checkpoint_path, model, optimizer)

    print("Training complete!")
    return model, clip_wrapper, key


def demo_text_to_image(model: CLIPConditionedVAE, clip_wrapper: CLIPWrapper,
                       key: jnp.ndarray, config: Config):
    """
    Demo: Generate faces from text descriptions.
    """
    prompts = [
        "a smiling person",
        "a person with glasses",
        "a person with curly hair",
        "a serious looking person",
    ]

    print("Generating images from text prompts...")
    images = generate_from_text(model, clip_wrapper, prompts, key, config)

    visualize_generation(images, titles=prompts, save_path="text_to_image_samples.png")


def demo_unconditional_generation(model: CLIPConditionedVAE, key: jnp.ndarray, config: Config):
    """
    Demo: Generate images without conditioning (zero CLIP embedding).
    """
    print("Generating unconditional samples (zero conditioning)...")

    # Zero CLIP embedding for unconditional generation
    batch_size = 8
    clip_emb = jnp.zeros((batch_size, config.clip_dim))

    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (batch_size, config.hidden_size))

    images = model.decode(z, clip_emb)

    titles = [f"Sample {i+1}" for i in range(batch_size)]
    visualize_generation(np.array(images), titles=titles, save_path="unconditional_samples.png")


if __name__ == "__main__":
    # Example usage
    print("CLIP-Conditioned VAE with FiLM")
    print("=" * 50)

    # Option 1: Train from scratch
    # model, clip_wrapper, key = train_clip_vae(num_epochs=5)

    # Option 2: Quick demo with untrained model
    print("\nRunning demo with untrained model (for architecture verification)...")
    rngs = nnx.Rngs(default=0)
    config = Config()
    model = CLIPConditionedVAE(config, rngs)
    clip_wrapper = CLIPWrapper()
    key = jax.random.PRNGKey(42)

    # Test unconditional generation
    demo_unconditional_generation(model, key, config)

    # Test text-to-image (embeddings will be meaningful but decoder untrained)
    demo_text_to_image(model, clip_wrapper, key, config)
