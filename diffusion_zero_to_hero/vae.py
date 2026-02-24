from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

import optax

from typing import Optional
#from jax_flow.datasets.fashion_mnist import make_dataloader, visualize_batch, FASHION_LABELS
from jax_flow.datasets.celeb_a import DataConfig, make_dataloader, visualize_batch
from jaxpt.utils import count_params

# Create dataloader with explicit config
cfg = DataConfig(batch_size=32, num_epochs=4, shuffle=True, as_chw=True)
print("Creating CelebA dataloader...")
train_it = make_dataloader("train", cfg)
print(f"DataLoader created with batch_size={cfg.batch_size}")

@dataclass
class Config:
    hidden_size: int = 64


class VAE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        # Encoder: 56x56 -> 28x28 -> 14x14 -> 7x7
        self.conv1 = nnx.Conv(in_features=3, out_features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME', rngs=rngs)
        self.conv3 = nnx.Conv(in_features=64, out_features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME', rngs=rngs)
        # 7x7x128 = 6272
        self.linear1 = nnx.Linear(7*7*128, 2 * config.hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(config.hidden_size, 7*7*128, rngs=rngs)
        # Decoder: 7x7 -> 14x14 -> 28x28 -> 56x56 using resize+conv (avoids checkerboard artifacts)
        self.deconv1 = nnx.Conv(in_features=128, out_features=64, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.deconv2 = nnx.Conv(in_features=64, out_features=32, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.deconv3 = nnx.Conv(in_features=32, out_features=3, kernel_size=(3, 3), padding='SAME', rngs=rngs)

    def _resize_conv(self, x, target_h, target_w, conv_layer):
        """Resize then conv for better upsampling than ConvTranspose."""
        x = jax.image.resize(x, (x.shape[0], target_h, target_w, x.shape[3]), method='bilinear')
        return conv_layer(x)

    def __call__(self, batch, key: jnp.ndarray = None):
        B, _, _, _ = batch.shape
        batch = batch.transpose(0, 2, 3, 1)
        # Encoder with ReLU activations
        x = jax.nn.relu(self.conv1(batch))  # 28x28
        x = jax.nn.relu(self.conv2(x))       # 14x14
        x = jax.nn.relu(self.conv3(x))       # 7x7
        x_flat = x.reshape(B, -1)
        x = self.linear1(x_flat)
        mu, log_var = jnp.split(x, 2, axis=1)
        key, subkey = jax.random.split(key)
        epsilon = jax.random.normal(subkey, log_var.shape)
        l = mu + jnp.sqrt(jnp.exp(log_var)) * epsilon
        x = self.linear2(l)
        x = x.reshape(B, 7, 7, 128)
        # Decoder with resize-conv and ReLU
        x = jax.nn.relu(self._resize_conv(x, 14, 14, self.deconv1))
        x = jax.nn.relu(self._resize_conv(x, 28, 28, self.deconv2))
        y = self._resize_conv(x, 56, 56, self.deconv3)  # No activation on output
        assert(batch.shape == y.shape), f"Shape mismatch: {batch.shape} vs {y.shape}"
        y = y.transpose(0, 3, 1, 2)
        return y, mu, log_var, key


# ---------------------------
# VAE-specific generation
# ---------------------------

def decoder_wrapper(_params_unused, z, _rng=None):
    # `z` may be a numpy or jax array; decode_z returns CHW images
    imgs = decode_z(m, z)
    # convert to HWC if needed
    arr = np.array(imgs)
    if arr.ndim == 4 and arr.shape[1] in (1, 3):
        arr = arr.transpose(0, 2, 3, 1)
    return arr


def decode_z(vae: VAE, z):
    """Decode latent batch z (numpy/jax array) to images using VAE decoder layers.

    Returns images in CHW format matching the forward output.
    """
    x = vae.linear2(z)
    if hasattr(x, 'reshape'):
        B = x.shape[0]
        x = x.reshape(B, 7, 7, 128)
    x = jax.nn.relu(vae._resize_conv(x, 14, 14, vae.deconv1))
    x = jax.nn.relu(vae._resize_conv(x, 28, 28, vae.deconv2))
    y = vae._resize_conv(x, 56, 56, vae.deconv3)
    y = y.transpose(0, 3, 1, 2)
    return y


def reconstruct_images(model, key, n_images=8, save_path=None):
    """Reconstruct real images from the dataset and display original vs reconstructed."""
    import matplotlib.pyplot as plt

    test_cfg = DataConfig(batch_size=n_images, num_epochs=1, shuffle=True, as_chw=True)
    test_it = make_dataloader("test", test_cfg)
    images, _ = next(test_it)

    reconstructed, mu, log_var, _ = model(images, key)

    # Convert to HWC for plotting
    orig = np.array(images)
    recon = np.array(reconstructed)
    if orig.shape[1] in (1, 3):
        orig = orig.transpose(0, 2, 3, 1)
        recon = recon.transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 4))
    for i in range(n_images):
        axes[0, i].imshow(np.clip(orig[i], 0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        axes[1, i].imshow(np.clip(recon[i], 0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved reconstruction to {save_path}")
    plt.show()



@nnx.jit
def step_fn(m, x, key, kl_weight: float = 0.001):
    def loss_fn(m, x, key):
        y, mu, log_var, key = m(x, key)
        assert(y.shape == x.shape)
        # Reconstruction loss (MSE)
        recon_loss = jnp.sum((y - x) ** 2) / y.shape[0]
        # KL divergence with tunable weight
        kl_loss = 0.5 * jnp.sum(jnp.exp(log_var) + mu ** 2 - 1 - log_var) / y.shape[0]
        loss = recon_loss + kl_weight * kl_loss
        return loss, (y, key, recon_loss, kl_loss)

    return nnx.value_and_grad(loss_fn, has_aux=True)(m, x, key)


rngs = nnx.Rngs(default=0)
config = Config()
m = VAE(config, rngs)
print(f"Number of parameters: {count_params(m):,}")

tx = optax.adam(1e-3)
optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)

#print(f"Total iterations: {len(train_it)}")
print("Starting training")

# KL Annealing configuration
kl_warmup_epochs = 2  # Linearly increase β from 0 to 1 over this many epochs
steps_per_epoch = 5000  # Approximate steps per epoch

def get_kl_weight(step, warmup_epochs, steps_per_epoch):
    """Linear KL annealing: 0 -> 1 over warmup_epochs, then stays at 1."""
    total_warmup_steps = warmup_epochs * steps_per_epoch
    if step < total_warmup_steps:
        return step / total_warmup_steps
    return 1.0

print(f"KL Annealing: Linear β=0 -> 1 over {kl_warmup_epochs} epochs, then β=1")
key = jax.random.PRNGKey(42)
import time
start_time = time.time()

epoch = 0
steps_in_epoch = 0
total_step = 0

for i, (x, labels) in enumerate(train_it):
    iter_start = time.time()

    # Linear KL annealing
    kl_weight = get_kl_weight(total_step, kl_warmup_epochs, steps_per_epoch)

    (loss, (y, key, recon_loss, kl_loss)), grads = step_fn(m, x, key, kl_weight)
    optimizer.update(m, grads)
    iter_time = time.time() - iter_start

    steps_in_epoch += 1
    total_step += 1

    if i == 0:
        print(f"First iteration completed in {iter_time:.2f}s (includes JIT compilation)")
    if i % 100 == 0:
        print(f"Epoch {epoch+1}, Step {total_step}: total={loss:.2f}, recon={recon_loss:.2f}, kl={kl_loss:.4f}, β={kl_weight:.3f}, time={iter_time:.2f}s")

    # Detect epoch boundary
    if steps_in_epoch >= steps_per_epoch:
        epoch += 1
        steps_in_epoch = 0
        print(f"=== Completed epoch {epoch} ===")

total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f}s")

# Reconstruct real images to show VAE reconstruction quality
reconstruct_images(m, key, n_images=8, save_path="vae_reconstruction.png")

# Generate from random latents
from jax_flow.generate import plot_samples as generic_plot
generic_plot(jax.random.PRNGKey(2000), None, decoder_wrapper, n_row=4, latent_dim=config.hidden_size, save_to="vae_generation.png")
