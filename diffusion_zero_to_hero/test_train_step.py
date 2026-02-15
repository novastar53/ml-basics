"""Test single training step for CLIP-conditioned VAE."""
import warnings
warnings.filterwarnings('ignore')

from vae_clip import CLIPConditionedVAE, CLIPWrapper, Config, step_fn
from jax_flow.datasets.celeb_a import DataConfig, make_dataloader
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
import numpy as np
from PIL import Image

print('Creating model...')
rngs = nnx.Rngs(default=0)
config = Config()
model = CLIPConditionedVAE(config, rngs)
clip_wrapper = CLIPWrapper()

print('Creating optimizer...')
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

print('Loading data...')
cfg = DataConfig(batch_size=8, num_epochs=1)
train_it = make_dataloader('train', cfg)

print('Getting first batch...')
x, _ = next(train_it)
print(f'Batch shape: {x.shape}')

print('Extracting CLIP embeddings...')
x_np = np.array(x)
x_hwc = x_np.transpose(0, 2, 3, 1)
pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in x_hwc]
clip_emb_np = clip_wrapper.encode_images(pil_images)
clip_emb = jnp.array(clip_emb_np)
print(f'CLIP embeddings shape: {clip_emb.shape}')

print('Running training step...')
key = jax.random.PRNGKey(42)
(loss, (_, key)), grads = step_fn(model, x, clip_emb, key)
print(f'Loss: {loss:.4f}')

optimizer.update(model, grads)
print('Training step complete!')
