import jax
import jax.numpy as jnp

B, N = 256, 50304

#preds = jax.random.normal(jax.random.key(0), (B, N))
preds = jax.nn.one_hot(jax.random.randint(jax.random.key(1), (B,), 0, N), N)*0.949 + 1e-6
logprobs = jnp.log(preds)
labels = jax.nn.one_hot(jax.random.randint(jax.random.key(0), (B,), 0, N), N)
ce = -labels * logprobs
print(ce)
loss = jnp.sum(ce) / B
print(loss)
