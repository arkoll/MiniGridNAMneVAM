"""Свёртки на V100 через cuDNN: смоук на тот самый размер батча, что в PPO."""

import sys

import flax.linen as nn
import jax
import jax.numpy as jnp


class M(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.0
        for c in (16, 32, 64, 64):
            x = nn.relu(nn.Conv(c, (3, 3), strides=(2, 2), padding="SAME")(x))
        return x.reshape(x.shape[0], -1).sum()


batch = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
m = M()
x = jnp.zeros((batch, 128, 128, 3), dtype=jnp.uint8)
p = m.init(jax.random.key(0), x)
g = jax.jit(jax.grad(lambda p, x: m.apply(p, x)))(p, x)
jax.block_until_ready(g)
print(f"CONV_BWD_OK batch={batch}")
