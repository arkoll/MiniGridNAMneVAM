"""Блокер номер один: работает ли jax на Volta (sm_70) и видит ли он ровно одну карту."""

import os
import time

import jax
import jax.numpy as jnp

print("jax", jax.__version__)
print("devices:", jax.devices())
assert len(jax.devices()) == 1, "ожидалась ровно одна видимая карта, проверь CUDA_VISIBLE_DEVICES"
d = jax.devices()[0]
print("платформа:", d.platform, "| устройство:", d.device_kind)
assert d.platform == "gpu", "jax не видит GPU — дальше нет смысла"

# настоящий matmul, а не просто создание массива: на неподдерживаемой архитектуре падает именно тут
key = jax.random.key(0)
a = jax.random.normal(key, (4096, 4096), dtype=jnp.float32)
f = jax.jit(lambda x: x @ x.T)
r = jax.block_until_ready(f(a))
print("matmul ok, сумма:", float(r.sum()))

t = time.time()
for _ in range(10):
    r = f(a)
jax.block_until_ready(r)
dt = time.time() - t
flops = 10 * 2 * 4096**3
print(f"скорость: {flops / dt / 1e12:.2f} TFLOPS fp32")

print("JAX_V100_STATUS=OK")
