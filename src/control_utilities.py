import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.vmap, in_axes=(0, 0, None, None), out_axes=0)
def feedforward_controller(
    q_desired: jax.Array,
    qd: jax.Array,
    kd: jax.Array,
    saturation: jax.Array,
) -> jnp.ndarray:
    u = q_desired - kd * qd
    u = jnp.clip(u, saturation[:, 0], saturation[:, -1])
    return u


@functools.partial(jax.vmap, in_axes=(0, None), out_axes=0)
def saturation_controller(
    q_desired: jax.Array,
    saturation: jax.Array,
) -> jnp.ndarray:
    u = jnp.clip(q_desired, saturation[:, 0], saturation[:, -1])
    return u
