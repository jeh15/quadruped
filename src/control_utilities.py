import functools

import jax
import jax.numpy as jnp

from brax.base import System


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


@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 0, 0), out_axes=1)
def remap_controller(
    value: jax.Array,
    original_range: jax.Array,
    target_range: jax.Array,
) -> jnp.ndarray:
    """
        value: Desired value to remap from original_range to target_range
        original_range: Original range of value
        target_range: Target remap range
    """
    remapped_value = (
        target_range[0]
        + (value - original_range[0])
        * (target_range[1] - target_range[0])
        / (original_range[1] - original_range[0])
    )
    clipped_value = jnp.clip(remapped_value, target_range[0], target_range[1])
    return clipped_value
