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


# Position Based Controller:
@jax.jit
@functools.partial(
    jax.vmap, in_axes=(0, 0, 0, None, None, None), out_axes=0,
)
def pd_controller(
    q_desired: jax.Array,
    q: jax.Array,
    qd: jax.Array,
    kp: jax.typing.ArrayLike,
    kd: jax.typing.ArrayLike,
    control_limit: jax.Array,
) -> jnp.ndarray:
    # Error Calculation:
    position_error = q_desired - q
    velocity_error = -qd
    # PD Controller:
    u = kp * position_error + kd * velocity_error
    # Saturate Action:
    u = jnp.clip(u, -control_limit, control_limit)
    u = q + u
    return u


@jax.jit
def pd_torque_controller(
    action: jax.Array,
    q: jax.Array,
    qd: jax.Array,
    q_default: jax.Array,
    kp: jax.Array,
    kd: jax.Array,
    action_scale: jax.Array,
) -> jnp.ndarray:
    # Calculate Scaled Action and Error:
    scaled_action = action_scale * (action - q_default)
    position_error = q_default - q
    velocity_error = -qd
    # PD Controller:
    u = kp * (scaled_action + position_error) + kd * velocity_error
    u = jnp.clip(u, -33.5, 33.5)
    return u


@jax.jit
def calculate_control_saturation(
    control_range: jnp.ndarray,
    scale: float,
) -> jnp.ndarray:
    max_range = jnp.abs(control_range[:, -1] - control_range[:, 0])
    saturation_limit = max_range / scale
    return saturation_limit


# Relative Position Based Controller:
@jax.jit
@functools.partial(
    jax.vmap, in_axes=(0, None, None), out_axes=0,
)
def relative_controller(
    q_desired: jax.Array,
    q_default: jax.Array,
    control_limit: jax.Array,
) -> jnp.ndarray:
    """
        Remaps action from Desired position to 
        Desired Position Relative to Default Position
    """
    clipped_action = jnp.clip(q_desired, control_limit[:, 0], control_limit[:, -1])
    action = clipped_action - q_default
    return action