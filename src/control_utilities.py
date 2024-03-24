import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.vmap, in_axes=(0, 0, 0, None, None, None), out_axes=0)
def pd_controller(
    q_desired: jax.Array,
    q: jax.Array,
    qd: jax.Array,
    kp: jax.Array,
    kd: jax.Array,
    saturation: float,
) -> jnp.ndarray:
    u = kp * (q_desired - q) - kd * qd
    u = jnp.clip(u, -saturation, saturation)
    return kp * (q_desired - q) - kd * qd
