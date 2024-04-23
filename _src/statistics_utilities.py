import functools
from typing import Optional

import jax
import jax.numpy as jnp
from brax.training.acme import running_statistics as rs
from brax.training.acme.running_statistics import RunningStatisticsState


@jax.jit
def update(
    state: RunningStatisticsState,
    x: jax.Array,
) -> RunningStatisticsState:
    def _loop(carry, xs):
        state = carry
        state = rs.update(state, xs)
        return state, None

    state, _ = jax.lax.scan(
        f=_loop,
        init=state,
        xs=x,
    )

    return state


@functools.partial(jax.jit, static_argnames=['max_abs_value'])
@functools.partial(jax.vmap, in_axes=(None, 0, None), out_axes=0)
def normalize(
    state: RunningStatisticsState,
    x: jax.Array,
    max_abs_value: Optional[float] = None,
) -> jnp.ndarray:
    return rs.normalize(x, state, max_abs_value)
