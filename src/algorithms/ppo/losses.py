from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp

from src.algorithms.ppo import networks as ppo_networks


def calculate_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    truncation_mask: jnp.ndarray,
    termination_mask: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculates the Generalized Advantage Estimation."""
    values_ = jnp.concatenate(
        [values, jnp.expand_dims(bootstrap_value, axis=0)], axis=0,
    )
    deltas = rewards + gamma * termination_mask * values_[1:] - values_[:-1]
    deltas *= truncation_mask

    initial_gae = jnp.zeros_like(bootstrap_value)

    def scan_loop(carry, xs):
        gae = carry
        truncation_mask, termination_mask, delta = xs
        gae = (
            delta
            + gamma * gae_lambda * termination_mask * truncation_mask * gae
        )
        return gae, gae

    _, vs = jax.lax.scan(
        scan_loop,
        initial_gae,
        (truncation_mask, termination_mask, deltas),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )

    vs = jnp.add(vs, values)
    vs_ = jnp.concatenate(
        [vs[1:], jnp.expand_dims(bootstrap_value, axis=0)], axis=0,
    )
    advantages = (
        rewards
        + gamma * termination_mask * vs_ - values
    ) * truncation_mask
    return vs, advantages


def test(argv=None):
    # Test Case Import:
    from brax.training.agents.ppo.losses import compute_gae

    key = jax.random.PRNGKey(42)
    shape = (10, 3)
    bootstrap_shape = (1, 3)
    rewards = jax.random.normal(
        key, shape=shape,
    )
    values = jax.random.normal(
        key, shape=shape,
    )
    bootstrap_value = jax.random.normal(
        key, shape=bootstrap_shape,
    )
    truncation_mask = jax.random.randint(
        key, shape=shape, minval=0, maxval=1,
    )
    termination_mask = jax.random.randint(
        key, shape=shape, minval=0, maxval=1,
    )

    vs, advantages = calculate_gae(
        rewards, values, bootstrap_value, truncation_mask, termination_mask,
    )

    truncation_mask = 1 - truncation_mask
    termination_mask = 1 - termination_mask
    _vs, _advantages = compute_gae(
        rewards, values, bootstrap_value, truncation_mask, termination_mask,
    )

    # Test Values:
    print(f'vs test: {jnp.allclose(vs, _vs)}')
    print(f'advantages test: {jnp.allclose(advantages, _advantages)}')


if __name__ == "__main__":
    test()
