import functools

import jax
import jax.numpy as jnp

# Test Case Import:
from src.algorithms.ppo.losses import calculate_gae
from brax.training.agents.ppo.losses import compute_gae


def test(argv=None):
    key = jax.random.PRNGKey(42)
    shape = (10, 3)
    bootstrap_shape = (3,)
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
        key, shape=shape, minval=0, maxval=0,
    )
    termination_mask = jax.random.randint(
        key, shape=shape, minval=0, maxval=2,
    )

    # Remake:
    vs, advantages = calculate_gae(
        rewards, values, bootstrap_value, truncation_mask, termination_mask,
    )

    # Old Version:
    @functools.partial(jax.jit, static_argnames=["episode_length"])
    @functools.partial(jax.vmap, in_axes=(1, 1, 1, None), out_axes=(1, 1))
    def calculate_advantage(
        rewards: jax.Array,
        values: jax.Array,
        mask: jax.Array,
        episode_length: int,
    ):
        gamma = 0.99
        lam = 0.95
        gae = 0.0
        advantage = []
        mask = jnp.squeeze(mask)
        for i in reversed(range(episode_length)):
            error = rewards[i] + gamma * values[i + 1] * mask[i] - values[i]
            gae = error + gamma * lam * mask[i] * gae
            advantage.append(gae)
        advantage = jnp.array(advantage)[::-1]
        returns = advantage + values[:-1]
        return advantage, returns

    _advantages, _vs = calculate_advantage(
        rewards, jnp.concatenate([values, jnp.expand_dims(bootstrap_value, 0)]), truncation_mask, shape[0],
    )

    # Brax Version:
    truncation_mask = 1 - truncation_mask
    termination_mask = 1 - termination_mask
    __vs, __advantages = compute_gae(
        truncation_mask, termination_mask, rewards, values, bootstrap_value, 0.95, 0.99
    )

    # Test Values:
    print('Refactored vs Old:')
    print(f'vs test: {jnp.allclose(vs, _vs)}')
    print(f'advantages test: {jnp.allclose(advantages, _advantages)}')

    print('Refactored vs Brax:')
    print(f'vs test: {jnp.allclose(vs, __vs)}')
    print(f'advantages test: {jnp.allclose(advantages, __advantages)}')

    print('Old vs Brax:')
    print(f'vs test: {jnp.allclose(_vs, __vs)}')
    print(f'advantages test: {jnp.allclose(_advantages, __advantages)}')


if __name__ == "__main__":
    test()
