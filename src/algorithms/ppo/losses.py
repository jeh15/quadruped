from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp

from src.algorithms.ppo import networks as ppo_networks
from src import network_types as types


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


def loss_function(
    params: PPONetworkParams,
    ppo_networks: ppo_networks.PPONetworks,
    normalization_params: Any,
    data: types.Transition,
    rng: types.PRNGKey,
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, types.Metrics]:
    # Unpack PPO networks:
    action_distribution = ppo_networks.action_distribution
    policy_apply = ppo_networks.policy_network.apply
    value_apply = ppo_networks.value_network.apply

    # Reorder data: (B, T, ...) -> (T, B, ...)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    logits = policy_apply(
        normalization_params, params.policy, data.observation,
    )
    values = value_apply(
        normalization_params, params.value, data.observation,
    )
    bootstrap_values = value_apply(
        normalization_params, params.value, data.next_observation[-1],
    )

    # Be careful with these definitions:
    truncation_mask = 1 - data.extras['state_extras']['truncation']
    termination_mask = data.termination_mask * truncation_mask


