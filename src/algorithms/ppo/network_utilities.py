from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

import flax
import flax.struct
from flax import linen as nn
import distrax

from src import networks
from src import distribution_utilities
from src import module_types as types
from src.module_types import PRNGKey


@flax.struct.dataclass
class PPONetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    action_distribution: distribution_utilities.ParametricDistribution


@flax.struct.dataclass
class PPONetworkParams:
    policy_params: types.Params
    value_params: types.Params


def make_inference_fn(ppo_networks: PPONetworks):
    """Creates the params and inference function for the PPO networks."""

    def make_policy(
        params: types.NetworkParams,
        deterministic: bool = False,
    ) -> types.Policy:
        policy_network = ppo_networks.policy_network
        action_distribution = ppo_networks.action_distribution

        def policy(
            x: jnp.ndarray,
            key: PRNGKey,
        ) -> Tuple[types.Action, types.PolicyData]:
            logits = policy_network.apply(*params, x)
            transformed_distribution = action_distribution.create_distribution(logits)
            base_distribution = transformed_distribution.distribution
            if deterministic:
                actions = transformed_distribution.mode()
                policy_data = {}
            else:
                raw_actions = base_distribution.sample(seed=key)
                actions, log_probs = transformed_distribution.sample_and_log_prob(seed=key)
                log_prob = jnp.sum(log_probs, axis=-1)
                policy_data = {"log_prob": log_prob, "raw_action": raw_actions}

            return actions, policy_data

        return policy

    return make_policy


def make_ppo_networks(
    observation_size: int,
    action_size: int,
    input_normalization_fn: types.InputNormalizationFn = types
    .identity_normalization_fn,
    policy_layer_sizes: Sequence[int] = (256, 256),
    value_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = nn.tanh,
    kernel_init: types.Initializer = jax.nn.initializers.lecun_uniform(),
    action_distribution: distribution_utilities.ParametricDistribution = distribution_utilities
    .ParametricDistribution(distribution=distrax.Normal),
) -> PPONetworks:
    """Creates the Policy and Value Networks for PPO."""
    policy_network = networks.make_policy_network(
        input_size=observation_size,
        output_size=2*action_size,
        input_normalization_fn=input_normalization_fn,
        layer_sizes=policy_layer_sizes,
        activation=activation,
        kernel_init=kernel_init,
    )

    value_network = networks.make_value_network(
        input_size=observation_size,
        input_normalization_fn=input_normalization_fn,
        layer_sizes=value_layer_sizes,
        activation=activation,
        kernel_init=kernel_init,
    )

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=action_distribution,
    )