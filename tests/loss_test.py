from absl.testing import absltest

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import distrax

from brax.envs import fast
from brax.envs.wrappers.training import wrap
from src.algorithms.ppo.network_utilities import PPONetworkParams
import src.algorithms.ppo.network_utilities as ppo_networks
from src.module_types import identity_normalization_fn
from src.distribution_utilities import ParametricDistribution
from src.training_utilities import unroll_policy_steps

# Test Case Import:
from src.algorithms.ppo.loss_utilities import calculate_gae, loss_function
from brax.training.agents.ppo.losses import compute_gae, compute_ppo_loss

jax.config.parse_flags_with_absl()


class LossUtilitiesTest(absltest.TestCase):
    def test_gae_fn(self):
        rng_key = jax.random.key(seed=42)
        shape = (10, 3)
        bootstrap_shape = (3,)
        rewards = jax.random.normal(
            rng_key, shape=shape,
        )
        values = jax.random.normal(
            rng_key, shape=shape,
        )
        bootstrap_value = jax.random.normal(
            rng_key, shape=bootstrap_shape,
        )
        truncation_mask = jax.random.randint(
            rng_key, shape=shape, minval=0, maxval=2,
        )
        termination_mask = jax.random.randint(
            rng_key, shape=shape, minval=0, maxval=2,
        )

        # Refactored Function:
        returns, advantages = calculate_gae(
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            truncation_mask=truncation_mask,
            termination_mask=termination_mask,
        )

        # Brax Function:
        brax_returns, brax_advantages = compute_gae(
            truncation=1 - truncation_mask,
            termination=1 - termination_mask,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            lambda_=0.95,
            discount=0.99,
        )

        # Tests:
        np.testing.assert_array_almost_equal(returns, brax_returns)
        np.testing.assert_array_almost_equal(advantages, brax_advantages)

    def test_loss_function(self):
        rng_key = jax.random.key(seed=42)

        # Brax Environment:
        env = fast.Fast()
        env = wrap(
            env,
            episode_length=10,
            action_repeat=1,
        )

        # Network Params:
        layer_sizes = (32, 32)
        input_size = env.observation_size
        output_size = env.action_size
        normalization_params = None
        input_normalization_fn = identity_normalization_fn
        activation = nn.tanh
        kernel_initializer = nn.initializers.lecun_uniform()

        # Network:
        networks = ppo_networks.make_ppo_networks(
            observation_size=input_size,
            action_size=output_size,
            input_normalization_fn=input_normalization_fn,
            policy_layer_sizes=layer_sizes,
            value_layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=kernel_initializer,
            action_distribution=ParametricDistribution(
                distribution=distrax.Normal,
                bijector=distrax.Tanh(),
            ),
        )
        policy_params = networks.policy_network.init(rng_key)
        value_params = networks.value_network.init(rng_key)
        policy_generator = ppo_networks.make_inference_fn(networks)
        policy_fn = policy_generator([normalization_params, policy_params])

        # Unroll Policy Steps:
        num_steps = 10
        batch_size = 32
        num_minibatches = 4
        num_envs = 1

        process_count = jax.process_count()
        local_device_count = jax.local_device_count()
        local_devices_to_use = local_device_count

        key_envs = jax.random.split(rng_key, num_envs // process_count)
        key_envs = jnp.reshape(
            key_envs,
            (local_devices_to_use, -1) + key_envs.shape[1:],
        )
        state = env.reset(key_envs)

        # Needs batch dimension:
        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = unroll_policy_steps(
                env,
                current_state,
                policy_fn,
                current_key,
                num_steps,
                extra_fields=('truncation',)
            )
            return (next_state, next_key), data

        _, transitions = jax.lax.scan(
            f,
            (state, rng_key),
            None,
            length=batch_size * num_minibatches // num_envs,
        )

        params = PPONetworkParams(
            policy_params=policy_params,
            value_params=value_params,
        )

        # Loss Function:
        loss, metrics = loss_function(
            params=params,
            ppo_networks=networks,
            normalization_params=normalization_params,
            data=transitions,
            rng_key=rng_key,
            clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantages=False,
        )


if __name__ == '__main__':
    absltest.main()
