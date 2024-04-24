from absl.testing import absltest

import jax
import numpy as np

# Test Case Import:
from src.algorithms.ppo.loss_utilities import calculate_gae
from brax.training.agents.ppo.losses import compute_gae

jax.config.parse_flags_with_absl()


class GAETest(absltest.TestCase):
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


if __name__ == '__main__':
    absltest.main()
