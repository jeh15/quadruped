from absl.testing import absltest

import jax
import numpy as np
import distrax

import brax.training.distribution
import src.distribution


class DistributionTest(absltest.TestCase):
    def test_distribution_fn(self):
        rng_key = jax.random.key(seed=42)
        shape = (12,)

        # Params:
        test_input = jax.random.normal(rng_key, shape)
        std = 0.001
        var = 1.0

        # Brax Distribution:
        brax_distribution = brax.training.distribution.NormalTanhDistribution(
            event_size=shape,
            min_std=std,
        )

        # Distrax Distribution:
        distrax_distribution = src.distribution.ParametricDistribution(
            distribution=distrax.Normal,
            bijector=distrax.Tanh(),
            min_std=std,
            var_scale=var,
        )

        # Calculate Test Values:
        brax_raw_actions = brax_distribution.sample_no_postprocessing(test_input, rng_key)
        brax_log_prob = brax_distribution.log_prob(test_input, brax_raw_actions)
        brax_postprocessed_actions = brax_distribution.postprocess(brax_raw_actions)

        distrax_distribution = distrax_distribution.create_distribution(test_input)
        distrax_raw_actions = distrax_distribution.distribution.sample(seed=rng_key)
        distrax_postprocessed_actions, distrax_log_probs = distrax_distribution.sample_and_log_prob(seed=rng_key)
        distrax_log_prob = np.sum(distrax_log_probs, axis=-1)

        # Tests:
        np.testing.assert_array_almost_equal(brax_raw_actions, distrax_raw_actions)
        np.testing.assert_array_almost_equal(brax_log_prob, distrax_log_prob)
        np.testing.assert_array_almost_equal(brax_postprocessed_actions, distrax_postprocessed_actions)


if __name__ == '__main__':
    absltest.main()
