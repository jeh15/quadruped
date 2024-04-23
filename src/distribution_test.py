import jax
import numpy as np

import distrax

from brax.training import distribution as bd
import src.distribution as d


def main(argv=None):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (12, ))

    # Params:
    std = 0.001
    var = 1.0

    # Create Brax Distribution:
    brax_dist = bd.NormalTanhDistribution(
        event_size=x.shape,
        min_std=std,
    )

    # Create Distrax Distribution:
    distrax_dist = d.ParametricDistribution(
        distribution=distrax.Normal,
        bijector=distrax.Tanh(),
        min_std=std,
        var_scale=var,
    )

    # Brax Distribution Values:
    r_b = brax_dist.sample_no_postprocessing(x, key)
    lp_b = brax_dist.log_prob(x, r_b)
    ppa_b = brax_dist.postprocess(r_b)

    # Distrax Distribution Values:
    distrax_distribution = distrax_dist.create_distribution(x)
    r_d = distrax_distribution.distribution.sample(seed=key)
    ppa_d, lp_d = distrax_distribution.sample_and_log_prob(seed=key)
    lp_d = np.sum(lp_d, axis=-1)

    # Test:
    print(f'Raw Action Test: {np.allclose(r_b, r_d)}')
    print(f'Log Prob Test: {np.allclose(lp_b, lp_d)}')
    print(f'Post Processed Action Test: {np.allclose(ppa_b, ppa_d)}')


if __name__ == '__main__':
    main()
