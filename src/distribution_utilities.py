from typing import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import distrax

import src.module_types as types


@dataclass
class ParametricDistribution():
    distribution: Callable[..., distrax.Distribution]
    bijector: distrax.Bijector = distrax.Lambda(lambda x: x)
    min_std: float = 1e-3
    var_scale: float = 1.0

    def create_distribution(self, params: jnp.ndarray) -> distrax.Distribution:
        loc, scale = jnp.split(params, 2, axis=-1)
        scale = (jax.nn.softplus(scale) + self.min_std) * self.var_scale
        return distrax.Transformed(
            distribution=self.distribution(loc=loc, scale=scale),
            bijector=self.bijector,
        )

    def entropy(
        self,
        params: jnp.ndarray,
        rng_key: types.PRNGKey,
    ) -> jnp.ndarray:
        transformed_distribution = self.create_distribution(params=params)
        sample = transformed_distribution.sample(seed=rng_key)
        entropy = transformed_distribution.distribution.entropy()
        forward_log_det_jacobian = self.bijector.forward_log_det_jacobian(
            sample,
        )
        return entropy + forward_log_det_jacobian
