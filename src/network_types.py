from typing import Any, Callable, Sequence, Tuple, NamedTuple, Protocol, Mapping

import jax.numpy as jnp

Params = Any
PRNGKey = jnp.ndarray
NomralizationParams = Any
NetworkParams = Tuple[NomralizationParams, Params]

Action = jnp.ndarray
PolicyData = Mapping[str, Any]
Metrics = Mapping[str, Any]


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    termination: jnp.ndarray
    next_observation: jnp.ndarray
    extras: Mapping[str, Any]


class Policy(Protocol):
    def __call__(
        self,
        x: jnp.ndarray,
        key: PRNGKey,
    ) -> Tuple[Action, PolicyData]:
        pass


class InputNormalizationFn(Protocol):
    def __call__(
        self,
        x: jnp.ndarray,
        normalization_params: NomralizationParams
    ) -> jnp.ndarray:
        pass


def identity_normalization_fn(
    x: jnp.ndarray,
    normalization_params: NomralizationParams
) -> jnp.ndarray:
    del normalization_params
    return x
