from typing import Callable, Tuple, Any
import functools

import jax
import jax.numpy as jnp
import distrax
from brax.training.acme import running_statistics as rs

import statistics_utilities

# Types:
from flax.core import FrozenDict
PRNGKey = jax.Array


@functools.partial(jax.jit, static_argnames=["apply_fn"])
def forward_pass(
    model_params: FrozenDict,
    apply_fn: Callable[..., Any],
    statistics_state: rs.RunningStatisticsState,
    x: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Normalize Observations:
    x = statistics_utilities.normalize(statistics_state, x, None)
    # Pass Normalized Observations:
    mean, std, values = apply_fn({"params": model_params}, x)
    return mean, std, values


@functools.partial(jax.jit, static_argnames=["multivariate"])
def select_action(
    mean: jax.Array,
    std: jax.Array,
    key: PRNGKey,
    multivariate: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if multivariate:
        probability_distribution = distrax.MultivariateNormalDiag(
            loc=mean,
            scale_diag=std,
        )
        actions = probability_distribution.sample(seed=key)
        log_probability = probability_distribution.log_prob(actions)
        entropy = probability_distribution.entropy()
    else:
        probability_distribution = distrax.Normal(
            loc=mean,
            scale=std,
        )
        actions = probability_distribution.sample(seed=key)
        log_probability = jnp.sum(probability_distribution.log_prob(actions), axis=-1)
        entropy = jnp.sum(probability_distribution.entropy(), axis=-1)
    return actions, log_probability, entropy


@functools.partial(jax.jit, static_argnames=["multivariate"])
def evaluate_action(
    mean: jax.Array,
    std: jax.Array,
    action: jax.Array,
    multivariate: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if multivariate:
        probability_distribution = distrax.MultivariateNormalDiag(
            loc=mean,
            scale_diag=std,
        )
        log_probability = probability_distribution.log_prob(action)
        entropy = probability_distribution.entropy()
    else:
        probability_distribution = distrax.Normal(
            loc=mean,
            scale=std,
        )
        log_probability = jnp.sum(probability_distribution.log_prob(action), axis=-1)
        entropy = jnp.sum(probability_distribution.entropy(), axis=-1)
    return log_probability, entropy


@functools.partial(jax.jit, static_argnames=["episode_length"])
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None), out_axes=(0, 0))
def calculate_advantage(
    rewards: jax.Array,
    values: jax.Array,
    mask: jax.Array,
    episode_length: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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


# Vmapped Replay Function:
@functools.partial(jax.jit, static_argnames=["apply_fn"])
@functools.partial(jax.vmap, in_axes=(None, None, None, 1, 1), out_axes=(1, 1, 1, 1, 1))
def replay(
    model_params: FrozenDict,
    apply_fn: Callable[..., Any],
    statistics_state: rs.RunningStatisticsState,
    model_input: jax.typing.ArrayLike,
    actions: jax.typing.ArrayLike,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mean, std, values = forward_pass(
        model_params,
        apply_fn,
        statistics_state,
        model_input,
    )
    log_probability, entropy = evaluate_action(mean, std, actions)
    return (
        jnp.squeeze(values),
        jnp.squeeze(log_probability),
        jnp.squeeze(entropy),
        jnp.squeeze(mean),
        jnp.squeeze(std),
    )


@functools.partial(jax.jit, static_argnames=["apply_fn"])
def loss_function(
    model_params: FrozenDict,
    apply_fn: Callable[..., Any],
    statistics_state: rs.RunningStatisticsState,
    model_input: jax.typing.ArrayLike,
    actions: jax.typing.ArrayLike,
    advantages: jax.typing.ArrayLike,
    returns: jax.typing.ArrayLike,
    previous_log_probability: jax.typing.ArrayLike,
    previous_mean: jax.typing.ArrayLike,
    previous_std: jax.typing.ArrayLike,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    def _calculate_kl_divergance(previous_mean, previous_std, mean, std):
        kl = jnp.sum(
            jnp.log(std / previous_std + 1.0e-5)
            + (jnp.square(previous_std) + jnp.square(previous_mean - mean))
            / (2.0 * jnp.square(std))
            - 0.5,
            axis=-1,
        )
        return jnp.mean(kl)

    # Algorithm Coefficients:
    value_coeff = 0.5
    entropy_coeff = 0.01
    clip_coeff = 0.2

    # Vmapped Replay:
    values, log_probability, entropy, mean, std = replay(
        model_params,
        apply_fn,
        statistics_state,
        model_input,
        actions,
    )
    # Calculate KL Divergence:
    kl = jax.lax.stop_gradient(
        _calculate_kl_divergance(
            previous_mean,
            previous_std,
            mean,
            std,
        ),
    )

    # Calculate Ratio: (Should this be No Grad?)
    log_ratios = log_probability - previous_log_probability
    ratios = jnp.exp(log_ratios)

    # Policy Loss:
    unclipped_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(
        1.0 - clip_coeff,
        ratios,
        1.0 + clip_coeff,
    )
    ppo_loss = -jnp.mean(
        jnp.minimum(unclipped_loss, clipped_loss),
    )

    # Value Loss:
    value_loss = value_coeff * jnp.mean(
        jnp.square(values - returns),
    )

    # Entropy Loss:
    entropy_loss = -entropy_coeff * jnp.mean(entropy)

    return ppo_loss + value_loss + entropy_loss, kl
