from typing import Callable, Tuple, Any
import functools

import jax
import jax.numpy as jnp
import distrax

# Types:
from flax.core import FrozenDict
PRNGKey = jax.Array


@functools.partial(jax.jit, static_argnames=["apply_fn"])
def forward_pass(
    model_params: FrozenDict,
    apply_fn: Callable[..., Any],
    x: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    rewards: jax.typing.ArrayLike,
    values: jax.typing.ArrayLike,
    mask: jax.typing.ArrayLike,
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


@functools.partial(jax.jit, static_argnames=["apply_fn"])
def loss_function(
    model_params: FrozenDict,
    apply_fn: Callable[..., Any],
    model_input: jax.typing.ArrayLike,
    actions: jax.typing.ArrayLike,
    advantages: jax.typing.ArrayLike,
    returns: jax.typing.ArrayLike,
    previous_log_probability: jax.typing.ArrayLike,
) -> jnp.ndarray:
    print('Compiling Loss Function')
    # Algorithm Coefficients:
    value_coeff = 0.5
    entropy_coeff = 0.01
    clip_coeff = 0.2

    # Vmapped Replay:
    values, log_probability, entropy = replay(
        model_params,
        apply_fn,
        model_input,
        actions,
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

    return ppo_loss + value_loss + entropy_loss


# Vmapped Replay Function:
@functools.partial(jax.jit, static_argnames=["apply_fn"])
@functools.partial(jax.vmap, in_axes=(None, None, 1, 1), out_axes=(1, 1, 1))
def replay(
    model_params: FrozenDict,
    apply_fn: Callable[..., Any],
    model_input: jax.typing.ArrayLike,
    actions: jax.typing.ArrayLike,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    print('Compiling Replay Function')
    mean, std, values = forward_pass(
        model_params,
        apply_fn,
        model_input,
    )
    log_probability, entropy = evaluate_action(mean, std, actions)
    return jnp.squeeze(values), jnp.squeeze(log_probability), jnp.squeeze(entropy)


# Serialized Replay Function:
@functools.partial(jax.jit, static_argnames=["apply_fn", "length"])
def replay_serial(
    model_params: FrozenDict,
    apply_fn: Callable[..., Any],
    model_input: jax.Array,
    actions: jax.Array,
    length: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def forward_pass_rollout(
            carry: None,
            xs: Tuple[jax.Array, jax.Array],
    ) -> Tuple[None, Tuple[jax.Array, jax.Array, jax.Array]]:
        model_input, actions = xs
        mean, std, values = forward_pass(
            model_params,
            apply_fn,
            model_input,
        )
        log_probability, entropy = evaluate_action(
            mean,
            std,
            actions,
        )
        carry = None
        data = (jnp.squeeze(values), log_probability, entropy)
        return carry, data

    # Scan over replay:
    _, data = jax.lax.scan(
        forward_pass_rollout,
        None,
        (model_input, actions),
        length,
    )
    values, log_probability, entropy = data
    values = jnp.swapaxes(
        jnp.asarray(values), axis1=1, axis2=0,
    )
    log_probability = jnp.swapaxes(
        jnp.asarray(log_probability), axis1=1, axis2=0,
    )
    entropy = jnp.swapaxes(
        jnp.asarray(entropy), axis1=1, axis2=0,
    )
    return values, log_probability, entropy


# @functools.partial(
#     jax.jit, static_argnames=["ppo_steps"]
# )
# def train_step(
#     model_state: TrainState,
#     model_input: jax.typing.ArrayLike,
#     actions: jax.typing.ArrayLike,
#     advantages: jax.typing.ArrayLike,
#     returns: jax.typing.ArrayLike,
#     previous_log_probability: jax.typing.ArrayLike,
#     ppo_steps: int,
# ) -> Tuple[TrainState, jnp.ndarray]:
#     # PPO Optimixation Loop:
#     def ppo_loop(carry, xs):
#         model_state = carry
#         loss, gradients = gradient_function(
#             model_state.params,
#             model_state.apply_fn,
#             model_input,
#             actions,
#             advantages,
#             returns,
#             previous_log_probability,
#         )
#         model_state = model_state.apply_gradients(grads=gradients)

#         # Pack carry and data:
#         carry = model_state
#         data = loss
#         return carry, data

#     gradient_function = jax.value_and_grad(loss_function)

#     carry, data = jax.lax.scan(
#         f=ppo_loop,
#         init=(model_state),
#         xs=None,
#         length=ppo_steps,
#     )

#     # Unpack carry and data:
#     model_state, _ = carry
#     loss = data
#     loss = jnp.mean(loss)

#     return model_state, loss

@functools.partial(jax.jit, static_argnames=['ppo_steps'])
def train_step(
    model_state,
    model_input,
    actions,
    advantages,
    returns,
    previous_log_probability,
    ppo_steps,
):
    # Print Statement:
    print('Compiling Train Step...')
    gradient_function = jax.value_and_grad(loss_function)
    # PPO Optimixation Loop:
    for ppo_step in range(ppo_steps):
        loss, gradients = gradient_function(
            model_state.params,
            model_state.apply_fn,
            model_input,
            actions,
            advantages,
            returns,
            previous_log_probability,
        )
        model_state = model_state.apply_gradients(grads=gradients)

    return model_state, loss
