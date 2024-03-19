from typing import Tuple
import functools

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

import model_utilities

Batch = Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


@functools.partial(
    jax.jit, static_argnames=["ppo_steps"]
)
def train_step(
    model_state: TrainState,
    batch: Batch,
    ppo_steps: int,
) -> Tuple[TrainState, jnp.ndarray]:
    # PPO Optimixation Loop:
    def ppo_loop(carry, xs):
        model_state = carry
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

        # Pack carry and data:
        carry = model_state
        data = loss
        return carry, data

    # Compute gradient function:
    gradient_function = jax.value_and_grad(model_utilities.loss_function)

    # Unpack Batch:
    model_input, actions, advantages, returns, previous_log_probability = batch

    # Loop over PPO steps:
    carry, data = jax.lax.scan(
        f=ppo_loop,
        init=(model_state),
        xs=None,
        length=ppo_steps,
    )

    # Unpack carry and data:
    model_state = carry
    loss = data
    loss = jnp.mean(loss)

    return model_state, loss


@functools.partial(jax.jit, static_argnames=["mini_batch_size"])
def create_mini_batch(array: jax.Array, mini_batch_size: int) -> jnp.ndarray:
    return jnp.asarray(
        jnp.split(array, mini_batch_size, axis=1),
    )


def fit(
    model_state: TrainState,
    batch: Batch,
    mini_batch_size: int,
    ppo_steps: int,
) -> Tuple[TrainState, jnp.ndarray]:
    # Unpack Batch:
    model_input, actions, advantages, returns, previous_log_probability = batch

    # Split Batch and create mini batches along the episode dimension:
    model_input = create_mini_batch(
        model_input, mini_batch_size,
    )
    actions = create_mini_batch(
        actions, mini_batch_size,
    )
    advantages = create_mini_batch(
        advantages, mini_batch_size,
    )
    returns = create_mini_batch(
        returns, mini_batch_size,
    )
    previous_log_probability = create_mini_batch(
        previous_log_probability, mini_batch_size,
    )

    loss = []
    for i in range(mini_batch_size):
        # Pack mini batch:
        mini_batch = (
            model_input[i],
            actions[i],
            advantages[i],
            returns[i],
            previous_log_probability[i]
        )
        model_state, loss_mini_batch = train_step(
            model_state,
            mini_batch,
            ppo_steps,
        )
        loss.append(loss_mini_batch)

    return model_state, jnp.mean(jnp.asarray(loss))
