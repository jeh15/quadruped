import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from brax.training.acme.types import NestedArray

from model_utilities import loss_function


class TrainingDataClass(NamedTuple):
    model_input: NestedArray
    actions: NestedArray
    advantages: NestedArray
    returns: NestedArray
    previous_log_probability: NestedArray
    previous_mean: NestedArray
    previous_std: NestedArray


@functools.partial(jax.jit, static_argnames=['ppo_steps', 'num_minibatches'])
def train_step(
    model_state,
    statistics_state,
    model_input,
    actions,
    advantages,
    returns,
    previous_log_probability,
    previous_mean,
    previous_std,
    ppo_steps,
    num_minibatches,
    key,
):
    gradient_function = jax.value_and_grad(loss_function, has_aux=True)

    # Minibatch Step:
    def minibatch_step(carry, data, statistics_state):
        model_state = carry

        (loss, kl), gradients = gradient_function(
            model_state.params,
            model_state.apply_fn,
            statistics_state,
            data.model_input,
            data.actions,
            data.advantages,
            data.returns,
            data.previous_log_probability,
            data.previous_mean,
            data.previous_std,
        )

        # Calculate Learning Rate:
        desired_kl = 0.01
        learning_rate = model_state.opt_state.hyperparams['learning_rate']
        learning_rate = jnp.where(
            kl > desired_kl * 2.0,
            jnp.max(
                jnp.array([1e-5, learning_rate / 1.5]),
            ),
            learning_rate,
        )
        learning_rate = jnp.where(
            jnp.logical_and(kl < desired_kl / 2.0, kl > 0.0),
            jnp.min(
                jnp.array([1e-2, learning_rate * 1.5]),
            ),
            learning_rate,
        )  # type: ignore

        # Update Learning Rate:
        model_state.opt_state.hyperparams['learning_rate'] = learning_rate
        model_state.tx.update(model_state.params, model_state.opt_state)

        # Apply Gradients:
        model_state = model_state.apply_gradients(grads=gradients)

        # Pack carry and data:
        carry = model_state
        metrics = loss, kl, learning_rate

        return carry, metrics

    def sgd_step(
        carry,
        unused_t,
        data,
        statistics_state,
    ):
        model_state, key = carry
        key, key_perm = jax.random.split(key)

        def shuffle_data(x):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(shuffle_data, data)
        model_state, metrics = jax.lax.scan(
            f=functools.partial(
                minibatch_step, statistics_state=statistics_state,
            ),
            init=model_state,
            xs=shuffled_data,
            length=num_minibatches,
        )

        return (model_state, key), metrics

    data = TrainingDataClass(
        model_input=model_input,
        actions=actions,
        advantages=advantages,
        returns=returns,
        previous_log_probability=previous_log_probability,
        previous_mean=previous_mean,
        previous_std=previous_std,
    )

    (model_state, _), metrics = jax.lax.scan(
        f=functools.partial(
            sgd_step, data=data, statistics_state=statistics_state,
        ),
        init=(model_state, key),
        xs=None,
        length=ppo_steps,
    )

    return model_state, metrics


# @functools.partial(jax.jit, static_argnames=['ppo_steps'])
# def train_step(
#     model_state,
#     statistics_state,
#     model_input,
#     actions,
#     advantages,
#     returns,
#     previous_log_probability,
#     previous_mean,
#     previous_std,
#     ppo_steps,
# ):
#     gradient_function = jax.value_and_grad(loss_function, has_aux=True)
#     # PPO Optimixation Loop:
#     for ppo_step in range(ppo_steps):
#         (loss, kl), gradients = gradient_function(
#             model_state.params,
#             model_state.apply_fn,
#             statistics_state,
#             model_input,
#             actions,
#             advantages,
#             returns,
#             previous_log_probability,
#             previous_mean,
#             previous_std,
#         )

#         # Calculate Learning Rate:
#         desired_kl = 0.01
#         learning_rate = model_state.opt_state.hyperparams['learning_rate']
#         learning_rate = jnp.where(
#             kl > desired_kl * 2.0,
#             jnp.max(
#                 jnp.array([1e-5, learning_rate / 1.5]),
#             ),
#             learning_rate,
#         )
#         learning_rate = jnp.where(
#             jnp.logical_and(kl < desired_kl / 2.0, kl > 0.0),
#             jnp.min(
#                 jnp.array([1e-2, learning_rate * 1.5]),
#             ),
#             learning_rate,
#         )  # type: ignore

#         # Update Learning Rate:
#         model_state.opt_state.hyperparams['learning_rate'] = learning_rate
#         model_state.tx.update(model_state.params, model_state.opt_state)

#         # Apply Gradients:
#         model_state = model_state.apply_gradients(grads=gradients)

#     return model_state, loss
