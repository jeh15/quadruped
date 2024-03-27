import functools

import jax
import jax.numpy as jnp

from model_utilities import loss_function


@functools.partial(jax.jit, static_argnames=['ppo_steps'])
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
):
    gradient_function = jax.value_and_grad(loss_function, has_aux=True)
    # PPO Optimixation Loop:
    for ppo_step in range(ppo_steps):
        (loss, kl), gradients = gradient_function(
            model_state.params,
            model_state.apply_fn,
            statistics_state,
            model_input,
            actions,
            advantages,
            returns,
            previous_log_probability,
            previous_mean,
            previous_std,
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

    return model_state, loss
