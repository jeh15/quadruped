import functools
import time
from typing import Callable, Optional, Tuple

from absl import logging
import flax.struct
import jax
import jax.numpy as jnp
import flax
import optax
import orbax.checkpoint as ocp
import numpy as np

from brax import base
from brax import envs
from brax.envs.wrappers.training import wrap
from brax.training.acme import running_statistics
import src.module_types as types
from src.algorithms.ppo.network_utilities import PPONetworkParams

import src.algorithms.ppo.network_utilities as ppo_networks
import src.algorithms.ppo.loss_utilities as loss_utilities
import src.optimization_utilities as optimization_utilities
import src.training_utilities as trainining_utilities


InferenceParams = Tuple[running_statistics.NestedMeanStd, types.Params]

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainState:
    tx: optax.GradientTransformation
    opt_state: optax.OptState
    params: PPONetworkParams
    normalization_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def strip_weak_type(pytree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)
    return jax.tree_map(f, pytree)



"""
    Requires Network Factory with captured parameters.
    Requires Loss Function with captured parameters.


"""


def train(
    environment: envs.Env,
    num_epochs: int,
    episode_length: int,
    num_policy_steps: int = 10,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_evaluation_envs: int = 128,
    num_evaluations: int = 1,
    num_reset_per_evaluation: int = 0,
    deterministic_evaluation: bool = False,
    seed: int = 0,
    batch_size: int = 32,
    num_minibatches: int = 15,
    num_ppo_iterations: int = 4,
    normalize_observations: bool = True,
    normalize_advantages: bool = False,
    network_factory: types.NetworkFactory[ppo_networks.PPONetworks] = ppo_networks.make_ppo_networks,
    optimizer: optax.GradientTransformation = optax.adam(1e-4),
    loss_function: Callable[..., Tuple[jnp.ndarray, types.Metrics]] =
    loss_utilities.loss_function,
    progress_fn: Callable[[int, types.Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
    assert batch_size * num_minibatches % num_envs == 0

    # JAX Device management:
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    device_count = local_devices_to_use * process_count

    assert num_envs % device_count == 0

    # Training Loop Iteration Parameters:
    num_steps_per_train_step = (
        batch_size * num_minibatches * episode_length * action_repeat
    )
    num_steps_per_epoch = (
        batch_size * num_minibatches * num_policy_steps * episode_length
    )

    # Generate Random Key:
    key = jax.random.key(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, env_key, eval_key = jax.random.split(local_key, 3)
    policy_key, value_key = jax.random.split(global_key)
    del global_key

    # Initialize Environment:
    _randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // device_count
        randomization_key = jax.random.split(env_key, randomization_batch_size)
        _randomization_fn = functools.partial(
            randomization_fn, rng=randomization_key,
        )

    env = wrap(
        env=environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=_randomization_fn,
    )

    # vmap for multiple devices:
    reset_fn = jax.jit(jax.vmap(env.reset))
    envs_key = jax.random.split(env_key, num_envs // process_count)
    envs_key = jnp.reshape(
        envs_key, (local_devices_to_use, -1) + envs_key.shape[1:],
    )
    env_state = reset_fn(envs_key)

    # Initialize Normalization Function:
    normalization_fn = lambda x, y: x
    if normalize_observations:
        normalization_fn = running_statistics.normalize

    # Initialize Network:
    # functools.partial network_factory to capture parameters:
    ppo_networks = network_factory(
        observation_size=env.observation_size,
        action_size=env.action_size,
        input_normalization_fn=normalization_fn,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_networks=ppo_networks)

    # Initialize Loss Function:
    # functools.partial loss_function to capture parameters:
    loss_fn = functools.partial(
        loss_function,
        ppo_networks=ppo_networks,
    )

    gradient_udpate_fn = optimization_utilities.gradient_update_fn(
        loss_fn=loss_fn,
        optimizer=optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
        has_aux=True,
    )

    def minibatch_step(
        carry,
        data: types.Transition,
        normalization_params: running_statistics.RunningStatisticsState,
    ):
        opt_state, params, key = carry
        key, subkey = jax.random.split(key)
        (_, metrics), params, opt_state = gradient_udpate_fn(
            params,
            normalization_params,
            data,
            subkey,
            opt_state=opt_state,
        )

        return (opt_state, params, key), metrics

    def sgd_step(
        carry,
        unusted_t,
        data: types.Transition,
        normalization_params: running_statistics.RunningStatisticsState,
    ):
        opt_state, params, key = carry
        key, permutation_key, grad_key = jax.random.split(key)

        # Shuffle Data:
        def permute_data(x: jnp.ndarray):
            x = jax.random.permutation(permutation_key, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(permute_data, data)
        (opt_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalization_params=normalization_params),
            (opt_state, params, grad_key),
            shuffled_data,
            length=num_minibatches,
        )

        return (opt_state, params, key), metrics

    def training_step(
        carry: Tuple[TrainState, envs.State, types.PRNGKey],
        unused_t,
    ) -> Tuple[Tuple[TrainState, envs.State, types.PRNGKey], types.Metrics]:
        train_state, state, key = carry
        # Changed key order: (Does this matter?)
        next_key, sgd_key, policy_step_key = jax.random.split(key, 3)

        policy_fn = make_policy([
            train_state.normalization_params, train_state.params.policy_params,
        ])

        # Generates Episode Data:
        def f(carry, unused_t):
            state, key = carry
            key, subkey = jax.random.split(key)
            next_state, data = trainining_utilities.unroll_policy_steps(
                env=env,
                state=state,
                policy=policy_fn,
                key=key,
                num_steps=num_policy_steps,
                extra_fields=('truncation'),
            )
            return (next_state, subkey), data

        (state, _), data = jax.lax.scan(
            f,
            (state, policy_step_key),
            None,
            length=batch_size * num_minibatches // num_envs,
        )

        # Swap leading dimensions: (T, B, ...) -> (B, T, ...)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data,
        )

        # Update Normalization:
        normalization_params = running_statistics.update(
            train_state.normalization_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        (opt_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalization_params=normalization_params,
            ),
            (train_state.opt_state, train_state.params, sgd_key),
            None,
            length=num_ppo_iterations,
        )

        new_train_state = TrainState(
            tx=train_state.tx,
            opt_state=opt_state,
            params=params,
            normalization_params=normalization_params,
            env_steps=train_state.env_steps + num_steps_per_train_step,
        )

        return (new_train_state, state, next_key), metrics

    def training_epoch(
        train_state: TrainState,
        state: envs.State,
        key: types.PRNGKey,
    ) -> Tuple[TrainState, envs.State, types.Metrics]:
        (train_state, state, _), loss_metrics = jax.lax.scan(
            training_step,
            (train_state, state, key),
            None,
            length=num_steps_per_epoch,
        )
        return train_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)