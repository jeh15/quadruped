import os

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import orbax.checkpoint as ocp
from brax.training.acme import running_statistics, specs
from brax.io import html

from src.envs import barkour
from src.algorithms.ppo import checkpoint_utilities
from src.algorithms.ppo import network_utilities as ppo_networks
from src.algorithms.ppo.network_utilities import PPONetworkParams
from src.algorithms.ppo.train import TrainState

jax.config.update("jax_enable_x64", True)


def main(argv=None):
    # Load Metadata:
    unique_id = 'tusken-parsec-16'
    checkpoint_direrctory = os.path.join(
        os.path.dirname(__file__),
        f"checkpoints/{unique_id}",
    )
    manager = ocp.CheckpointManager(
        directory=checkpoint_direrctory,
        options=checkpoint_utilities.default_checkpoint_options(),
        item_names=(
            'network_metadata',
            'loss_metadata',
            'training_metadata',
        ),
    )

    metadata = checkpoint_utilities.load_checkpoint(
        manager=manager,
        network_metadata=checkpoint_utilities.empty_network_metadata(),
        loss_metadata=checkpoint_utilities.empty_loss_metadata(),
        training_metadata=checkpoint_utilities.empty_training_metadata(),
    )
    network_metadata = metadata.network_metadata
    loss_metadata = metadata.loss_metadata
    training_metadata = metadata.training_metadata

    env = barkour.BarkourEnv()

    # Restore Networks:
    policy_layer_sizes = (network_metadata.policy_layer_size,) * network_metadata.policy_depth
    value_layer_sizes = (network_metadata.value_layer_size,) * network_metadata.value_depth
    if training_metadata.normalize_observations:
        normalization_fn = running_statistics.normalize
    else:
        normalization_fn = lambda x, y: x

    network = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        input_normalization_fn=normalization_fn,
        policy_layer_sizes=policy_layer_sizes,
        value_layer_sizes=value_layer_sizes,
        activation=eval(network_metadata.activation),
        kernel_init=eval(network_metadata.kernel_init),
    )
    optimizer = eval(training_metadata.optimizer)

    # Create Keys and Structures:
    key = jax.random.key(training_metadata.seed)
    init_params = PPONetworkParams(
        policy_params=network.policy_network.init(key),
        value_params=network.value_network.init(key),
    )
    # Can't pass optimizer function to device_put_replicated:
    train_state = TrainState(
        opt_state=optimizer.init(init_params),
        params=init_params,
        normalization_params=running_statistics.init_state(
            specs.Array(env.observation_size, jnp.dtype('float32'))
        ),
        env_steps=0,
    )

    # Restore Train State:
    manager = ocp.CheckpointManager(
        directory=checkpoint_direrctory,
        options=checkpoint_utilities.default_checkpoint_options(),
        item_names=(
            'train_state',
        ),
    )
    restored_train_state = checkpoint_utilities.load_checkpoint(
        manager=manager,
        train_state=train_state,
    )
    train_state = restored_train_state.train_state

    # Construct Policy:
    make_policy = ppo_networks.make_inference_fn(ppo_networks=network)
    params = (
        train_state.normalization_params, train_state.params.policy_params,
    )
    inference_function = make_policy(params)
    inference_fn = jax.jit(inference_function)

    # Run Inference:
    env = barkour.BarkourEnv()
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state = reset_fn(jax.random.PRNGKey(0))

    x_vel = 1.0
    y_vel = 0.0
    ang_vel = 0.0

    the_command = jnp.array([x_vel, y_vel, ang_vel])

    # initialize the state
    rng = jax.random.PRNGKey(0)
    state.info['command'] = the_command
    state_history = [state.pipeline_state]

    # grab a trajectory
    n_steps = 500

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, ctrl)
        state_history.append(state.pipeline_state)

    html_string = html.render(
        env.sys,
        state_history,
        height="100vh",
        colab=False,
    )
    html_path = os.path.join(
        os.path.dirname(__file__),
        "visualization/visualization.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    main()
