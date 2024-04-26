import os
from absl import app, flags

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from brax.io import html
import orbax.checkpoint as ocp
from brax.training.acme import running_statistics as rs

import model
import model_utilities
import control_utilities
import checkpoint
import render_utilities

import src.envs.barkour as barkour

jax.config.update('jax_default_device', jax.devices('cpu')[0])
jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', None, 'Checkpoint file name.', short_name='f')


def init_params(module, input_size, key):
    params = module.init(
        key,
        jnp.ones(input_size),
    )['params']
    return params


# @optax.inject_hyperparams allows introspection of learning_rate
@optax.inject_hyperparams
def optimizer(learning_rate):
    return optax.chain(
        optax.amsgrad(
            learning_rate=learning_rate,
        ),
    )


def create_train_state(module, params, optimizer):
    return train_state.TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=optimizer,
    )


def main(argv=None):
    # RNG Key:
    # key_seed = 42
    key_seed = 32

    # Create Environment:
    num_envs = 1
    env = barkour.BarkourEnv()
    episode_run_time = 20.0  # Seconds
    episode_length = int(episode_run_time / env.dt)

    # Initize Networks and States:
    initial_key = jax.random.PRNGKey(key_seed)
    key, env_key = jax.random.split(initial_key)

    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)
    states = reset_fn(env_key)

    network = model.ActorCriticNetworkVmap(
        action_space=env.action_size,
    )

    model_input_size = (num_envs, jnp.shape(states.obs)[0])
    initial_params = init_params(
        module=network,
        input_size=model_input_size,
        key=initial_key,
    )

    # Hyperparameters:
    learning_rate = 1e-3

    # Create a train state:
    tx = optimizer(learning_rate=learning_rate)
    model_state = create_train_state(
        module=network,
        params=initial_params,
        optimizer=tx,
    )
    del initial_params

    # Create Running Statistics:
    statistics_state = rs.init_state(
        jnp.zeros((model_input_size[-1],))
    )

    # Extract remap ranges:
    action_range = jnp.tile(
        A=jnp.array([-1.0, 1.0]),
        reps=(env.action_size, 1),
    )
    control_range = env.sys.actuator_ctrlrange

    # Create Checkpoint Manager:
    checkpoint_metadata = checkpoint.default_checkpoint_metadata()
    manager_options = checkpoint.default_checkpoint_options()
    checkpoint_directory = os.path.join(os.path.dirname(__file__), "checkpoints")
    manager = ocp.CheckpointManager(
        directory=checkpoint_directory,
        options=manager_options,
        item_names=('state', 'statistics_state', 'metadata'),
    )
    model_state, statistics_state, metadata = checkpoint.load_checkpoint(
        manager=manager,
        train_state=model_state,
        statistics_state=statistics_state,
        metadata=checkpoint_metadata,
    )

    key, env_key = jax.random.split(env_key)
    states = reset_fn(env_key)
    state_history = []
    metrics_history = []
    action_history = []
    state_history.append(states.pipeline_state)
    metrics_history.append(states)
    actions = jnp.zeros((env.action_size,))
    for environment_step in range(episode_length):
        key, env_key = jax.random.split(env_key)
        model_input = states.obs
        model_input = jnp.expand_dims(model_input, axis=0)
        mean, std, values = model_utilities.forward_pass(
            model_params=model_state.params,
            apply_fn=model_state.apply_fn,
            statistics_state=statistics_state,
            x=model_input,
        )
        actions, log_probability, entropy = model_utilities.select_action(
            mean=mean,
            std=std,
            key=env_key,
        )
        control_input = control_utilities.remap_controller(
            actions,
            action_range,
            control_range,
        )
        control_input = control_utilities.relative_controller(
            control_input,
            env._default_pose,
            control_range,
        )
        next_states = step_fn(
            states,
            jnp.squeeze(control_input),
        )

        # if states.metrics['knee_termination'] or states.metrics['base_termination'] == 1:
        #     break

        states = next_states
        metrics_history.append(states)
        action_history.append(actions)
        state_history.append(states.pipeline_state)

    # Only runs if started locally.
    # if not termination_flag:
    #     video_path = os.path.join(
    #         os.path.dirname(
    #             os.path.dirname(__file__),
    #         ),
    #         f"videos/output_{simulation_iteration}.mp4",
    #     )
    #     render_utilities.create_video(
    #         env=env,
    #         trajectory=state_history,
    #         filepath=video_path,
    #         height=720,
    #         width=1280,
    #         camera=None,
    #     )

    html_string = html.render(
        env.sys,
        state_history,
        height="100vh",
        colab=False,
    )
    html_path = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__),
        ),
        "visualization/visualization.html",
    )
    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)
