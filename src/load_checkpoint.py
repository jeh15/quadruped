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

import unitree_brax


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
    key_seed = 42

    # Create Environment:
    episode_length = 500
    num_envs = 1
    env = unitree_brax.Unitree(backend='mjx')

    # Initize Networks and States:
    initial_key = jax.random.PRNGKey(key_seed)
    key, env_key = jax.random.split(initial_key)

    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)
    states = reset_fn(env_key)


    network = model.ActorCriticNetworkVmap(
        action_space=env.action_size,
    )

    initial_params = init_params(
        module=network,
        input_size=(num_envs, env.observation_size + env.action_size),
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
        jnp.zeros((env.observation_size + env.action_size),)
    )

    # Extract remap ranges:
    action_range = jnp.tile(
        A=jnp.array([-1.0, 1.0]),
        reps=(env.action_size, 1),
    )
    control_range = env.sys.actuator_ctrlrange

    # Create Running Statistics:
    statistics_state = rs.init_state(
        jnp.zeros((env.observation_size + env.action_size),)
    )

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

    state_history = []
    metrics_history = []
    action_history = []
    state_history.append(states.pipeline_state)
    metrics_history.append(states)
    actions = jnp.zeros((env.action_size,))
    for environment_step in range(episode_length):
            key, env_key = jax.random.split(env_key)
            model_input = jnp.concatenate([states.obs, jnp.squeeze(actions)])
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
            next_states = step_fn(
                states,
                jnp.squeeze(control_input),
            )
            if environment_step % 100 == 0:
                pass

            # if states.metrics['knee_termination'] or states.metrics['base_termination'] == 1:
            #     break

            states = next_states
            metrics_history.append(states)
            action_history.append(actions)
            state_history.append(states.pipeline_state)

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
