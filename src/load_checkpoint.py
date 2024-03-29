import os
from absl import app, flags
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from brax.io import html
import orbax.checkpoint

import model
import model_utilities

import quadruped


jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', None, 'Checkpoint file name.', short_name='f')
flags.mark_flag_as_required('filename')


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
    env = quadruped.Quadruped(backend='generalized')

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
        input_size=(num_envs, env.observation_size),
        key=initial_key,
    )

    # Hyperparameters:
    learning_rate = 1e-3
    end_learning_rate = 1e-6
    transition_steps = 100
    transition_begin = 100
    ppo_steps = 10

    # Create a train state:
    schedule = optax.linear_schedule(
        init_value=learning_rate,
        end_value=end_learning_rate,
        transition_steps=ppo_steps * transition_steps,
        transition_begin=ppo_steps * transition_begin,
    )
    tx = optimizer(learning_rate=schedule)
    model_state = create_train_state(
        module=network,
        params=initial_params,
        optimizer=tx,
    )
    del initial_params

    target = {'model': model_state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_path = os.path.join(os.path.dirname(__file__), FLAGS.filename)
    model_state = orbax_checkpointer.restore(checkpoint_path, item=target)['model']

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
                x=model_input,
            )
            actions, log_probability, entropy = model_utilities.select_action(
                mean=mean,
                std=std,
                key=env_key,
            )
            next_states = step_fn(
                states,
                jnp.squeeze(actions),
            )
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
