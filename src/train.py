import os
import pickle
from absl import app, flags
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from brax.envs.wrappers.training import wrap
from brax.training.acme import running_statistics as rs
import orbax.checkpoint as ocp

import model
import model_utilities
import optimization_utilities
import statistics_utilities
import control_utilities
import checkpoint

import unitree

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
flags.DEFINE_boolean('load_checkpoint', False, 'Load latest checkpoint.', short_name='l')


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

    best_reward = np.NINF
    best_iteration = 0

    # Create Environment:
    env = unitree.Unitree(backend='mjx')
    episode_run_time = 20.0  # Seconds
    batch_run_time = 0.5  # Seconds
    batch_iterator_length = int(batch_run_time / env.dt)
    episode_iterator_length = int(episode_run_time / batch_run_time)
    episode_length = int(episode_run_time / env.dt)
    num_envs = 1024 * 4
    env = wrap(
        env=env,
        episode_length=episode_length,
        action_repeat=1,
    )

    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # Model initialization:
    initial_key = jax.random.PRNGKey(key_seed)
    reset_key = jax.random.split(initial_key, num=num_envs)
    states = reset_fn(reset_key)

    network = model.ActorCriticNetworkVmap(
        action_space=env.action_size,
    )

    # Model Input: obs + action
    model_input_size = (jnp.shape(states.info['model_input']))
    initial_params = init_params(
        module=network,
        input_size=model_input_size,
        key=initial_key,
    )

    # Hyperparameters:
    learning_rate = 1e-3
    ppo_steps = 5

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

    # Create Checkpoint Manager:
    checkpoint_metadata = checkpoint.default_checkpoint_metadata()
    manager_options = checkpoint.default_checkpoint_options()
    checkpoint_directory = os.path.join(os.path.dirname(__file__), "checkpoints")
    manager = ocp.CheckpointManager(
        directory=checkpoint_directory,
        options=manager_options,
        item_names=('state', 'statistics_state', 'metadata'),
    )

    # Extract remap ranges:
    action_range = jnp.tile(
        A=jnp.array([-1.0, 1.0]),
        reps=(env.action_size, 1),
    )
    control_range = env.sys.actuator_ctrlrange

    iteration_step = 0
    if FLAGS.load_checkpoint is True:
        model_state, statistics_state, metadata = checkpoint.load_checkpoint(
            manager=manager,
            train_state=model_state,
            statistics_state=statistics_state,
            metadata=checkpoint_metadata,
        )
        iteration_step = metadata['iteration']

    # Learning Loop:
    training_length = 300
    key, env_key = jax.random.split(initial_key)
    checkpoint_enabled = True
    for iteration in range(training_length):
        # Episode Loop:
        # Different randomization for each environment:
        reset_key = jax.random.split(env_key, num=num_envs)
        # Same randomization for each environment:
        # reset_key = jnp.zeros((num_envs, 2), dtype=jnp.uint32)
        states = reset_fn(reset_key)
        state_history = [states]
        model_input_episode = []
        states_episode = []
        values_episode = []
        log_probability_episode = []
        actions_episode = []
        rewards_episode = []
        masks_episode = []
        metrics_episode = []
        mean_episode = []
        std_episode = []
        for environment_step in range(episode_iterator_length):
            for batch_step in range(batch_iterator_length):
                key, env_key = jax.random.split(env_key)
                model_input = states.info['model_input']
                statistics_state = statistics_utilities.update(
                    state=statistics_state,
                    x=model_input,
                )
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
                    control_input,
                )
                states_episode.append(states.obs)
                values_episode.append(jnp.squeeze(values))
                log_probability_episode.append(jnp.squeeze(log_probability))
                actions_episode.append(jnp.squeeze(actions))
                rewards_episode.append(jnp.squeeze(states.reward))
                masks_episode.append(
                    jnp.squeeze(
                        jnp.where(states.done == 0, 1.0, 0.0)
                    )
                )
                model_input_episode.append(model_input)
                mean_episode.append(jnp.squeeze(mean))
                std_episode.append(jnp.squeeze(std))
                # Save Metrics
                metrics_episode.append(states.metrics)
                states = next_states
                state_history.append(states)

            iteration_step += 1

            # Convert to Jax Arrays:
            states_episode = jnp.swapaxes(
                jnp.asarray(states_episode), axis1=1, axis2=0,
            )
            values_episode = jnp.swapaxes(
                jnp.asarray(values_episode), axis1=1, axis2=0,
            )
            log_probability_episode = jnp.swapaxes(
                jnp.asarray(log_probability_episode), axis1=1, axis2=0,
            )
            actions_episode = jnp.swapaxes(
                jnp.asarray(actions_episode), axis1=1, axis2=0,
            )
            rewards_episode = jnp.swapaxes(
                jnp.asarray(rewards_episode), axis1=1, axis2=0,
            )
            masks_episode = jnp.swapaxes(
                jnp.asarray(masks_episode), axis1=1, axis2=0,
            )
            model_input_episode = jnp.swapaxes(
                jnp.asarray(model_input_episode), axis1=1, axis2=0,
            )
            mean_episode = jnp.swapaxes(
                jnp.asarray(mean_episode), axis1=1, axis2=0,
            )
            std_episode = jnp.swapaxes(
                jnp.asarray(std_episode), axis1=1, axis2=0,
            )

            # No Gradient Calculation: (This is unneeded)
            model_input = states.info['model_input']
            _, _, values = model_utilities.forward_pass(
                model_params=model_state.params,
                apply_fn=model_state.apply_fn,
                statistics_state=statistics_state,
                x=model_input,
            )

            # Calculate Advantage:
            values_episode = jnp.concatenate(
                [values_episode, values],
                axis=1,
            )

            advantage_episode, returns_episode = model_utilities.calculate_advantage(
                rewards_episode,
                values_episode,
                masks_episode,
                batch_iterator_length,
            )

            # Update Function:
            model_state, loss = optimization_utilities.train_step(
                model_state,
                statistics_state,
                model_input_episode,
                actions_episode,
                advantage_episode,
                returns_episode,
                log_probability_episode,
                mean_episode,
                std_episode,
                ppo_steps,
            )

            average_reward = np.mean(
                np.sum(
                    (rewards_episode),
                    axis=1,
                ),
            )

            average_value = np.mean(
                np.mean(
                    (values_episode),
                    axis=1,
                ),
                axis=0
            )

            if average_reward >= best_reward:
                best_reward = average_reward
                best_iteration = iteration

            current_learning_rate = model_state.opt_state.hyperparams['learning_rate']
            print(
                f'Epoch: {iteration_step} \t' +
                f'Average Reward: {average_reward} \t' +
                f'Loss: {loss} \t' +
                f'Average Value: {average_value} \t' +
                f'Learning Rate: {current_learning_rate}',
            )

            # Reset for next policy run:
            state_history = [states]
            model_input_episode = []
            states_episode = []
            values_episode = []
            log_probability_episode = []
            actions_episode = []
            rewards_episode = []
            masks_episode = []
            metrics_episode = []
            mean_episode = []
            std_episode = []

            if checkpoint_enabled:
                checkpoint_metadata['iteration'] = iteration_step
                checkpoint.save_checkpoint(
                    manager=manager,
                    train_state=model_state,
                    statistics_state=statistics_state,
                    metadata=checkpoint_metadata,
                )

        print(f'Iteration: {iteration}')

    print(f'The best reward of {best_reward} was achieved at iteration {best_iteration}')

    if checkpoint_enabled:
        checkpoint.save_checkpoint(
            manager=manager,
            train_state=model_state,
            statistics_state=statistics_state,
            metadata=checkpoint_metadata,
        )


if __name__ == '__main__':
    app.run(main)
