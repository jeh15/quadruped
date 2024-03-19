import os
import pickle
from absl import app
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from brax.io import html
from brax.envs.wrappers.training import wrap

import model
import model_utilities
import optimization_utilities
import save_checkpoint

import quadruped

jax.config.update("jax_enable_x64", True)


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
    episode_length = 500
    episode_mini_batch_length = 25
    num_envs = 2

    env = quadruped.Quadruped(backend='generalized')
    env = wrap(
        env=env,
        episode_length=episode_length,
        action_repeat=1,
    )

    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # Initize Networks:
    initial_key = jax.random.PRNGKey(key_seed)

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
    num_mini_batch = episode_length // episode_mini_batch_length
    ppo_steps = 10

    # Create a train state:
    schedule = optax.linear_schedule(
        init_value=learning_rate,
        end_value=end_learning_rate,
        transition_steps=ppo_steps * num_mini_batch * transition_steps,
        transition_begin=ppo_steps * num_mini_batch * transition_begin,
    )
    tx = optimizer(learning_rate=schedule)
    model_state = create_train_state(
        module=network,
        params=initial_params,
        optimizer=tx,
    )
    del initial_params

    # Learning Loop:
    training_length = 300
    key, env_key = jax.random.split(initial_key)
    checkpoint_enabled = False
    pickle_enabled = False
    visualization_enabled = False
    # Metrics:
    reward_history = []
    loss_history = []
    time_history = []
    epoch_time = []
    for iteration in range(training_length):
        # Episode Loop:
        states = reset_fn(env_key)
        state_history = [states]
        model_input_episode = []
        states_episode = []
        values_episode = []
        log_probability_episode = []
        actions_episode = []
        rewards_episode = []
        masks_episode = []
        episode_start = time.time()
        for environment_step in range(episode_length):
            print(f'Environment Step: {environment_step}')
            key, env_key = jax.random.split(env_key)
            model_input = states.obs
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
                actions,
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
            states = next_states
            state_history.append(states)

        episode_end = time.time() - episode_start

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

        # No Gradient Calculation:
        model_input = states.obs
        _, _, values = jax.lax.stop_gradient(
            model_utilities.forward_pass(
                model_params=model_state.params,
                apply_fn=model_state.apply_fn,
                x=model_input,
            ),
        )

        # Calculate Advantage:
        values_episode = jnp.concatenate(
            [values_episode, values],
            axis=1,
        )

        advantage_episode, returns_episode = jax.lax.stop_gradient(
            model_utilities.calculate_advantage(
                rewards_episode,
                values_episode,
                masks_episode,
                episode_length,
            )
        )

        # Pack Data:
        batch = (
            model_input_episode,
            actions_episode,
            advantage_episode,
            returns_episode,
            log_probability_episode,
        )

        # Update Function:
        loss_history = []
        train_start = time.time()
        model_state, loss = optimization_utilities.fit(
            model_state=model_state,
            batch=batch,
            mini_batch_size=num_mini_batch,
            ppo_steps=ppo_steps,
        )
        train_end = time.time() - train_start

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
            f'Epoch: {iteration} \t' +
            f'Average Reward: {average_reward} \t' +
            f'Loss: {loss} \t' +
            f'Average Value: {average_value} \t' +
            f'Learning Rate: {current_learning_rate}',
        )
        print(
            f'Episode time: {episode_end} \t' +
            f'Train time: {train_end}'
        )

        if visualization_enabled:
            if iteration % 25 == 0:
                html_string = html.render(
                    env.sys,
                    state_history[0].pipeline_state,
                    height="100vh", colab=False
                )
                html_path = os.path.join(
                    os.path.dirname(
                        os.path.dirname(__file__),
                    ),
                    "visualization/visualization.html",
                )
                with open(html_path, "w") as f:
                    f.writelines(html_string)

        if checkpoint_enabled:
            if iteration % 25 == 0:
                directory = os.path.dirname(__file__)
                checkpoint_path = os.path.join(directory, "checkpoints")
                save_checkpoint.save_checkpoint(
                    state=model_state,
                    path=checkpoint_path,
                    iteration=iteration,
                )

    print(f'The best reward of {best_reward} was achieved at iteration {best_iteration}')

    directory = os.path.dirname(__file__)
    if checkpoint_enabled:
        checkpoint_path = os.path.join(directory, "checkpoints")
        save_checkpoint.save_checkpoint(
            state=model_state,
            path=checkpoint_path,
            iteration=iteration,
        )

    # Pickle Metrics:
    if pickle_enabled:
        metrics_path = os.path.join(directory, "metrics")
        with open(metrics_path + "/metrics.pkl", "wb") as f:
            pickle.dump(
                {
                    "reward_history": reward_history,
                    "time_history": time_history,
                    "epoch_time": epoch_time,
                    "loss_history": loss_history,
                },
                f,
            )


if __name__ == '__main__':
    app.run(main)
