import os
import pickle
import time
from absl import app
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import brax
from brax.envs.wrappers import training as wrapper
from brax.envs.base import Env

import model
import model_utilities
import ur5e
import custom_wrapper
import visualize
import save_checkpoint

# jax.config.update("jax_enable_x64", True)
dtype = jnp.float32


def create_environment(
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    **kwargs,
) -> Env:
    """Creates an environment from the registry.
    Args:
        episode_length: length of episode
        action_repeat: how many repeated actions to take per environment step
        auto_reset: whether to auto reset the environment after an episode is done
        batch_size: the number of environments to batch together
        **kwargs: keyword argments that get passed to the Env class constructor
    Returns:
        env: an environment
    """
    env = ur5e.ur5e(**kwargs)

    if episode_length is not None:
        env = wrapper.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = wrapper.VmapWrapper(env, batch_size)
    if auto_reset:
        env = custom_wrapper.AutoResetWrapper(env)

    return env


def init_params(module, input_size, key):
    params = module.init(
        key,
        jnp.ones(input_size),
        jax.random.split(key, input_size[0]),
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
    episode_length = 200
    num_envs = 32
    env = create_environment(
        episode_length=episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=num_envs,
        backend='generalized'
    )
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # Initize Networks:
    initial_key = jax.random.PRNGKey(key_seed)

    # Filepath:
    filename = "models/universal_robots/scene_brax.xml"
    filepath = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__),
            ),
        ),
        filename,
    )
    pipeline_model = brax.io.mjcf.load(filepath)
    pipeline_model = pipeline_model.replace(dt=0.002)

    network = model.ActorCriticNetworkVmap(
        action_space=env.action_size,
        nodes=5,
        sys=pipeline_model,
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
    transition_begin = 200
    ppo_steps = 5

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

    # Learning Loop:
    training_length = 300
    key, env_key = jax.random.split(initial_key)
    visualize_flag = False
    checkpoint_enabled = True
    pickle_enabled = False
    # Metrics:
    reward_history = []
    loss_history = []
    time_history = []
    epoch_time = []
    start_time = time.time()
    for iteration in range(training_length):
        states = reset_fn(env_key)
        state_history = []
        state_history.append(states)
        model_input_episode = []
        states_episode = []
        values_episode = []
        log_probability_episode = []
        actions_episode = []
        rewards_episode = []
        masks_episode = []
        keys_episode = []
        iteration_start_time = time.time()
        # Episode Loop:
        for environment_step in range(episode_length):
            key, env_key = jax.random.split(env_key)
            model_key = jax.random.split(env_key, num_envs)
            model_input = states.obs
            mean, std, values = model_utilities.forward_pass(
                model_params=model_state.params,
                apply_fn=model_state.apply_fn,
                x=model_input,
                key=model_key,
            )
            actions, log_probability, entropy = model_utilities.select_action(
                mean=mean,
                std=std,
                key=env_key,
            )
            next_states = step_fn(
                states,
                actions,
                env_key,
            )
            states_episode.append(states.obs)
            values_episode.append(jnp.squeeze(values))
            log_probability_episode.append(jnp.squeeze(log_probability))
            actions_episode.append(jnp.squeeze(actions))
            rewards_episode.append(jnp.squeeze(states.reward))
            masks_episode.append(jnp.where(states.done == 0, 1.0, 0.0))
            model_input_episode.append(model_input)
            keys_episode.append(model_key)
            states = next_states
            state_history.append(states)

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
        keys_episode = jnp.swapaxes(
            jnp.asarray(keys_episode), axis1=1, axis2=0,
        )

        # No Gradient Calculation:
        model_input = states.obs
        model_key = jax.random.split(env_key, num_envs)
        _, _, values = jax.lax.stop_gradient(
            model_utilities.forward_pass(
                model_params=model_state.params,
                apply_fn=model_state.apply_fn,
                x=model_input,
                key=model_key,
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

        # Update Function:
        model_state, loss = model_utilities.train_step(
            model_state=model_state,
            model_input=model_input_episode,
            actions=actions_episode,
            advantages=advantage_episode,
            returns=returns_episode,
            previous_log_probability=log_probability_episode,
            keys=keys_episode,
            batch_size=num_envs,
            episode_length=episode_length,
            ppo_steps=ppo_steps,
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

        if checkpoint_enabled:
            if iteration % 25 == 0:
                directory = os.path.dirname(__file__)
                checkpoint_path = os.path.join(directory, "checkpoints")
                save_checkpoint.save_checkpoint(
                    state=model_state,
                    path=checkpoint_path,
                    iteration=iteration,
                )

        if visualize_flag:
            if iteration % 25 == 0:
                video_filepath = os.path.join(
                    os.path.dirname(__file__),
                    f"unitree_training_iteration_{iteration}",
                )
                visualize.create_video(
                    sys=pipeline_model,
                    states=state_history,
                    width=1280,
                    height=720,
                    name="Unitree a1",
                    filepath=video_filepath,
                )

        current_learning_rate = model_state.opt_state.hyperparams['learning_rate']
        print(f'Epoch: {iteration} \t Average Reward: {average_reward} \t Loss: {loss} \t Average Value: {average_value} \t Learning Rate: {current_learning_rate}')

        # Save Metrics:
        iteration_time = time.time() - iteration_start_time
        elapsed_time = time.time() - start_time
        reward_history.append(rewards_episode)
        loss_history.append(loss)
        time_history.append(elapsed_time)
        epoch_time.append(iteration_time)


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
