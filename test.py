import os
import functools

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from brax.io import html

from src.envs import barkour
from src.algorithms.ppo import network_utilities as ppo_networks
from src.algorithms.ppo.loss_utilities import loss_function
from src.algorithms.ppo.train import train

jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=4)


def main(argv=None):
    # Initialize Functions with Params:
    randomization_fn = barkour.domain_randomize
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_layer_sizes=(128, 128, 128, 128),
        value_layer_sizes=(256, 256, 256, 256, 256),
        activation=nn.tanh,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )
    loss_fn = functools.partial(
        loss_function,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=False,
    )
    env = barkour.BarkourEnv()
    eval_env = barkour.BarkourEnv()

    def progress_fn(num_steps, metrics):
        print(
            f'Num Steps: {num_steps} \t'
            f'Episode Reward: {metrics["eval/episode_reward"]:.3f} \t'
        )
        if num_steps > 0:
            print(
                f'Training Loss: {metrics["training/loss"]:.3f} \t'
                f'Policy Loss: {metrics["training/policy_loss"]:.3f} \t'
                f'Value Loss: {metrics["training/value_loss"]:.3f} \t'
                f'Entropy Loss: {metrics["training/entropy_loss"]:.3f} \t'
                f'Training Wall Time: {metrics["training/walltime"]:.3f} \t'
            )
        print('\n')

    train_fn = functools.partial(
        train,
        num_epochs=10,
        num_training_steps=34,
        episode_length=1000,
        num_policy_steps=20,
        action_repeat=1,
        num_envs=8192,
        num_evaluation_envs=128,
        num_evaluations=1,
        deterministic_evaluation=True,
        reset_per_epoch=False,
        seed=0,
        batch_size=256,
        num_minibatches=32,
        num_ppo_iterations=4,
        normalize_observations=True,
        network_factory=make_networks_factory,
        loss_function=loss_fn,
        progress_fn=progress_fn,
        randomization_fn=randomization_fn,
    )

    policy_generator, params, metrics = train_fn(
        environment=env,
        evaluation_environment=eval_env,
    )

    inference_fn = policy_generator(params)
    jit_inference_fn = jax.jit(inference_fn)

    # Brax Env:
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
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
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
