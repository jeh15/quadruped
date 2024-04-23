import os

from datetime import datetime
import functools

import jax
import jax.numpy as jnp

from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model, html

import barkour

jax.config.update("jax_enable_x64", True)


def main(argv=None):
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, 10)

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128),
    )
    train_fn = functools.partial(
        ppo.train, num_timesteps=50_000_000, num_evals=10,
        reward_scaling=1, episode_length=1000, normalize_observations=True,
        action_repeat=1, unroll_length=20, num_minibatches=32,
        num_updates_per_batch=4, discounting=0.97, learning_rate=3.0e-4,
        entropy_cost=1e-2, num_envs=8192, batch_size=256,
        network_factory=make_networks_factory,
        randomization_fn=barkour.domain_randomize, seed=0)

    def progress(num_steps, metrics):
        print(f'Number of Steps: {num_steps} \t Episode Reward: {metrics["eval/episode_reward"]}')

    # Reset environments since internals may be overwritten by tracers from the
    # domain randomization function.
    env = barkour.BarkourEnv()
    eval_env = barkour.BarkourEnv()
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )

    # Save and reload params.x
    filename = 'mjx_brax_quadruped_policy'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    model.save_params(filepath, params)

    inference_fn = make_inference_fn(params)
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
        os.path.dirname(
            os.path.dirname(__file__),
        ),
        "visualization/visualization.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    main()
