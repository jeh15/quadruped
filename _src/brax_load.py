import os
import functools

import jax
import jax.numpy as jnp

from brax.io import model, html
from brax.training.agents.ppo import networks as ppo_networks

import src.envs.barkour as barkour

jax.config.update("jax_enable_x64", True)


def main(argv=None):
    # Make PPO NetworK:
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128),
    )

    # Brax Env:
    env = barkour.BarkourEnv()
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state = reset_fn(jax.random.PRNGKey(0))

    ppo_network = network_factory(
        state.obs.shape[-1],
        env.action_size,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    filename = 'mjx_brax_quadruped_policy'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )

    params = model.load_params(filepath)
    inference_fn = make_policy(params)
    jit_inference_fn = jax.jit(inference_fn)

    # Run Inference:
    x_vel = 1.0 
    y_vel = 0.0 
    ang_vel = -0.5

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
