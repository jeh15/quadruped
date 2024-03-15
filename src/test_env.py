from typing import Optional
import os
from absl import app

import numpy as np
import jax
import jax.numpy as jnp

import brax
from brax.io import mjcf, html
from brax.envs.wrappers import training as wrapper
from brax.envs.base import Env
from tqdm import tqdm

import time

import quadruped
import custom_wrapper


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
    env = quadruped.Quadruped(**kwargs)

    if episode_length is not None:
        env = wrapper.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = wrapper.VmapWrapper(env, batch_size)
    if auto_reset:
        env = custom_wrapper.AutoResetWrapper(env)

    return env


def main(argv=None):
    n_frames = 10
    env = create_environment(
        episode_length=None,
        auto_reset=False,
        batch_size=False,
        backend='generalized',
        n_frames=n_frames,
    )

    # Load mjcf model:
    pipeline_model = mjcf.load(env.filepath)
    motor_mask = pipeline_model.actuator.qd_id

    initial_q = env.initial_q
    base_ctrl = jnp.zeros_like(motor_mask)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Compile Step Function for timing:
    initial_key = jax.random.key(42)
    state = reset_fn(initial_key)
    _ = step_fn(state, base_ctrl)

    # Use utility function:
    jit_foot_position = jax.jit(quadruped.calculate_foot_position)
    foot_x = jit_foot_position(
        env.sys,
        state.pipeline_state.q,
        state.pipeline_state.qd,
    )

    # foot_x = quadruped.calculate_foot_position(
    #     env.sys,
    #     state.pipeline_state.q,
    #     state.pipeline_state.qd,
    # )

    state_history = [state.pipeline_state]
    simulation_steps = 1000
    desired_joint_position = np.array([0.0, 0.0])
    kp = 10.0

    start_time = time.time()
    for i in tqdm(range(simulation_steps)):
        front_left_leg_state = state.pipeline_state.q[env.front_left_id]
        front_right_leg_state = state.pipeline_state.q[env.front_right_id]
        back_left_leg_state = state.pipeline_state.q[env.back_left_id]
        back_right_leg_state = state.pipeline_state.q[env.back_right_id]
        joint_positions = [
            front_left_leg_state, front_right_leg_state,
            back_left_leg_state, back_right_leg_state,
        ]
        control = []
        for joint_position in zip(joint_positions):
            control.append(kp * (desired_joint_position - joint_position[0]))
        ctrl_input = np.asarray(control).flatten()
        state = step_fn(state, ctrl_input)
        state_history.append(state.pipeline_state)

    end_time = time.time() - start_time
    simulation_time = simulation_steps * env.step_dt
    print(f"Run Time: {end_time}")
    print(f"Simulation Time: {simulation_time}")
    print(f"Time Ratio: {simulation_time / end_time}")

    html_string = html.render(
        env.sys,
        state_history,
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


if __name__ == "__main__":
    app.run(main)
