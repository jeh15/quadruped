import os
from absl import app

import numpy as np
import jax
import jax.numpy as jnp
from brax.io import html

import time

import barkour
import control_utilities

jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=4)


def main(argv=None):
    env = barkour.BarkourEnv()
    episode_run_time = 10.0  # Seconds
    batch_run_time = 0.5  # Seconds
    episode_length = int(episode_run_time / env.dt)

    control_range = env.sys.actuator_ctrlrange

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    # reset_fn = env.reset
    # step_fn = env.step

    # Compile Step Function for timing:
    initial_key = jax.random.key(42)
    state = reset_fn(initial_key)

    state_history = [state.pipeline_state]

    start_time = time.time()
    for i in range(episode_length):
        ctrl_input = env._default_ctrl
        if i % 100 == 0:
            print(f'Control Input: {ctrl_input}')
        state = step_fn(state, jnp.squeeze(ctrl_input))
        state_history.append(state.pipeline_state)

    end_time = time.time() - start_time
    print(f"Run Time: {end_time}")
    print(f"Simulation Time: {episode_run_time}")
    print(f"Time Ratio: {episode_run_time / end_time}")

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


if __name__ == "__main__":
    app.run(main)
