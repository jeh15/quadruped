import os
from absl import app

import numpy as np
import jax
import jax.numpy as jnp
from brax.io import html

import time

import quadruped

jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=4)


def main(argv=None):
    env = quadruped.Quadruped(backend='mjx')
    episode_run_time = 20.0  # Seconds
    batch_run_time = 0.5  # Seconds
    episode_length = int(episode_run_time / env.dt)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Compile Step Function for timing:
    initial_key = jax.random.key(42)
    state = reset_fn(initial_key)

    state_history = [state.pipeline_state]
    kp = np.array([2.0, 6.0, 2.0, 6.0, 2.0, 6.0, 2.0, 6.0])
    kd = 0.1

    start_time = time.time()
    for i in range(episode_length):
        leg_q = state.pipeline_state.q[7:]
        leg_qd = state.pipeline_state.qd[6:]
        ctrl_input = kp * (np.zeros_like(leg_q) - leg_q) - kd * leg_qd
        if i % 100 == 0:
            print(f'Control Input: {ctrl_input}')
        state = step_fn(state, ctrl_input)
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
