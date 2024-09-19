from absl import app
import os

import time

import jax
import jax.numpy as jnp

from brax.mjx import pipeline
from brax.io import mjcf, html

jax.config.update('jax_enable_x64', True)


def main(argv=None):
    # Load Model:
    filepath = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx.xml',
    )

    sys = mjcf.load(filepath)
    sys = sys.tree_replace({'opt.timestep': 0.004})

    q = jnp.array(sys.mj_model.keyframe('home').qpos)
    action = jnp.array(sys.mj_model.keyframe('home').ctrl)
    qd = jnp.zeros(sys.qd_size())

    state = jax.jit(pipeline.init)(sys, q, qd)
    step_fn = jax.jit(pipeline.step)

    states = [state]

    num_steps = 1000
    start_time = time.time()
    for i in range(num_steps):
        state = step_fn(sys, state, action)
        states.append(state)

    print(f"Time: {time.time() - start_time}")

    # Save to HTML:
    html_string = html.render(
        sys=sys,
        states=states,
        height="100vh",
        colab=False,
    )

    html_path = os.path.join(
        os.path.dirname(__file__),
        'visualization/visualization.html',
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)
