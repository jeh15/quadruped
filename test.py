from absl import app
import os

import jax
import jax.numpy as jnp

from brax.mjx import pipeline
from brax.io import mjcf, html


def main(argv=None):
    filepath = os.path.join(
        os.path.dirname(__file__),
        "models/unitree_go1/scene.xml",
    )

    sys = mjcf.load(filepath)

    init_fn = jax.jit(pipeline.init)
    step_fn = jax.jit(pipeline.step)

    q_init = jnp.array(sys.mj_model.keyframe('home').qpos)
    qd_init = jnp.zeros(sys.qd_size())
    default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)

    pipeline_state = init_fn(sys, q_init, qd_init)

    simulation_steps = 100
    state_history = []
    for i in range(simulation_steps):
        print(f"Step: {i}")
        pipeline_state = step_fn(sys, pipeline_state, default_ctrl)
        state_history.append(pipeline_state)

    html_string = html.render(
        sys=sys,
        states=state_history,
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
    app.run(main)
