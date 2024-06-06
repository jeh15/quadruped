import os
from absl import app, flags
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from brax.io import mjcf, html
from brax.mjx import pipeline


FLAGS = flags.FLAGS
flags.DEFINE_bool(
    'change_params', False, 'Change parameters flag.', short_name='c',
)


def main(argv=None):
    filename = 'models/barkour/fixed_scene.xml'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    sys = mjcf.load(filepath)
    ctrl_range = sys.actuator_ctrlrange
    ctrl_midpoint = np.mean(ctrl_range, axis=1)

    sys = sys.replace(
        actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
        actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
    )

    # Change Parameters to regress:
    if FLAGS.change_params:
        sys = sys.replace(
            dof_damping=sys.dof_damping.at[:].set(0.5239),
        )

    # Create a pipeline
    initial_q = jnp.array([
        0.0, 0.5, 1.0,
    ])
    initial_qd = jnp.zeros_like(initial_q)
    state = jax.jit(pipeline.init)(sys, initial_q, initial_qd)

    states = []
    control = []
    for i in range(1000):
        freq = 100
        position_ctrl = np.array([
            ctrl_midpoint[0],
            ctrl_midpoint[1] + 0.75 * np.sin(i / freq),
            ctrl_midpoint[2],
        ])
        state = jax.jit(pipeline.step)(sys, state, position_ctrl)
        states.append(state)
        control.append(position_ctrl)

    # Save states:
    with open(f'params_{FLAGS.change_params}.pkl', 'wb') as f:
        pickle.dump([states, control], f)

    html_string = html.render(
        sys,
        states,
        height="100vh",
        colab=False,
    )

    html_path = os.path.join(
        os.path.dirname(__file__),
        'visualization/visualization.html',
    )

    with open(html_path, 'w') as f:
        f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)
