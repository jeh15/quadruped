from typing import Tuple

import os
from absl import app
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from brax.io import mjcf
from brax.mjx import pipeline

jax.config.update('jax_enable_x64', True)


def main(argv=None):
    filename = 'models/barkour/fixed_scene.xml'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    sys = mjcf.load(filepath)

    sys = sys.replace(
        actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
        actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
    )

    # Change Parameters to regress:
    sys = sys.replace(
        dof_damping=sys.dof_damping.at[:].set(0.5239),
    )

    # Random Initial State Generator:
    def random_initial_state(
        home_position: jax.Array,
        key: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        initial_q = home_position + jax.random.uniform(
            key, (sys.q_size(),), minval=-0.1, maxval=0.1,
        )
        initial_qd = jnp.zeros((sys.qd_size(),))
        return initial_q, initial_qd

    vmap_initial_state = jax.vmap(
        random_initial_state, in_axes=(None, 0), out_axes=(0, 0),
    )
    initial_state_fn = jax.jit(vmap_initial_state)

    key = jax.random.key(42)
    key, initial_state_key, ctrl_key = jax.random.split(key, 3)
    num_trials = 100
    initial_state_keys = jax.random.split(initial_state_key, num_trials)
    home_position = jnp.array([
        0.0, 0.5, 1.0,
    ])
    initial_q, initial_qd = initial_state_fn(home_position, initial_state_keys)

    vmap_init_fn = jax.vmap(pipeline.init, in_axes=(None, 0, 0))
    init_fn = jax.jit(vmap_init_fn)
    states = init_fn(sys, initial_q, initial_qd)

    vmap_step_fn = jax.vmap(pipeline.step, in_axes=(None, 0, 0))
    step_fn = jax.jit(vmap_step_fn)

    # Random Control Input:
    num_time_steps = 1000
    ctrl_range = sys.actuator_ctrlrange
    ctrl_midpoint = np.mean(ctrl_range, axis=1)

    def random_control(
        ctrl_midpoint: jnp.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        amplitude = jax.random.uniform(
            key, (sys.act_size(),), minval=-0.5, maxval=0.5,
        )
        frequency = jax.random.randint(
            key, (sys.act_size(),), minval=50, maxval=150,
        )
        control_history = []
        for i in range(num_time_steps):
            ctrl = ctrl_midpoint + amplitude * jnp.sin(
                i / frequency,
            )
            control_history.append(ctrl)
        return jnp.asarray(control_history)

    vmap_random_control = jax.vmap(
        random_control, in_axes=(None, 0), out_axes=0,
    )
    random_control_fn = jax.jit(vmap_random_control)
    ctrl_keys = jax.random.split(ctrl_key, num_trials)
    ctrl_input = random_control_fn(ctrl_midpoint, ctrl_keys)

    state_history = []
    for i in range(num_time_steps):
        states = step_fn(sys, states, ctrl_input[:, i])
        state_history.append(states)

    # Save states:
    with open('train_params.pkl', 'wb') as f:
        pickle.dump([state_history, ctrl_input], f)


if __name__ == '__main__':
    app.run(main)
