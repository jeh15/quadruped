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
    filename = 'models/unitree_go2/scene_mjx_fixed.xml'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    sys = mjcf.load(filepath)

    # Random Initial State Generator:
    def random_initial_state(
        home_position: jax.Array,
        key: jax.Array,
    ) -> jnp.ndarray:
        home_position = jnp.reshape(home_position, (4, -1))
        abduction_range = jnp.array([
            -0.1, 0.1,
        ])
        hip_range = jnp.array([
            -0.3, 0.3,
        ])
        knee_range = jnp.array([
            -0.2, 0.2,
        ])
        bounds = jnp.vstack(
            (abduction_range, hip_range, knee_range),
        )

        qpos = home_position + jax.random.uniform(
            key, shape=home_position.shape, minval=bounds[:, 0], maxval=bounds[:, 1],
        )
        return qpos
    
    def random_control(
        initial_position: jax.Array,
        key: jax.Array,
        num_time_steps: int = 500,
    ) -> jnp.ndarray:
        abduction_amplitude = jax.random.uniform(
            key, shape=(4,), minval=-0.1, maxval=0.1,
        )
        hip_amplitude = jax.random.uniform(
            key, shape=(4,), minval=-0.3, maxval=0.3,
        )
        knee_amplitude = jax.random.uniform(
            key, shape=(4,), minval=-0.2, maxval=0.2,
        )

        abduction_frequency = jax.random.randint(
            key, shape=(4,), minval=50, maxval=150,
        )
        hip_frequency = jax.random.randint(
            key, shape=(4,), minval=50, maxval=150,
        )
        knee_frequency = jax.random.randint(
            key, shape=(4,), minval=50, maxval=150,
        )

        x = jnp.arange(num_time_steps)
        abduction_trajectory = initial_position + abduction_amplitude * jnp.sin(
            abduction_frequency * x,
        )
        hip_trajectory = initial_position + hip_amplitude * jnp.sin(
            hip_frequency * x,
        )
        knee_trajectory = initial_position + knee_amplitude * jnp.sin(
            knee_frequency * x,
        )
        control_trajectory = jnp.vstack(
            (abduction_trajectory, hip_trajectory, knee_trajectory),
        )
        return control_trajectory
        

    vmap_initial_state = jax.vmap(
        random_initial_state, in_axes=(None, 0), out_axes=0,
    )
    initial_state_fn = jax.jit(vmap_initial_state)

    vmap_control_trajectory = jax.vmap(
        random_control, in_axes=(0, 0), out_axes=0,
    )
    control_trajectory_fn = jax.jit(vmap_control_trajectory)

    key = jax.random.key(42)
    key, state_key, ctrl_key = jax.random.split(key, 3)
    num_trials = 100
    state_keys = jax.random.split(state_key, num_trials)
    control_keys = jax.random.split(ctrl_key, num_trials)

    home_position = jnp.array(sys.mj_model.keyframe('home').qpos[:])
    qpos = initial_state_fn(home_position, state_keys)
    control_trajectory = control_trajectory_fn(
        qpos, control_keys,
    )
    pass



if __name__ == '__main__':
    app.run(main)