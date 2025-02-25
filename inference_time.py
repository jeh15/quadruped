from absl import app, flags
import os

import jax
import jax.numpy as jnp

from brax.io import html

from src.envs import unitree_go2
from src.algorithms.ppo.load_utilities import load_policy

import numpy as np
import time

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)


def main(argv=None):
    # Load from Env:
    env = unitree_go2.UnitreeGo2Env()
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Load Policy:
    make_policy, params = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        restore_iteration=FLAGS.checkpoint_iteration,
        environment=env,
    )
    inference_function = make_policy(params)
    inference_fn = jax.jit(inference_function)

    # Initialize Simulation:
    key = jax.random.key(0)

    velocity = 0.0
    key, subkey = jax.random.split(key)
    state = reset_fn(subkey)
    state.info['command'] = jnp.array([velocity, 0.0, 0.0])

    num_steps = 1000
    states = []

    ramp_time = 200
    state_obs = []
    keys = []
    times = []
    for i in range(num_steps):
        # Stop Command Sampling:
        if i <= ramp_time:
            velocity = (i * env.step_dt) / (ramp_time * env.step_dt)
        state.info['command'] = jnp.array([velocity, 0.0, 0.0])
        key, subkey = jax.random.split(key)

        # # Inference: Test np.asarray -- Avg 2337 Hz
        # start_time = time.time()
        # state_obs_gpu = state.obs
        # state_obs_gpu.block_until_ready()
        # state_obs_cpu = np.asarray(state_obs_gpu)
        # action, _ = inference_fn(state_obs_cpu, subkey)
        # action.block_until_ready()
        # cpu_array = np.asarray(action)
        # elapsed_time = time.time() - start_time

        # Inference: Test CPU Put Device -- Avg 3935 Hz
        # start_time = time.time()
        # state_obs_gpu = state.obs
        # state_obs_gpu.block_until_ready()
        # state_obs_cpu = jax.device_put(state_obs_gpu, jax.devices("cpu")[0])
        # state_obs_cpu.block_until_ready()
        # action, _ = inference_fn(state_obs_cpu, subkey)
        # action.block_until_ready()
        # cpu_array = jax.device_put(action, jax.devices("cpu")[0])
        # cpu_array.block_until_ready()
        # elapsed_time = time.time() - start_time

        # Inference: Test CPU Put Device and np.asarray -- Avg 2044 Hz
        # start_time = time.time()
        # state_obs_gpu = state.obs
        # state_obs_gpu.block_until_ready()
        # state_obs_cpu = jax.device_put(state_obs_gpu, jax.devices("cpu")[0])
        # state_obs_cpu.block_until_ready()
        # state_obs_cpu = np.asarray(state_obs_cpu)
        # action, _ = inference_fn(state_obs_cpu, subkey)
        # action.block_until_ready()
        # cpu_array = jax.device_put(action, jax.devices("cpu")[0])
        # cpu_array.block_until_ready()
        # cpu_array = np.asarray(cpu_array)
        # elapsed_time = time.time() - start_time

        # Inference: Test CPU Put Device write to list -- Avg 3653 Hz
        start_time = time.time()
        state_obs_gpu = state.obs
        state_obs_gpu.block_until_ready()
        state_obs_cpu = jax.device_put(state_obs_gpu, jax.devices("cpu")[0])
        state_obs_cpu.block_until_ready()
        state_obs_list = state_obs_cpu.tolist()
        action, _ = inference_fn(state_obs_cpu, subkey)
        action.block_until_ready()
        cpu_array = jax.device_put(action, jax.devices("cpu")[0])
        cpu_array.block_until_ready()
        cpu_list = cpu_array.tolist
        elapsed_time = time.time() - start_time

        state = step_fn(state, action)
        states.append(state.pipeline_state)

        # Store state obs, key, and elapsed time
        state_obs.append(state.obs)
        keys.append(subkey)
        times.append(elapsed_time)

    avg_time = np.mean(times)
    print(f"Average Inference Time: {avg_time} seconds")
    print(f"Average Inference Rate: {1.0 / avg_time} Hz")
    print(f"Total Inference Time: {np.sum(times)} seconds")


if __name__ == '__main__':
    app.run(main)
