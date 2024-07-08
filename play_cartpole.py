from absl import app, flags
import time

import jax
import numpy as np

import mujoco
import mujoco.viewer

from src.envs import inverted_pendulum
# from src.algorithms.apg.load_utilities import load_policy
from src.algorithms.gb_ppo.load_utilities import load_policy

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)


def main(argv=None):
    # Load from Env:
    env = inverted_pendulum.InvertedPendulum()
    model = env.sys.mj_model

    data = mujoco.MjData(model)  # type: ignore
    mujoco.mj_resetData(model, data)  # type: ignore
    control_rate = env.step_dt
    num_steps = int(control_rate / model.opt.timestep)

    # Load Policy:
    make_policy, params = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
    )
    inference_function = make_policy(params)
    inference_fn = jax.jit(inference_function)

    # Initialize Simulation:
    key = jax.random.key(0)
    initial_state = jax.jit(env.reset)(key)
    data.qpos = initial_state.pipeline_state.q.flatten()
    data.qvel = initial_state.pipeline_state.qd.flatten()
    observation = initial_state.obs

    # Setup Joystick:
    joysticks = {}

    termination_flag = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not termination_flag:

            for joystick in joysticks.values():
                if joystick.get_button(11) == 1:
                    termination_flag = True

            step_time = time.time()
            action_rng, key = jax.random.split(key)
            observation = env.mujoco_get_observation(
                mj_data=data,
            )
            action, _ = inference_fn(observation, action_rng)
            data.ctrl = action

            for _ in range(num_steps):
                mujoco.mj_step(model, data)  # type: ignore

            viewer.sync()

            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == '__main__':
    app.run(main)
