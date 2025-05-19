import os
from absl import app, flags, logging
import functools
import time

import pygame

import jax
import numpy as np
import numpy.typing as npt

import mujoco
import mujoco.viewer

from src.envs import unitree_go2_height_control_v3 as unitree_go2
from src.algorithms.ppo.load_utilities import load_policy


jax.config.update("jax_enable_x64", True)
pygame.init()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)


def controller(
    action: npt.ArrayLike,
    default_control: npt.ArrayLike,
    action_scale: float,
) -> np.ndarray:
    motor_targets = default_control + action * action_scale
    return motor_targets


def main(argv=None):
    # Set up Logger:
    logging.use_absl_handler()
    log_directory = os.path.join(
        os.path.dirname(__file__),
        'logs',
    )
    logging.get_absl_handler().use_absl_log_file(program_name='simulation_test', log_dir=log_directory) 
    logging.set_verbosity(logging.INFO)

    # Load from Env:
    env = unitree_go2.UnitreeGo2Env()
    model_mjx = env.sys.mj_model

    # High Fidelity Model:
    model_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx_collision.xml',
    )

    model = mujoco.MjModel.from_xml_path(
        model_path,
    )
    model.opt.timestep = 0.004

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    control_rate = 0.02
    num_steps = int(control_rate / model.opt.timestep)

    # Load Policy:
    make_policy, params, _ = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
        restore_iteration=FLAGS.checkpoint_iteration,
    )
    inference_function = make_policy(params, deterministic=True)
    inference_fn = jax.jit(inference_function)

    # Controller:
    controller_fn = functools.partial(
        controller,
        default_control=env.default_ctrl,
        action_scale=env.action_scale,
    )

    # Initialize:
    action = np.zeros(12)
    observation = {
        'state': np.zeros((env.num_observations,)),
        'privileged_state': np.zeros((env.num_privileged_observations,)),
    }

    # Setup Joystick:
    joysticks = {}

    key = jax.random.key(0)
    termination_flag = False


    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5

        while viewer.is_running() and not termination_flag:
            for event in pygame.event.get():
                if event.type == pygame.JOYDEVICEADDED:
                    joy = pygame.joystick.Joystick(event.device_index)
                    joysticks[joy.get_instance_id()] = joy
                    print(f"Joystick {joy.get_instance_id()} connencted")

                if event.type == pygame.JOYDEVICEREMOVED:
                    del joysticks[event.instance_id]
                    print(f"Joystick {event.instance_id} disconnected")

            for joystick in joysticks.values():
                # If Switch Controller:
                if joystick.get_button(11) == 1:
                    termination_flag = True
                
                # Feet Tracking and Height Control Policy:
                command = -1 * joystick.get_axis(1)

            # Height Control Policy:
            command = np.array([
                command
            ])
            lb, ub = 0.0, 0.4
            command = np.where(np.abs(command) < 0.1, 0.0, command)
            command = np.clip(command, -1.0, 1.0)
            command = lb + (command - -1) * (ub - lb) / (1 - -1)
            command = np.clip(command, lb, ub)

            step_time = time.time()
            action_rng, key = jax.random.split(key)

            # Get Observation:
            observation = env.np_observation(
                mj_data=data,
                observation_history=observation['state'],
                command=command,
                previous_action=action,
                add_noise=False,
            )
            action, _ = jax.block_until_ready(
                inference_fn(observation, action_rng),
            )
            ctrl = controller_fn(action)

            # data.ctrl = ctrl
            data.ctrl = ctrl

            for _ in range(num_steps):
                mujoco.mj_step(model, data)  # type: ignore

            viewer.sync()

            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                time.sleep(sleep_time)



if __name__ == '__main__':
    app.run(main)
