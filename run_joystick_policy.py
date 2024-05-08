from absl import app, flags
import time

import pygame

import jax
import numpy as np

import mujoco
import mujoco.viewer

from src.envs import barkour
from src.load_utilities import load_policy

jax.config.update("jax_enable_x64", True)
pygame.init()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)


def controller(env, policy_action):
    lb = np.array([-0.7, -1.0, 0.05] * 4)
    ub = np.array([0.52, 2.1, 2.1] * 4)
    motor_targets = env._default_pose + policy_action * env._action_scale
    motor_targets = np.clip(motor_targets, lb, ub)
    return motor_targets


def main(argv=None):
    # Load from Env:
    env = barkour.BarkourEnv()
    model = env.sys.mj_model

    # Adjust Model:
    model.opt.timestep = 0.004
    model.dof_damping[6:] = 0.5239
    model.actuator_gainprm[:, 0] = 35.0
    model.actuator_biasprm[:, 1] = -35.0

    data = mujoco.MjData(model)  # type: ignore
    mujoco.mj_resetData(model, data)  # type: ignore
    control_rate = 0.02
    num_steps = int(control_rate / model.opt.timestep)

    # Load Policy:
    make_policy, params = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
    )
    inference_function = make_policy(params)
    inference_fn = jax.jit(inference_function)

    # Test:
    data.qpos = model.key_qpos.flatten()
    command = np.array([0.0, 0.0, 0.0])
    action = model.key_ctrl.flatten()
    observation = np.zeros(15 * 31)

    # Setup Joystick:
    joysticks = {}

    key = jax.random.key(0)
    termination_flag = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not termination_flag:

            # Move all of this to an event handler:
            for event in pygame.event.get():
                if event.type == pygame.JOYDEVICEADDED:
                    joy = pygame.joystick.Joystick(event.device_index)
                    joysticks[joy.get_instance_id()] = joy
                    print(f"Joystick {joy.get_instance_id()} connencted")

                if event.type == pygame.JOYDEVICEREMOVED:
                    del joysticks[event.instance_id]
                    print(f"Joystick {event.instance_id} disconnected")

            for joystick in joysticks.values():
                if joystick.get_button(11) == 1:
                    termination_flag = True

                forward_command = -1 * joystick.get_axis(1)
                lateral_command = -1 * joystick.get_axis(0)
                rotation_command = -1 * joystick.get_axis(2)

                # Filter and Clip Command:
                command = np.array([
                    forward_command, lateral_command, rotation_command,
                ])
                command = np.where(np.abs(command) < 0.1, 0.0, command)
                command = np.clip(command, -0.75, 0.75)

            step_time = time.time()
            action_rng, key = jax.random.split(key)
            observation = env.get_observation(
                mj_data=data,
                command=command,
                previous_action=action,
                observation_history=observation,
            )
            action, _ = inference_fn(observation, action_rng)
            ctrl = controller(env, action)
            data.ctrl = ctrl

            for _ in range(num_steps):
                mujoco.mj_step(model, data)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

            viewer.sync()

            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == '__main__':
    app.run(main)
