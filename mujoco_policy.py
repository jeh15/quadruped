from absl import app, flags
import functools
import time

import pygame

import jax
import numpy as np
import numpy.typing as npt

import mujoco
import mujoco.viewer

import matplotlib.pyplot as plt

from src.envs import unitree_go2_mujoco_playground as unitree_go2
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
    ctrl_lb: npt.ArrayLike,
    ctrl_ub: npt.ArrayLike,
    action_scale: float,
) -> np.ndarray:
    motor_targets = default_control + action * action_scale
    motor_targets = np.clip(motor_targets, ctrl_lb, ctrl_ub)
    return motor_targets


def rotate(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    if len(vec.shape) != 1:
        raise ValueError('vec must have no batch dimensions.')
    s, u = quat[0], quat[1:]
    r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
    r = r + 2 * s * np.cross(u, vec)
    return r


def quat_inv(q: np.ndarray) -> np.ndarray:
    return q * np.array([1, -1, -1, -1])


def main(argv=None):
    # Load from Env:
    env = unitree_go2.UnitreeGo2Env(filename='unitree_go2/scene_mjx.xml')
    model = env.sys.mj_model

    data = mujoco.MjData(model)  # type: ignore
    mujoco.mj_resetData(model, data)  # type: ignore
    control_rate = 0.02
    num_steps = int(control_rate / model.opt.timestep)

    # Load Policy:
    make_policy, params = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
        restore_iteration=FLAGS.checkpoint_iteration,
    )
    inference_function = make_policy(params)
    inference_fn = jax.jit(inference_function)

    # Controller:
    controller_fn = functools.partial(
        controller,
        default_control=env.default_ctrl,
        ctrl_lb=env.ctrl_lb,
        ctrl_ub=env.ctrl_ub,
        action_scale=env._action_scale,
    )

    # Test:
    data.qpos = model.key_qpos.flatten()
    command = np.array([0.0, 0.0, 0.0])
    action = model.key_ctrl.flatten()
    observation = np.zeros(env.history_length * env.num_observations)

    # Setup Joystick:
    joysticks = {}

    key = jax.random.key(0)
    termination_flag = False
    command_history = []

    global_steps = 0
    previous_ctrl = action
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5

        while viewer.is_running() and not termination_flag:
            if global_steps >= 1000:
                termination_flag = True

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
                # if joystick.get_button(11) == 1:
                #     termination_flag = True
                # In Logitech Controller:
                if joystick.get_button(7) == 1:
                    termination_flag = True

                forward_command = -1 * joystick.get_axis(1)
                lateral_command = -1 * joystick.get_axis(0)
                # If Switch Controller:
                # rotation_command = -1 * joystick.get_axis(2)
                # If Logitech Controller:
                rotation_command = -1 * joystick.get_axis(3)


            # Filter and Clip Command:
            command = np.array([
                forward_command, lateral_command, rotation_command,
            ])
            command = np.where(np.abs(command) < 0.1, 0.0, command)
            command = np.clip(command, -0.75, 0.75)

            step_time = time.time()
            action_rng, key = jax.random.split(key)
            observation = env.np_observation(
                mj_data=data,
                command=command,
                previous_action=action,
                observation_history=observation,
            )
            action, _ = inference_fn(observation, action_rng)
            ctrl = controller_fn(action)

            # Compare gyro and original way:
            gyro = env.get_gyro(data)
            base_w = data.qpos[3:7]
            base_dw = data.qvel[3:6]
            inverse_trunk_rotation = quat_inv(base_w)
            body_frame_angular_vel = rotate(
                base_dw, inverse_trunk_rotation,
            )
            print(f"Gyro: {gyro}")
            print(f"Body Frame Angular Vel: {body_frame_angular_vel}")

            # Smooth Control:
            # alpha = 1.0
            # ctrl = alpha * ctrl + (1 - alpha) * previous_ctrl
            # previous_ctrl = ctrl

            data.ctrl = ctrl

            for _ in range(num_steps):
                mujoco.mj_step(model, data)  # type: ignore

            viewer.sync()

            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            global_steps += 1


if __name__ == '__main__':
    app.run(main)
