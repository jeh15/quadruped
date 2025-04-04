from absl import app, flags
import functools
import time

import pygame

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from unitree_api_bindings import unitree_api

import mujoco
import mujoco.viewer

from src.envs import unitree_go2_mujoco_playground as unitree_go2
from src.algorithms.ppo.load_utilities import load_policy

jax.config.update("jax_enable_x64", True)
pygame.init()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
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


def get_observation(
    observation: npt.ArrayLike,
    imu_state: unitree_api.IMUState,
    motor_state: unitree_api.MotorState,
    command: npt.ArrayLike,
    previous_action: npt.ArrayLike,
    default_position: npt.ArrayLike,
) -> npt.ArrayLike:
    base_rotation = np.asarray(imu_state.quaternion)
    base_angular_velocity = np.asarray(imu_state.gyroscope)
    joint_positions = np.asarray(motor_state.q)

    # Calculate Body frame Yaw Rate and Projected Gravity:
    inverse_base_rotation = quat_inv(base_rotation)
    projected_gravity = rotate(
        np.array([0.0, 0.0, -1.0]),
        inverse_base_rotation,
    )

    new_observation = np.concatenate([
        base_angular_velocity,
        projected_gravity,
        joint_positions - default_position,
        previous_action,
        command,
    ])

    # Stack Observation:
    observation = np.roll(observation, new_observation.size)
    observation[:new_observation.size] = new_observation

    return observation


def main(argv=None):
    # Load from Env:
    env = unitree_go2.UnitreeGo2Env(filename='unitree_go2/scene_mjx.xml')
    model = env.sys.mj_model

    data = mujoco.MjData(model)
    control_rate = 0.02
    num_physics_steps = int(control_rate / model.opt.timestep)

    # Controller:
    controller_fn = functools.partial(
        controller,
        default_control=env.default_ctrl,
        ctrl_lb=env.ctrl_lb,
        ctrl_ub=env.ctrl_ub,
        action_scale=env._action_scale,
    )

    # Initialize Unitree-Api:
    network_name = "eno2"
    inner_control_rate = 2000
    unitree_driver = unitree_api.UnitreeDriver(
        network_name,
        inner_control_rate,
    )
    unitree_driver.initialize()

    # Show State:
    imu_state = unitree_driver.get_imu_state()
    motor_state = unitree_driver.get_motor_state()
    base_rotation = np.asarray(imu_state.quaternion)
    print(f"Base Rotation: {base_rotation}")

    # Update Data:
    data.qpos = model.key_qpos.flatten()
    mujoco.mj_forward(model, data) 

    key = jax.random.key(0)
    termination_flag = False

    global_steps = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5

        while viewer.is_running() and not termination_flag:
            step_time = time.time()
            action_rng, key = jax.random.split(key)
            imu_state = unitree_driver.get_imu_state()
            motor_state = unitree_driver.get_motor_state()
            
            # Update Data:
            orientation = np.asarray(imu_state.quaternion)
            base_angular_velocity = np.asarray(imu_state.gyroscope)
            joint_position = np.asarray(motor_state.q)
            joint_velocity = np.asarray(motor_state.qd)

            data.qpos = np.array([
                0.0, 0.0, 0.35,
                *orientation,
                *joint_position,
            ])
            data.qvel = np.array([
                0.0, 0.0, 0.0,
                *base_angular_velocity,
                *joint_velocity,
            ])

            print(f'Right Leg Joint Position: {joint_position[0:3]}')

            mujoco.mj_forward(model, data)

            # Compare Gravity Calculation:
            inverse_base_rotation = quat_inv(orientation)
            projected_gravity = rotate(
                np.array([0.0, 0.0, -1.0]),
                inverse_base_rotation,
            )
            # print(f'Projected Gravity: {projected_gravity}')

            for _ in range(num_physics_steps):
                mujoco.mj_step(model, data)  # type: ignore

            viewer.sync()

            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print('Warning: Control rate exceeded.')
            
            global_steps += 1


if __name__ == '__main__':
    app.run(main)
