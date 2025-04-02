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

import matplotlib.pyplot as plt

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

    # Load Policy:
    make_policy, params = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
    )
    inference_function = make_policy(params, deterministic=True)
    inference_fn = jax.jit(inference_function)

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

    # Default Control:
    motor_commands = unitree_api.MotorCommand()
    motor_commands.q_setpoint = [0.0, 0.9, -1.8] * 4
    motor_commands.qd_setpoint = [0.0, 0.0, 0.0] * 4
    motor_commands.torque_feedforward = [0.0, 0.0, 0.0] * 4
    motor_commands.stiffness = [0.0, 0.0, 0.0] * 4
    motor_commands.damping = [0.0, 0.0, 0.0] * 4
    unitree_driver.update_command(motor_commands)

    # Initialize Thread:
    unitree_driver.initialize_thread()

    # Ramp to Default Control:
    ramp_time = 5.0
    num_steps = 1000
    stifness_ramp = np.linspace(0.0, 60.0, num_steps)
    damping_ramp = np.linspace(0.0, 5.0, num_steps)
    for stiffness, damping in zip(stifness_ramp, damping_ramp):
        motor_commands.stiffness = [stiffness, stiffness, stiffness] * 4
        motor_commands.damping = [damping, damping, damping] * 4
        unitree_driver.update_command(motor_commands)
        time.sleep(ramp_time / num_steps)

    # Show State:
    imu_state = unitree_driver.get_imu_state()
    motor_state = unitree_driver.get_motor_state()
    base_rotation = np.asarray(imu_state.quaternion)
    print(f"Base Rotation: {base_rotation}")

    # Wait for Keyboard Input:
    print('Press any key to start the simulation...')
    input()

    # Initialize Observation History:
    observation = np.zeros(env.history_length * env.num_observations)
    action = np.asarray(env.default_ctrl)
    default_position = np.asarray(env.default_ctrl)
    command = np.array([0.0, 0.0, 0.0])
    observation_fn = functools.partial(
        get_observation,
        default_position=default_position,
    )
    for i in range(env.history_length):
        imu_state = unitree_driver.get_imu_state()
        motor_state = unitree_driver.get_motor_state()
        observation = observation_fn(
            observation=observation,
            imu_state=imu_state,
            motor_state=motor_state,
            command=command,
            previous_action=action,
        )

    # Update Data:
    data.qpos = model.key_qpos.flatten()
    mujoco.mj_forward(model, data) 

    key = jax.random.key(0)
    termination_flag = False

    global_steps = 0
    previous_ctrl = default_position
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5

        while viewer.is_running() and not termination_flag:
            step_time = time.time()
            action_rng, key = jax.random.split(key)
            imu_state = unitree_driver.get_imu_state()
            motor_state = unitree_driver.get_motor_state()
            observation = observation_fn(
                observation=observation,
                imu_state=imu_state,
                motor_state=motor_state,
                command=command,
                previous_action=action,
            )
            action, _ = inference_fn(observation, action_rng)
            action.block_until_ready()
            action = jax.device_put(action, jax.devices('cpu')[0])
            action = np.asarray(action)
            ctrl = controller_fn(action)

            # To Control the Robot:
            motor_commands.q_setpoint = ctrl.tolist()
            motor_commands.stiffness = [35.0, 35.0, 35.0] * 4
            motor_commands.damping = [0.5, 0.5, 0.5] * 4
            unitree_driver.update_command(motor_commands)

            # To Control Simulation:
            data.ctrl = ctrl

            for _ in range(num_physics_steps):
                mujoco.mj_step(model, data)  # type: ignore

            viewer.sync()

            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print('Warning: Control rate exceeded.')
            
            global_steps += 1

    # Stop Thread:
    unitree_driver.stop_thread()


if __name__ == '__main__':
    app.run(main)
