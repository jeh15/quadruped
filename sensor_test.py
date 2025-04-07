from absl import app
import time


import jax
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy.typing as npt

from unitree_api_bindings import unitree_api

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


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
    control_rate = 0.02

    # Initialize Unitree-Api:
    network_name = "eno2"
    inner_control_rate = 2000
    unitree_driver = unitree_api.UnitreeDriver(
        network_name,
        inner_control_rate,
    )
    unitree_driver.initialize()

    # Sleep for 2 seconds:
    time.sleep(2.0)

    # Show State:
    imu_state = unitree_driver.get_imu_state()
    motor_state = unitree_driver.get_motor_state()
    base_rotation = np.asarray(imu_state.quaternion)
    print(f"Base Rotation: {base_rotation}")

    acceleration_data = []
    angular_velocity_data = []
    orientation_data = []
    joint_velocity_data = []

    start_time = time.time()
    while (time.time() - start_time) < 10.0:
        step_time = time.time()
        imu_state = unitree_driver.get_imu_state()
        motor_state = unitree_driver.get_motor_state()
        
        # Update Data:
        acceleration_data.append(np.asarray(imu_state.accelerometer))
        angular_velocity_data.append(np.asarray(imu_state.gyroscope))
        orientation_data.append(np.asarray(imu_state.quaternion))
        joint_velocity_data.append(np.asarray(motor_state.qd))
        

        sleep_time = control_rate - (time.time() - step_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print('Warning: Control rate exceeded.')


    # Elapsed Time:
    elapsed_time = time.time() - start_time
    print(f'Elapsed Time: {elapsed_time:.2f} seconds')

    # Convert Data to Numpy Arrays:
    acceleration_data = np.array(acceleration_data)
    angular_velocity_data = np.array(angular_velocity_data)
    orientation_data = np.array(orientation_data)
    joint_velocity_data = np.array(joint_velocity_data)

    # Calculate Mean and Standard Deviation:
    acceleration_mean = np.mean(acceleration_data, axis=0)
    angular_velocity_mean = np.mean(angular_velocity_data, axis=0)
    orientation_mean = np.mean(orientation_data, axis=0)
    joint_velocity_mean = np.mean(joint_velocity_data, axis=0)

    acceleration_std = np.std(acceleration_data, axis=0)
    angular_velocity_std = np.std(angular_velocity_data, axis=0)
    orientation_std = np.std(orientation_data, axis=0)
    joint_velocity_std = np.std(joint_velocity_data, axis=0)

    # Rotate Acceleration to World Frame to calculate Bias:
    r = R.from_quat(orientation_mean)
    acceleration_world = r.as_matrix() @ acceleration_mean
    bias = np.array([0.0, 0.0, -9.81]) - acceleration_world

    # Plot Data:
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    axs[0].plot(acceleration_data)
    for i in range(acceleration_data.shape[1]):
        axs[0].fill_between(
            np.arange(acceleration_data.shape[0]),
            acceleration_data[:, i] - acceleration_std[i],
            acceleration_data[:, i] + acceleration_std[i],
            alpha=0.2,
        )

    axs[0].set_title('Accelerometer Data')
    axs[1].plot(angular_velocity_data)
    for i in range(angular_velocity_data.shape[1]):
        axs[1].fill_between(
            np.arange(angular_velocity_data.shape[0]),
            angular_velocity_data[:, i] - angular_velocity_std[i],
            angular_velocity_data[:, i] + angular_velocity_std[i],
            alpha=0.2,
        )
    axs[1].set_title('Gyroscope Data')
    axs[2].plot(orientation_data)
    for i in range(orientation_data.shape[1]):
        axs[2].fill_between(
            np.arange(orientation_data.shape[0]),
            orientation_data[:, i] - orientation_std[i],
            orientation_data[:, i] + orientation_std[i],
            alpha=0.2,
        )
    axs[2].set_title('Orientation Data')
    axs[3].plot(joint_velocity_data)
    for i in range(joint_velocity_data.shape[1]):
        axs[3].fill_between(
            np.arange(joint_velocity_data.shape[0]),
            joint_velocity_data[:, i] - joint_velocity_std[i],
            joint_velocity_data[:, i] + joint_velocity_std[i],
            alpha=0.2,
        )
    axs[3].set_title('Joint Velocity Data')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    app.run(main)
