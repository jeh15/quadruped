from absl import app, flags
import os
import dataclasses
import pickle

import numpy as np

import matplotlib.pyplot as plt

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
    directory = os.path.join(
        os.path.dirname(__file__),
        'data',
    )
    upright_data = os.path.join(
        directory,
        'upright.pkl',
    )
    down_data = os.path.join(
        directory,
        'down.pkl',
    )
    left_data = os.path.join(
        directory,
        'left.pkl',
    )
    flipped_data = os.path.join(
        directory,
        'flipped.pkl',
    )
    filepaths = [upright_data, down_data, left_data, flipped_data]
    hardware_data = []
    for filepath in filepaths:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            data = np.asarray(data)
            hardware_data.append(data)

    simulation_data = os.path.join(
        directory,
        'simulation_imu_data.pkl',
    )
    with open(filepath, 'rb') as file:
        simulation_data = pickle.load(file)

    names = ['upright', 'down', 'left', 'flipped']

    for hardware, simulation, name in zip(hardware_data, simulation_data, names):
        accelerometer_hardware_hardware = hardware[:, 0]
        gyroscope_hardware_hardware = hardware[:, 1]
        quaternion_hardware = hardware[:, 2]
        projected_gravity_hardware = []
        for quaternion in quaternion_hardware:
            inverse_base_rotation = quat_inv(quaternion)
            projected_gravity_hardware.append(
                rotate(
                    np.array([0, 0, -1]), inverse_base_rotation,
                )
            )
        projected_gravity_hardware = np.array(projected_gravity_hardware)

        accelerometer_simulation = simulation[:, 0]
        gyroscope_simulation = simulation[:, 1]
        projected_gravity_simulation = simulation[:, 2]

        # Plot and compare Accelerometer Data:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].plot(accelerometer_hardware[:, 0], label='Hardware Accelerometer X')
        axs[0].plot(accelerometer_simulation[:, 0], label='Simulation Accelerometer X')
        axs[0].set_title('Accelerometer X Data')
        axs[0].set_ylabel('Acceleration (m/s^2)')
        axs[0].legend()

        axs[1].plot(accelerometer_hardware[:, 1], label='Accelerometer Y')
        axs[0].plot(accelerometer_simulation[:, 1], label='Simulation Accelerometer Y')
        axs[1].set_title('Accelerometer Y Data')
        axs[1].set_ylabel('Acceleration (m/s^2)')
        axs[1].legend()

        axs[2].plot(accelerometer_hardware[:, 2], label='Accelerometer Z')
        axs[0].plot(accelerometer_simulation[:, 1], label='Simulation Accelerometer Z')
        axs[2].set_title('Accelerometer Z Data')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Acceleration (m/s^2)')
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(f'data/accelerometer_{name}.pdf')

        # Plot and compare Gyroscope Data:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].plot(gyroscope_hardware[:, 0], label='Hardware Gyroscope X')
        axs[0].plot(gyroscope_simulation[:, 0], label='Simulation Gyroscope X')
        axs[0].set_title('Gyroscope X Data')
        axs[0].set_ylabel('Angular Velocity (rad/s)')
        axs[0].legend()

        axs[1].plot(gyroscope_hardware[:, 1], label='Hardware Gyroscope Y')
        axs[1].plot(gyroscope_simulation[:, 1], label='Simulation Gyroscope Y')
        axs[1].set_title('Gyroscope Y Data')
        axs[1].set_ylabel('Angular Velocity (rad/s)')
        axs[1].legend()

        axs[2].plot(gyroscope_hardware[:, 2], label='Hardware Gyroscope Z')
        axs[2].plot(gyroscope_simulation[:, 2], label='Simulation Gyroscope Z')
        axs[2].set_title('Gyroscope Z Data')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Angular Velocity (rad/s)')
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(f'data/gyroscope_{name}.pdf')

        # Plot and compare RPY Data:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].plot(projected_gravity_hardware[:, 0], label='Hardware Projected Gravity X')
        axs[0].plot(projected_gravity_simulation[:, 0], label='Simulation Projected Gravity X')
        axs[0].set_title('Projected Gravity X Data')
        axs[0].set_ylabel('Projected Gravity (m/s^2)')
        axs[0].legend()
        
        axs[1].plot(projected_gravity_hardware[:, 1], label='Hardware Projected Gravity Y')
        axs[1].plot(projected_gravity_simulation[:, 1], label='Simulation Projected Gravity Y')
        axs[1].set_title('Projected Gravity Y Data')
        axs[1].set_ylabel('Projected Gravity (m/s^2)')
        axs[1].legend()

        axs[2].plot(projected_gravity_hardware[:, 2], label='Hardware Projected Gravity Z')
        axs[2].plot(projected_gravity_simulation[:, 2], label='Simulation Projected Gravity Z')
        axs[2].set_title('Projected Gravity Z Data')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Projected Gravity (m/s^2)')
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(f'data/projected_gravity_{name}.pdf')


if __name__ == "__main__":
    app.run(main)
