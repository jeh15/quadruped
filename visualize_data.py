from absl import app
import os
import dataclasses
import pickle

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

from unitree_api_bindings import unitree_api


@dataclasses.dataclass
class IMUState:
    accelerometer: npt.NDArray[np.float64]
    gyroscope: npt.NDArray[np.float64]
    quaternion: npt.NDArray[np.float64]


@dataclasses.dataclass
class MotorState:
    joint_positions: npt.NDArray[np.float64]
    joint_velocities: npt.NDArray[np.float64]
    joint_torques: npt.NDArray[np.float64]


def main(argv=None):
    # Load the data
    simulation_data_path = os.path.join(
        os.path.dirname(__file__),
        'data/simulation_imu_data.pkl',
    )
    with open(simulation_data_path, 'rb') as file:
        simulation_data = pickle.load(file)

    hardware_data_path = os.path.join(
        os.path.dirname(__file__),
        'data/hardware_imu_data.pkl',
    )
    with open(hardware_data_path, 'rb') as file:
        hardware_data = pickle.load(file)

    # Extract the data
    simulation_accelerometer_data = np.asarray(list(map(lambda x: x.accelerometer, simulation_data)))
    simulation_gyroscope_data = np.asarray(list(map(lambda x: x.gyroscope, simulation_data)))
    simulation_quaternion_data = np.asarray(list(map(lambda x: x.quaternion, simulation_data)))

    hardware_accelerometer_data = np.asarray(list(map(lambda x: x.accelerometer, hardware_data)))
    hardware_gyroscope_data = np.asarray(list(map(lambda x: x.gyroscope, hardware_data)))
    hardware_quaternion_data = np.asarray(list(map(lambda x: x.quaternion, hardware_data)))

    # Plot Data:
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(simulation_accelerometer_data, label='Simulation Accelerometer')
    axs[0].plot(hardware_accelerometer_data, label='Hardware Accelerometer')
    axs[0].set_title('Accelerometer Data')
    axs[0].set_ylabel('Acceleration (m/s^2)')
    axs[0].legend()

    axs[1].plot(simulation_gyroscope_data, label='Simulation Gyroscope')
    axs[1].plot(hardware_gyroscope_data, label='Hardware Gyroscope')
    axs[1].set_title('Gyroscope Data')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[1].legend()

    axs[2].plot(simulation_quaternion_data, label='Simulation Quaternion')
    axs[2].plot(hardware_quaternion_data, label='Hardware Quaternion')
    axs[2].set_title('Quaternion Data')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Quaternion')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()
    plt.savefig('data/imu_comparison.pdf')

    # Plot Only Simulation Data:
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(simulation_accelerometer_data)
    axs[0].set_title('Simulation Accelerometer Data')
    axs[0].set_ylabel('Acceleration (m/s^2)')
    axs[1].plot(simulation_gyroscope_data)
    axs[1].set_title('Simulation Gyroscope Data')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[2].plot(simulation_quaternion_data)
    axs[2].set_title('Simulation Quaternion Data')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Quaternion')

    plt.tight_layout()
    plt.show()
    plt.savefig('data/simulation_imu_data.pdf')

    # Plot Only Hardware Data:
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(hardware_accelerometer_data)
    axs[0].set_title('Hardware Accelerometer Data')
    axs[0].set_ylabel('Acceleration (m/s^2)')
    axs[1].plot(hardware_gyroscope_data)
    axs[1].set_title('Hardware Gyroscope Data')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[2].plot(hardware_quaternion_data)
    axs[2].set_title('Hardware Quaternion Data')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Quaternion')

    plt.tight_layout()
    plt.show()
    plt.savefig('data/hardware_imu_data.pdf')


if __name__ == "__main__":
    app.run(main)
