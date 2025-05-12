from absl import app
import os
import dataclasses
import pickle

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

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
    simulation_gyroscope_data = np.asarray(list(map(lambda x: x.groscope, simulation_data)))
    simulation_quaternion_data = np.asarray(list(map(lambda x: x.quaternion, simulation_data)))

    hardware_accelerometer_data = np.asarray(list(map(lambda x: x.accelerometer, hardware_data)))
    hardware_gyroscope_data = np.asarray(list(map(lambda x: x.groscope, hardware_data)))
    hardware_quaternion_data = np.asarray(list(map(lambda x: x.quaternion, hardware_data)))


if __name__ == "__main__":
    app.run(main)
