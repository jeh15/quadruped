from absl import app
import os
import pickle

import numpy as np

import matplotlib.pyplot as plt

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
    simulation_accelerometer_data = simulation_data.accelerometer
    simulation_gyroscope_data = simulation_data.gyroscope
    simulation_quaternion_data = simulation_data.quaternion

    hardware_accelerometer_data = hardware_data.accelerometer
    hardware_gyroscope_data = hardware_data.gyroscope
    hardware_quaternion_data = hardware_data.quaternion