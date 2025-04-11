import os
from absl import app, logging
import time


import jax
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy.typing as npt

from unitree_api_bindings import unitree_api

import matplotlib.pyplot as plt


def main(argv=None):
    # Load CSVs:
    acceleration_data = np.loadtxt('csv/accelerometer_data.csv', delimiter=',')
    gyroscope_data = np.loadtxt('csv/gyroscope_data.csv', delimiter=',')
    orientation_data = np.loadtxt('csv/orientation_data.csv', delimiter=',')
    joint_velocity_data = np.loadtxt('csv/joint_velocity_data.csv', delimiter=',')
    bias = np.loadtxt('csv/bias.csv', delimiter=',')

    accelerometer_mean = acceleration_data[0]
    accelerometer_std = acceleration_data[1]
    gyroscope_mean = gyroscope_data[0]
    gyroscope_std = gyroscope_data[1]
    orientation_mean = orientation_data[0]
    orientation_std = orientation_data[1]
    joint_velocity_mean = joint_velocity_data[0]

    r = R.from_quat(orientation_mean)
    acceleration_world = r.as_matrix().T @ accelerometer_mean
    bias = acceleration_world + np.array([0.0, 0.0, 9.81])
    

    pass


if __name__ == '__main__':
    app.run(main)
