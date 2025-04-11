import os
from absl import app, logging
import time


import jax
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy.typing as npt

from unitree_api_bindings import unitree_api

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


def main(argv=None):
    logging.use_absl_handler()
    log_directory = os.path.join(
        os.path.dirname(__file__),
        'logs',
    )
    logging.get_absl_handler().use_absl_log_file(program_name='calibration', log_dir=log_directory) 
    logging.set_verbosity(logging.INFO)

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
    r = R.from_quat(orientation_mean, scalar_first=True)
    acceleration_world = r.as_matrix().T @ acceleration_mean
    bias = acceleration_world + np.array([0.0, 0.0, -9.81])

    # Log Data:
    logging.info(f'Accelerometer Mean: {acceleration_mean}')
    logging.info(f'Accelerometer Std: {acceleration_std}')
    logging.info(f'Gyroscope Mean: {angular_velocity_mean}')
    logging.info(f'Gyroscope Std: {angular_velocity_std}')
    logging.info(f'Orientation Mean: {orientation_mean}')
    logging.info(f'Orientation Std: {orientation_std}')
    logging.info(f'Joint Velocity Mean: {joint_velocity_mean}')
    logging.info(f'Joint Velocity Std: {joint_velocity_std}')
    logging.info(f'Accelerometer Bias: {bias}')

    # Save to CSV for easy Loading:
    np.savetxt('csv/accelerometer_data.csv', (acceleration_mean, acceleration_std), delimiter=',')
    np.savetxt('csv/gyroscope_data.csv', (angular_velocity_mean, angular_velocity_std), delimiter=',')
    np.savetxt('csv/orientation_data.csv', (orientation_mean, orientation_std), delimiter=',')
    np.savetxt('csv/joint_velocity_data.csv', (joint_velocity_mean, joint_velocity_std), delimiter=',')
    np.savetxt('csv/bias.csv', bias, delimiter=',')

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
