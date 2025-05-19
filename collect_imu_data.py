from absl import app, flags
import os
import functools
import time
import pickle

import numpy as np

import pygame

from unitree_api_bindings import unitree_api


pygame.init()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'filename', None, 'Desired filename for pickle.', short_name='f',
)


def main(argv=None):
    if FLAGS.filename is None:
        raise ValueError('Please provide a filename for the pickle file.')

    # Initalize Unitree API:
    control_rate = 0.02
    control_rate_ns = 2e7

    # Initialize Unitree-Api:
    network_name = "enx7cc2c647de4f"
    inner_control_rate = 2000
    unitree_driver = unitree_api.UnitreeDriver(
        network_name,
        inner_control_rate,
    )
    unitree_driver.initialize()

    time.sleep(1.0)

    # Default Control:
    motor_commands = unitree_api.MotorCommand()
    motor_commands.q_setpoint = [0.0, 0.9, -1.8] * 4
    motor_commands.qd_setpoint = [0.0, 0.0, 0.0] * 4
    motor_commands.torque_feedforward = [0.0, 0.0, 0.0] * 4
    motor_commands.stiffness = [0.0, 0.0, 0.0] * 4
    motor_commands.damping = [0.0, 0.0, 0.0] * 4
    unitree_driver.update_command(motor_commands)

    time.sleep(1.0)

    # Collect Data:
    imu_states = []
    joysticks = {}
    is_running = True
    while is_running:
        next_time_ns += control_rate_ns

        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED:
                joy = pygame.joystick.Joystick(event.device_index)
                joysticks[joy.get_instance_id()] = joy
                print(f"Joystick {joy.get_instance_id()} connencted")

            if event.type == pygame.JOYDEVICEREMOVED:
                del joysticks[event.instance_id]
                print(f"Joystick {event.instance_id} disconnected")

        for joystick in joysticks.values():
            if joystick.get_button(0) == 1:
                string = 'Ending Test...'
                print(string)
                is_running = False
        
        imu_state = unitree_driver.get_imu_state()

        # Transform IMU State to numpy array:
        accelerometer = np.asarray(imu_state.accelerometer)
        gyroscope = np.asarray(imu_state.gyroscope)
        quaternion = np.asarray(imu_state.quaternion)
        rpy = np.asarray(imu_state.rpy)

        # Concatenate IMU State for csv file:
        imu_state = np.concatenate(
            (accelerometer, gyroscope, quaternion, rpy),
            axis=0,
        )
        
        imu_states.append(imu_state)

        now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        if now_ns < next_time_ns:
            sleep_time_ns = next_time_ns - now_ns
            time.sleep(sleep_time_ns / 1e9)
        else:
            print('Warning: Control rate exceeded.')
            next_time_ns = now_ns

        


    # Save Data:
    data = np.asarray(imu_states)

    # Save data to file:
    data_directory = os.path.join(
        os.path.dirname(__file__),
        'data',
    )
    os.makedirs(data_directory, exist_ok=True)
    data_file = os.path.join(
        data_directory,
        f'{FLAGS.filename}.pkl',
    )
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    app.run(main)