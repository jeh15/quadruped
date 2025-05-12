from absl import app
import os
import time
import pickle

import numpy as npnpt

from unitree_api_bindings import unitree_api


def main(argv=None):
    # Initialize Unitree-Api:
    control_rate = 0.02
    control_rate_ns = 2e7
    network_name = "enx7cc2c647de4f"
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

    time.sleep(1.0)

    # Data:
    imu_history = []
    motor_history = []

    is_running = True
    next_time_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    for i in range(500):
        next_time_ns += control_rate_ns

        imu_state = unitree_driver.get_imu_state()
        motor_state = unitree_driver.get_motor_state()

        imu_history.append(imu_state)
        motor_history.append(motor_state)

        now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        if now_ns < next_time_ns:
            sleep_time_ns = next_time_ns - now_ns
            time.sleep(sleep_time_ns / 1e9)
        else:
            print('Warning: Control rate exceeded.')
            next_time_ns = now_ns
    
    # Pickle Data:
    data_directory = os.path.join(
        os.path.dirname(__file__),
        'data',
    )
    os.makedirs(data_directory, exist_ok=True)
    imu_data_file = os.path.join(
        data_directory,
        'hardware_imu_data.pkl',
    )
    motor_data_file = os.path.join(
        data_directory,
        'hardware_motor_data.pkl',
    )

    with open(imu_data_file, 'wb') as f:
        pickle.dump(imu_history, f)

    with open(motor_data_file, 'wb') as f:
        pickle.dump(motor_history, f)


if __name__ == '__main__':
    app.run(main)
