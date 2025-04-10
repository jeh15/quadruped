from absl import app, logging
import os
import time

import pygame

import numpy as np

from unitree_api_bindings import unitree_api


pygame.init()


def main(argv=None):
    # Set up Logger:
    logging.use_absl_handler()
    log_directory = os.path.join(
        os.path.dirname(__file__),
        'logs',
    )
    logging.get_absl_handler().use_absl_log_file(program_name='hardware_test', log_dir=log_directory) 
    logging.set_verbosity(logging.INFO)

    control_rate = 0.02
    control_rate_ns = 2e7

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

    print('Press any key to start get up sequence...')
    input()

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


    # Wait for Keyboard Input:
    print('Press any key to start the Control...')
    input()

    # Choose Joint To Control:
    joint_id = 9


    # Setup Joystick:
    joysticks = {}
    joystick_control_mode = False
    damping_control_mode = False
    is_running = True
    q_setpoint = [0.0, 0.9, -1.8] * 4
    q_default = [0.0, 0.9, -1.8] * 4
    command = q_default[joint_id]
    next_time_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
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
                string = 'Switching to Joystick Control Mode...'
                logging.info(string)
                print(string)
                joystick_control_mode = True

            if joystick.get_button(7) == 1:
                string = 'Switching to Damping Control Mode...'
                logging.info(string)
                print(string)
                damping_control_mode = True
                joystick_control_mode = False

            if joystick.get_button(6) == 1:
                string = 'Terminating...'
                logging.info(string)
                print(string)
                is_running = False
                damping_control_mode = True
                joystick_control_mode = False

            forward_command = -1 * joystick.get_axis(1)
            lateral_command = -1 * joystick.get_axis(0)
            rotation_command = -1 * joystick.get_axis(3)

        # Filter and Clip Command:
        forward_command = np.clip(forward_command, -0.11, 0.11)
        forward_command = np.where(np.abs(forward_command) < 0.1, 0.0, forward_command)
        command += forward_command
        lb = -0.5 + q_default[joint_id]
        ub = 0.5 + q_default[joint_id]
        command = np.clip(command, lb, ub)
        command_pyval = (command.astype(np.float32)).item()

        motor_state = unitree_driver.get_motor_state()

        # To Control the Robot:
        if joystick_control_mode:
            motor_commands = unitree_api.MotorCommand()
            q_setpoint[joint_id] = command_pyval
            motor_commands.q_setpoint = q_setpoint
            motor_commands.qd_setpoint = [0.0, 0.0, 0.0] * 4
            motor_commands.torque_feedforward = [0.0, 0.0, 0.0] * 4
            motor_commands.stiffness = [35.0, 35.0, 35.0] * 4
            motor_commands.damping = [0.5, 0.5, 0.5] * 4
            unitree_driver.update_command(motor_commands)

        if damping_control_mode:
            motor_commands = unitree_api.MotorCommand()
            motor_commands.q_setpoint = [0.0, 0.9, -1.8] * 4
            motor_commands.qd_setpoint = [0.0, 0.0, 0.0] * 4
            motor_commands.torque_feedforward = [0.0, 0.0, 0.0] * 4
            motor_commands.stiffness = [0.0, 0.0, 0.0] * 4
            motor_commands.damping = [5.0, 5.0, 5.0] * 4
            unitree_driver.update_command(motor_commands)

        print(f'Motor Command: {q_setpoint}')
        print(f'Joint Position: {motor_state.q}')
        print(f'Joint Velocity: {motor_state.qd}')

        now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        if now_ns < next_time_ns:
            sleep_time_ns = next_time_ns - now_ns
            time.sleep(sleep_time_ns / 1e9)
        else:
            print('Warning: Control rate exceeded.')
            next_time_ns = now_ns

    
    # Stop Thread:
    unitree_driver.stop_thread()


if __name__ == '__main__':
    app.run(main)
