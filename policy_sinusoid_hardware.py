from absl import app, flags, logging
import os
import functools
import time
from dataclasses import dataclass

import pygame

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from unitree_api_bindings import unitree_api

from src.envs import sinusoid_test as unitree_go2
from src.algorithms.ppo.load_utilities import load_policy

jax.config.update("jax_enable_x64", True)
pygame.init()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)

def controller(
    action: npt.ArrayLike,
    default_control: npt.ArrayLike,
    action_scale: float,
) -> np.ndarray:
    motor_targets = default_control + action * action_scale
    return np.asarray(motor_targets)


def main(argv=None):
    # Set up Logger:
    logging.use_absl_handler()
    log_directory = os.path.join(
        os.path.dirname(__file__),
        'logs',
    )
    logging.get_absl_handler().use_absl_log_file(program_name='hardware_test', log_dir=log_directory) 
    logging.set_verbosity(logging.INFO)

    # Load from Env:
    env = unitree_go2.UnitreeGo2Env(filename='unitree_go2/scene_mjx_regressed_fixed.xml')

    control_rate = 0.02
    control_rate_ns = 2e7

    # Load Policy:
    make_policy, params, _ = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
    )
    inference_function = make_policy(params, deterministic=True)
    inference_fn = jax.jit(inference_function)

    # Controller:
    controller_fn = functools.partial(
        controller,
        default_control=env.default_ctrl,
        action_scale=env.action_scale,
    )

    # Initialize Unitree-Api:
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


    # Show State:
    imu_state = unitree_driver.get_imu_state()
    motor_state = unitree_driver.get_motor_state()
    base_rotation = np.asarray(imu_state.quaternion)
    print(f"Base Rotation: {base_rotation}")

    # Switch to Policy Kp and Kd
    motor_commands = unitree_api.MotorCommand()
    motor_commands.q_setpoint = [0.0, 0.9, -1.8] * 4
    motor_commands.qd_setpoint = [0.0, 0.0, 0.0] * 4
    motor_commands.torque_feedforward = [0.0, 0.0, 0.0] * 4
    motor_commands.stiffness = [35.0, 35.0, 35.0] * 4
    motor_commands.damping = [0.5, 0.5, 0.5] * 4
    unitree_driver.update_command(motor_commands)

    # Wait for Keyboard Input:
    print('Press any key to start the Control...')
    input()

    # Initialize Observation History:
    observation = np.zeros(env.num_observations)
    action = np.asarray(env.default_ctrl)
    command = np.array([0.0])
    history_length = 10
    for i in range(history_length):
        step_time = time.time()
        imu_state = unitree_driver.get_imu_state()
        motor_state = unitree_driver.get_motor_state()
        
        sleep_time = control_rate - (time.time() - step_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print('Warning: Control rate exceeded.')

    print(f'Observation History Completed...')

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    # Setup Joystick:
    joysticks = {}
    policy_control_mode = False
    damping_control_mode = False
    is_running = True

    # Data:
    joint_position_history = []
    joint_velocity_history = []
    action_history = []
    ctrl_history = []

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
                string = 'Switching to Policy Control Mode...'
                logging.info(string)
                print(string)
                policy_control_mode = True

            if joystick.get_button(1) == 1:
                string = 'Switching to Damping Control Mode...'
                logging.info(string)
                print(string)
                damping_control_mode = True
                policy_control_mode = False

            if joystick.get_button(9) == 1:
                string = 'Terminating...'
                logging.info(string)
                print(string)
                is_running = False
                damping_control_mode = True
                policy_control_mode = False

            command = -1 * joystick.get_axis(1)

        # Filter and Clip Command:
        command = np.array([
            command
        ])
        command = np.where(np.abs(command) < 0.1, 0.0, command)
        command = np.clip(command, -1.0, 1.0)
        command = 0.1 * command
        command = np.clip(command, -0.1, 0.1)

        key, subkey = jax.random.split(subkey)
        imu_state = unitree_driver.get_imu_state()
        motor_state = unitree_driver.get_motor_state()

        # Make Observation:
        observation = env.hardware_observation(
            motor_state=motor_state,
            command=command,
            previous_action=action,
        )

        action, _ = jax.block_until_ready(
            inference_fn(observation, subkey),
        )
        action = jax.device_put(action, jax.devices('cpu')[0])
        action = np.asarray(action)

        ctrl = controller_fn(action)
        ctrl = ctrl.astype(np.float32)

        action_list = action.tolist()
        q_setpoint = ctrl.tolist()

        # Append Data:
        if policy_control_mode:
            joint_position_history.append(motor_state.q)
            joint_velocity_history.append(motor_state.qd)
            action_history.append(action_list)
            ctrl_history.append(q_setpoint)

        # To Control the Robot:
        if policy_control_mode:
            motor_commands = unitree_api.MotorCommand()
            motor_commands.q_setpoint = q_setpoint
            motor_commands.qd_setpoint = [0.0, 0.0, 0.0] * 4
            motor_commands.torque_feedforward = [0.0, 0.0, 0.0] * 4
            motor_commands.stiffness = [35.0, 35.0, 35.0] * 4
            motor_commands.damping = [0.5, 0.5, 0.5] * 4
            unitree_driver.update_command(motor_commands)
            # policy_control_mode = False

        if damping_control_mode:
            motor_commands = unitree_api.MotorCommand()
            motor_commands.q_setpoint = ctrl.tolist()
            motor_commands.qd_setpoint = [0.0, 0.0, 0.0] * 4
            motor_commands.torque_feedforward = [0.0, 0.0, 0.0] * 4
            motor_commands.stiffness = [0.0, 0.0, 0.0] * 4
            motor_commands.damping = [5.0, 5.0, 5.0] * 4
            unitree_driver.update_command(motor_commands)

        now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        if now_ns < next_time_ns:
            sleep_time_ns = next_time_ns - now_ns
            time.sleep(sleep_time_ns / 1e9)
        else:
            print('Warning: Control rate exceeded.')
            next_time_ns = now_ns

    # Save Data:
    joint_position_data = np.asarray(joint_position_history)
    joint_velocity_data = np.asarray(joint_velocity_history)
    action_data = np.asarray(action_history)
    ctrl_data = np.asarray(ctrl_history)

    np.savetxt(
        os.path.join(log_directory, 'hardware_joint_position_data.txt'),
        joint_position_data,
        delimiter=',',
    )
    np.savetxt(
        os.path.join(log_directory, 'hardware_joint_velocity_data.txt'),
        joint_velocity_data,
        delimiter=',',
    )
    np.savetxt(
        os.path.join(log_directory, 'hardware_action_data.txt'),
        action_data,
        delimiter=',',
    )
    np.savetxt(
        os.path.join(log_directory, 'hardware_ctrl_data.txt'),
        ctrl_data,
        delimiter=',',
    )
    
    # Stop Thread:
    unitree_driver.stop_thread()


if __name__ == '__main__':
    app.run(main)
