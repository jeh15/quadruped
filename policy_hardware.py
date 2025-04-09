from absl import app, flags, logging
import os
import functools
import time

import pygame

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from unitree_api_bindings import unitree_api

import mujoco

from src.envs import unitree_go2_v8 as unitree_go2
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
    ctrl_lb: npt.ArrayLike,
    ctrl_ub: npt.ArrayLike,
    action_scale: float,
) -> np.ndarray:
    motor_targets = default_control + action * action_scale
    motor_targets = np.clip(motor_targets, ctrl_lb, ctrl_ub)
    return motor_targets


def main(argv=None):
    # Set up Logger:
    logging.use_absl_handler()
    log_directory = os.path.join(
        os.path.dirname(__file__),
        'logs',
    )
    logging.get_absl_handler().use_absl_log_file(program_name='policy_spoof', log_dir=log_directory) 
    logging.set_verbosity(logging.INFO)

    # Load from Env:
    env = unitree_go2.UnitreeGo2Env(filename='unitree_go2/scene_mjx.xml')
    model = env.sys.mj_model

    data = mujoco.MjData(model)
    control_rate = 0.02
    control_rate_ns = 2e7
    num_physics_steps = int(control_rate / model.opt.timestep)

    # Load Policy:
    make_policy, params = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
    )
    inference_function = make_policy(params, deterministic=True)
    inference_fn = jax.jit(inference_function)

    # Controller:
    controller_fn = functools.partial(
        controller,
        default_control=env.default_ctrl,
        ctrl_lb=env.ctrl_lb,
        ctrl_ub=env.ctrl_ub,
        action_scale=env._action_scale,
    )

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

    # Wait for Keyboard Input:
    print('Press any key to Calibrate IMU...')
    input()

    acceleration_data = []
    orientation_data = []

    start_time = time.time()
    while (time.time() - start_time) < 10.0:
        step_time = time.time()
        imu_state = unitree_driver.get_imu_state()
        
        # Update Data:
        acceleration_data.append(np.asarray(imu_state.accelerometer))
        orientation_data.append(np.asarray(imu_state.quaternion))

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
    orientation_data = np.array(orientation_data)

    # Calculate Mean and Standard Deviation:
    acceleration_mean = np.mean(acceleration_data, axis=0)
    orientation_mean = np.mean(orientation_data, axis=0)

    # Rotate Acceleration to World Frame to calculate Bias:
    r = R.from_quat(orientation_mean)
    acceleration_world_nt = r.as_matrix() @ acceleration_mean
    acceleration_world_t = r.as_matrix().T @ acceleration_mean
    accelerometer_bias_nt = np.array([0.0, 0.0, -9.81]) - acceleration_world_nt
    accelerometer_bias_t = np.array([0.0, 0.0, -9.81]) - acceleration_world_t

    print(f'Accelerometer Bias (Not Transposed): {accelerometer_bias_nt}')
    print(f'Accelerometer Bias (Transposed): {accelerometer_bias_t}')

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

    # Wait for Keyboard Input:
    print('Press any key to start the Control...')
    input()

    # Initialize Observation History:
    observation = np.zeros(env.num_observations)
    action = np.asarray(env.default_ctrl)
    command = np.array([0.0, 0.0, 0.0])
    history_length = 10
    for i in range(history_length):
        step_time = time.time()
        imu_state = unitree_driver.get_imu_state()
        motor_state = unitree_driver.get_motor_state()
        observation = env.hardware_observation(
            imu_state=imu_state,
            motor_state=motor_state,
            command=command,
            previous_action=action,
        )
        sleep_time = control_rate - (time.time() - step_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print('Warning: Control rate exceeded.')

    print(f'Observation History Completed...')

    # Update Data:
    data.qpos = model.key_qpos.flatten()
    mujoco.mj_forward(model, data) 

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    # Setup Joystick:
    joysticks = {}
    policy_control_mode = False
    damping_control_mode = False
    is_running = True
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

            if joystick.get_button(7) == 1:
                string = 'Switching to Damping Control Mode...'
                logging.info(string)
                print(string)
                damping_control_mode = True
                policy_control_mode = False

            if joystick.get_button(6) == 1:
                string = 'Terminating...'
                logging.info(string)
                print(string)
                is_running = False
                damping_control_mode = True
                policy_control_mode = False

            forward_command = -1 * joystick.get_axis(1)
            lateral_command = -1 * joystick.get_axis(0)
            rotation_command = -1 * joystick.get_axis(3)

        # Filter and Clip Command:
        command = np.array([
            forward_command, lateral_command, rotation_command,
        ])
        command = np.where(np.abs(command) < 0.1, 0.0, command)
        command = np.clip(command, -0.75, 0.75)

        key, subkey = jax.random.split(subkey)
        imu_state = unitree_driver.get_imu_state()
        motor_state = unitree_driver.get_motor_state()
        observation = env.hardware_observation(
            imu_state=imu_state,
            motor_state=motor_state,
            command=command,
            previous_action=action,
        )
        action, _ = inference_fn(observation, subkey)
        action.block_until_ready()
        action = jax.device_put(action, jax.devices('cpu')[0])
        action = np.asarray(action)
        ctrl = controller_fn(action)
        ctrl = ctrl.astype(np.float32)

        # To Control the Robot:
        if policy_control_mode:
            motor_commands = unitree_api.MotorCommand()
            motor_commands.q_setpoint = ctrl.tolist()
            motor_commands.qd_setpoint = [0.0, 0.0, 0.0] * 4
            motor_commands.torque_feedforward = [0.0, 0.0, 0.0] * 4
            motor_commands.stiffness = [35.0, 35.0, 35.0] * 4
            motor_commands.damping = [0.5, 0.5, 0.5] * 4
            unitree_driver.update_command(motor_commands)

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

    
    # Stop Thread:
    unitree_driver.stop_thread()


if __name__ == '__main__':
    app.run(main)
