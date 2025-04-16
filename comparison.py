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
from scipy.spatial.transform import Rotation as R
from scipy import signal

from unitree_api_bindings import unitree_api

from src.envs import unitree_go2_v13 as unitree_go2
from src.algorithms.ppo.load_utilities import load_policy

jax.config.update("jax_enable_x64", True)
pygame.init()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)


@dataclass 
class FilterData:
    accelerometer: np.ndarray
    gyroscope: np.ndarray
    projected_gravity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray


class Filter:
    def __init__(self, window_size: int, cutoff: float, fs: float, order: int = 5):
        assert window_size > 0, 'Window size must be greater than 0.'
        assert cutoff > 0, 'Cutoff frequency must be greater than 0.'
        assert fs > 0, 'Sampling frequency must be greater than 0.'
        assert order > 0, 'Order must be greater than 0.'
        
        # Sensor Index:
        self.accelerometer_id = slice(0, 3)
        self.gyroscope_id = slice(3, 6)
        self.projected_gravity_id = slice(6, 9)
        self.joint_position_id = slice(9, 21)
        self.joint_velocity_id = slice(21, 33)
        self.data_size = 33
        
        # Initialize Filter:
        self.window_size = window_size
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.queue = np.zeros((self.data_size, self.window_size))
        self.filtered_data = np.zeros_like(self.queue)
        self.lowpass_filter = signal.butter(
            N=order,
            Wn=cutoff,
            btype='low',
            analog=False,
            output='sos',
            fs=fs,
        )
    
    @staticmethod
    def rotate(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
            if len(vec.shape) != 1:
                raise ValueError('vec must have no batch dimensions.')
            s, u = quat[0], quat[1:]
            r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
            r = r + 2 * s * np.cross(u, vec)
            return r
    
    @staticmethod
    def quat_inv(q: np.ndarray) -> np.ndarray:
        return q * np.array([1, -1, -1, -1])

    def add_data(self, imu_state: unitree_api.IMUState, motor_state: unitree_api.MotorState):
        # Unpack Data to Numpy Arrays:
        quaternion = np.asarray(imu_state.quaternion, dtype=np.float32)
        accelerometer = np.asarray(imu_state.accelerometer, dtype=np.float32)
        gyroscope = np.asarray(imu_state.gyroscope, dtype=np.float32)
        joint_positions = np.asarray(motor_state.q, dtype=np.float32)
        joint_velocities = np.asarray(motor_state.qd, dtype=np.float32)

        # Cast to float64:
        quaternion = quaternion.astype(np.float64)
        accelerometer = accelerometer.astype(np.float64)
        gyroscope = gyroscope.astype(np.float64)
        joint_positions = joint_positions.astype(np.float64)
        joint_velocities = joint_velocities.astype(np.float64)

        # Normalize Quaternion Estimate and Calculate Projected Gravity:
        normalized_quaternion = quaternion / np.linalg.norm(quaternion)
        projected_gravity = self.rotate(
            vec=np.array([0, 0, -1]),
            quat=self.quat_inv(normalized_quaternion)
        )

        # Add data to queue:
        self.queue = np.roll(self.queue, -1, axis=-1)
        self.queue[:, -1] = np.concatenate([
            accelerometer,
            gyroscope,
            projected_gravity,
            joint_positions,
            joint_velocities,
        ])

    def apply_filter(self) -> FilterData:
        self.filtered_data = signal.sosfilt(self.lowpass_filter, self.queue, axis=-1)
        return FilterData(
            accelerometer=self.filtered_data[self.accelerometer_id, -1],
            gyroscope=self.filtered_data[self.gyroscope_id, -1],
            projected_gravity=self.filtered_data[self.projected_gravity_id, -1],
            joint_position=self.filtered_data[self.joint_position_id, -1],
            joint_velocity=self.filtered_data[self.joint_velocity_id, -1],
        )

    def get_acceleration(self) -> np.ndarray:
        return self.filtered_data[self.accelerometer_id, -1]
    
    def get_angular_velocity(self) -> np.ndarray:
        return self.filtered_data[self.gyroscope_id, -1]
    
    def get_projected_gravity(self) -> np.ndarray:
        return self.filtered_data[self.projected_gravity_id, -1]
    
    def get_joint_position(self) -> np.ndarray:
        return self.filtered_data[self.joint_position_id, -1]
    
    def get_joint_velocity(self) -> np.ndarray:
        return self.filtered_data[self.joint_velocity_id, -1]
    

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
    env = unitree_go2.UnitreeGo2Env(filename='unitree_go2/scene_mjx.xml', action_scale=0.5)

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
        action_scale=env._action_scale,
    )

    # Initialize Filter:
    sample_rate = int(1 / control_rate)
    lowpass_filter = Filter(
        window_size=10,
        cutoff=10.0,
        fs=sample_rate,
        order=5,
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

        # Filter:
        lowpass_filter.add_data(imu_state, motor_state)
        filtered_data = lowpass_filter.apply_filter()
        
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
        
        # Filter:
        lowpass_filter.add_data(imu_state, motor_state)
        filtered_data = lowpass_filter.apply_filter()

        # Make Observation:
        observation = np.concatenate([
            filtered_data.gyroscope,
            filtered_data.projected_gravity,
            filtered_data.joint_position - env.default_ctrl,
            filtered_data.joint_velocity,
            action,
            command,
        ])

        obs = {
            'state': observation,
            'privileged_state': np.zeros((env.num_privileged_observations,)),
        }

        action, _ = jax.block_until_ready(
            inference_fn(obs, subkey),
        )
        action = jax.device_put(action, jax.devices('cpu')[0])
        action = np.asarray(action)

        ctrl = controller_fn(action)
        ctrl = ctrl.astype(np.float32)

        action_list = action.tolist()
        q_setpoint = ctrl.tolist()

        # # Log Data:
        # logging.info(f'Accelerometer: {imu_state.accelerometer}')
        # logging.info(f'Gyroscope: {imu_state.gyroscope}')
        # logging.info(f'Quaternion: {imu_state.quaternion}')
        # logging.info(f'Joint Position: {motor_state.q}')
        # logging.info(f'Joint Velocity: {motor_state.qd}')
        # logging.info(f'Action: {action_list}')
        # logging.info(f'Command: {q_setpoint}')

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

    
    # Stop Thread:
    unitree_driver.stop_thread()


if __name__ == '__main__':
    app.run(main)
