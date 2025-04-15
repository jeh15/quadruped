import os
from absl import app, flags, logging
import functools
import time

import pygame

import jax
import numpy as np
import numpy.typing as npt

import mujoco
import mujoco.viewer

from src.envs import unitree_go2_v13 as unitree_go2
from src.algorithms.ppo.load_utilities import load_policy

# Filter Imports:
from dataclasses import dataclass
from scipy.signal import butter, sosfilt


jax.config.update("jax_enable_x64", True)
pygame.init()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)


@dataclass 
class SensorData:
    accelerometer: np.ndarray
    gyroscope: np.ndarray
    quaternion: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray


class Filter:
    def __init__(self, window_size: int, cutoff: float, fs: float, order: int = 5):
        self.data_size = 33
        self.window_size = window_size
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.queue = np.zeros((self.data_size, window_size))
        self.filtered_data = np.zeros_like(self.queue)
        self.lowpass_filter = butter(
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

    def add_data(self, data: SensorData):
        # Normalize Quaternion Estimate:
        quaternion_data = data.quaternion
        normalized_quaternion = quaternion_data / np.linalg.norm(quaternion_data)
        projected_gravity = self.rotate(
            vec=np.array([0, 0, -1]),
            quat=self.quat_inv(normalized_quaternion)
        )
        # Add data to queue:
        self.queue = np.roll(self.queue, -1, axis=-1)
        self.queue[:, -1] = np.concatenate([
            data.accelerometer,
            data.gyroscope,
            projected_gravity,
            data.joint_position,
            data.joint_velocity,
        ])

    def apply_filter(self, data: SensorData):
        # Add data to queue:
        self.add_data(data)
        self.filtered_data = sosfilt(self.lowpass_filter, self.queue, axis=-1)

    def get_acceleration(self) -> np.ndarray:
        start = 0
        size = 3
        accelerometer_id = slice(start, start+size)
        return self.filtered_data[accelerometer_id, -1]
    
    def get_angular_velocity(self) -> np.ndarray:
        start = 3
        size = 3
        gyroscope_id = slice(start, start+size)
        return self.filtered_data[gyroscope_id, -1]
    
    def get_projected_gravity(self) -> np.ndarray:
        start = 6
        size = 3
        projected_gravity_id = slice(start, start+size)
        return self.filtered_data[projected_gravity_id, -1]
    
    def get_joint_position(self) -> np.ndarray:
        start = 9
        size = 12
        joint_position_id = slice(start, start+size)
        return self.filtered_data[joint_position_id, -1]
    
    def get_joint_velocity(self) -> np.ndarray:
        start = 21
        size = 12
        joint_velocity_id = slice(start, start+size)
        return self.filtered_data[joint_velocity_id, -1]
    

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
    # # Set up Logger:
    # logging.use_absl_handler()
    # log_directory = os.path.join(
    #     os.path.dirname(__file__),
    #     'logs',
    # )
    # logging.get_absl_handler().use_absl_log_file(program_name='simulation_test', log_dir=log_directory) 
    # logging.set_verbosity(logging.INFO)

    # Load from Env:
    env = unitree_go2.UnitreeGo2Env(filename='unitree_go2/scene_mjx.xml', action_scale=0.5)
    model_mjx = env.sys.mj_model
    
    # High Fidelity Model:
    model_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx.xml',
    )

    model = mujoco.MjModel.from_xml_path(
        model_path,
    )
    model.opt.timestep = 0.004

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    control_rate = 0.02
    num_steps = int(control_rate / model.opt.timestep)

    # Load Policy:
    make_policy, params, _ = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
        restore_iteration=FLAGS.checkpoint_iteration,
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

    lowpass_filter = Filter(
        window_size=10,
        cutoff=3.0,
        fs=0.02,
        order=5,
    )

    # Test:
    data.qpos = model_mjx.key_qpos.flatten()
    command = np.array([0.0, 0.0, 0.0])
    action = model_mjx.key_ctrl.flatten()
    observation = np.zeros(env.num_observations)

    # Setup Joystick:
    joysticks = {}

    key = jax.random.key(0)
    termination_flag = False

    global_steps = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5

        while viewer.is_running() and not termination_flag:
            for event in pygame.event.get():
                if event.type == pygame.JOYDEVICEADDED:
                    joy = pygame.joystick.Joystick(event.device_index)
                    joysticks[joy.get_instance_id()] = joy
                    print(f"Joystick {joy.get_instance_id()} connencted")

                if event.type == pygame.JOYDEVICEREMOVED:
                    del joysticks[event.instance_id]
                    print(f"Joystick {event.instance_id} disconnected")

            for joystick in joysticks.values():
                # If Switch Controller:
                # if joystick.get_button(11) == 1:
                #     termination_flag = True
                # In Logitech Controller:
                if joystick.get_button(7) == 1:
                    termination_flag = True

                forward_command = -1 * joystick.get_axis(1)
                lateral_command = -1 * joystick.get_axis(0)
                # If Switch Controller:
                # rotation_command = -1 * joystick.get_axis(2)
                # If Logitech Controller:
                rotation_command = -1 * joystick.get_axis(3)


            # Filter and Clip Command:
            command = np.array([
                forward_command, lateral_command, rotation_command,
            ])
            command = np.where(np.abs(command) < 0.1, 0.0, command)
            command = np.clip(command, -0.75, 0.75)

            step_time = time.time()
            action_rng, key = jax.random.split(key)

            # Test Filtering:
            sensor_data = env.get_noisy_sensor_data(data)
            sensor_data = SensorData(
                accelerometer=sensor_data[0:3],
                gyroscope=sensor_data[3:6],
                quaternion=sensor_data[6:10],
                joint_position=sensor_data[10:22],
                joint_velocity=sensor_data[22:34],
            )
            lowpass_filter.apply_filter(sensor_data)
            filtered_accelerometer = lowpass_filter.get_acceleration()
            filtered_angular_velocity = lowpass_filter.get_angular_velocity()
            filtered_projected_gravity = lowpass_filter.get_projected_gravity()
            filtered_joint_position = lowpass_filter.get_joint_position()
            filtered_joint_velocity = lowpass_filter.get_joint_velocity()

            # Get Observation:
            observation = env.observation_test(
                accelerometer=filtered_accelerometer,
                gyroscope=filtered_angular_velocity,
                projected_gravity=filtered_projected_gravity,
                joint_positions=filtered_joint_position,
                joint_velocities=filtered_joint_velocity,
                command=command,
                previous_action=action,
            )
            action, _ = inference_fn(observation, action_rng)
            ctrl = controller_fn(action)

            # data.ctrl = ctrl
            data.ctrl = ctrl

            for _ in range(num_steps):
                mujoco.mj_step(model, data)  # type: ignore

            viewer.sync()

            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            global_steps += 1


if __name__ == '__main__':
    app.run(main)
