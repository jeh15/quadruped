from absl import app, flags, logging
import os
import time

from dataclasses import dataclass
import collections

import pygame

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import signal

import matplotlib.pyplot as plt

from unitree_api_bindings import unitree_api

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

    time.sleep(1.0)

    # Show State:
    imu_state = unitree_driver.get_imu_state()
    motor_state = unitree_driver.get_motor_state()
    base_rotation = np.asarray(imu_state.quaternion)
    print(f"Base Rotation: {base_rotation}")

    # Initialize Filter History:
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


    # Setup Joystick:
    is_running = True

    # Initialize Plots:
    fig, axs = plt.subplots(5, 1, figsize=(10, 6))
    accel_plt, = axs[0].plot([], [], label='Sensor')
    accel_filt_plt, = axs[0].plot([], [], label='Filtered')
    gyro_plt, = axs[1].plot([], [], label='Sensor')
    gyro_filt_plt, = axs[1].plot([], [], label='Filtered')
    projected_gravity_plt, = axs[2].plot([], [], label='Sensor')
    projected_gravity_filt_plt, = axs[2].plot([], [], label='Filtered')
    joint_position_plt, = axs[3].plot([], [], label='Sensor')
    joint_position_filt_plt, = axs[3].plot([], [], label='Filtered')
    joint_velocity_plt, = axs[4].plot([], [], label='Sensor')
    joint_velocity_filt_plt, = axs[4].plot([], [], label='Filtered')
    axs[0].set_title('Accelerometer')
    axs[1].set_title('Gyroscope')
    axs[2].set_title('Projected Gravity')
    axs[3].set_title('Joint Position')
    axs[4].set_title('Joint Velocity')
    axs[4].set_xlabel('Sample')

    plt.show(block=False)
    plt.draw()

    # Create a queue for the data:
    accel_x = collections.deque(maxlen=100)
    accel_y = collections.deque(maxlen=100)
    accel_z = collections.deque(maxlen=100)
    gyro_x = collections.deque(maxlen=100)
    gyro_y = collections.deque(maxlen=100)
    gyro_z = collections.deque(maxlen=100)
    projected_gravity_x = collections.deque(maxlen=100)
    projected_gravity_y = collections.deque(maxlen=100)
    projected_gravity_z = collections.deque(maxlen=100)
    joint_position_1 = collections.deque(maxlen=100)
    joint_position_2 = collections.deque(maxlen=100)
    joint_position_3 = collections.deque(maxlen=100)
    joint_position_4 = collections.deque(maxlen=100)
    joint_position_5 = collections.deque(maxlen=100)
    joint_position_6 = collections.deque(maxlen=100)
    joint_position_7 = collections.deque(maxlen=100)
    joint_position_8 = collections.deque(maxlen=100)
    joint_position_9 = collections.deque(maxlen=100)
    joint_position_10 = collections.deque(maxlen=100)
    joint_position_11 = collections.deque(maxlen=100)
    joint_position_12 = collections.deque(maxlen=100)
    joint_velocity_1 = collections.deque(maxlen=100)
    joint_velocity_2 = collections.deque(maxlen=100)
    joint_velocity_3 = collections.deque(maxlen=100)
    joint_velocity_4 = collections.deque(maxlen=100)
    joint_velocity_5 = collections.deque(maxlen=100)
    joint_velocity_6 = collections.deque(maxlen=100)
    joint_velocity_7 = collections.deque(maxlen=100)
    joint_velocity_8 = collections.deque(maxlen=100)
    joint_velocity_9 = collections.deque(maxlen=100)
    joint_velocity_10 = collections.deque(maxlen=100)
    joint_velocity_11 = collections.deque(maxlen=100)
    joint_velocity_12 = collections.deque(maxlen=100)

    accel_x_filtered = collections.deque(maxlen=100)
    accel_y_filtered = collections.deque(maxlen=100)
    accel_z_filtered = collections.deque(maxlen=100)
    gyro_x_filtered = collections.deque(maxlen=100)
    gyro_y_filtered = collections.deque(maxlen=100)
    gyro_z_filtered = collections.deque(maxlen=100)
    projected_gravity_x_filtered = collections.deque(maxlen=100)
    projected_gravity_y_filtered = collections.deque(maxlen=100)
    projected_gravity_z_filtered = collections.deque(maxlen=100)
    joint_position_1_filtered = collections.deque(maxlen=100)
    joint_position_2_filtered = collections.deque(maxlen=100)
    joint_position_3_filtered = collections.deque(maxlen=100)
    joint_position_4_filtered = collections.deque(maxlen=100)
    joint_position_5_filtered = collections.deque(maxlen=100)
    joint_position_6_filtered = collections.deque(maxlen=100)
    joint_position_7_filtered = collections.deque(maxlen=100)
    joint_position_8_filtered = collections.deque(maxlen=100)
    joint_position_9_filtered = collections.deque(maxlen=100)
    joint_position_10_filtered = collections.deque(maxlen=100)
    joint_position_11_filtered = collections.deque(maxlen=100)
    joint_position_12_filtered = collections.deque(maxlen=100)
    joint_velocity_1_filtered = collections.deque(maxlen=100)
    joint_velocity_2_filtered = collections.deque(maxlen=100)
    joint_velocity_3_filtered = collections.deque(maxlen=100)
    joint_velocity_4_filtered = collections.deque(maxlen=100)
    joint_velocity_5_filtered = collections.deque(maxlen=100)
    joint_velocity_6_filtered = collections.deque(maxlen=100)
    joint_velocity_7_filtered = collections.deque(maxlen=100)
    joint_velocity_8_filtered = collections.deque(maxlen=100)
    joint_velocity_9_filtered = collections.deque(maxlen=100)
    joint_velocity_10_filtered = collections.deque(maxlen=100)
    joint_velocity_11_filtered = collections.deque(maxlen=100)
    joint_velocity_12_filtered = collections.deque(maxlen=100)

    accel_queue = [accel_x, accel_y, accel_z]
    gyro_queue = [gyro_x, gyro_y, gyro_z]
    projected_gravity_queue = [projected_gravity_x, projected_gravity_y, projected_gravity_z]
    joint_position_queue = [
        joint_position_1, joint_position_2, joint_position_3,
        joint_position_4, joint_position_5, joint_position_6,
        joint_position_7, joint_position_8, joint_position_9,
        joint_position_10, joint_position_11, joint_position_12,
    ]
    joint_velocity_queue = [
        joint_velocity_1, joint_velocity_2, joint_velocity_3,
        joint_velocity_4, joint_velocity_5, joint_velocity_6,
        joint_velocity_7, joint_velocity_8, joint_velocity_9,
        joint_velocity_10, joint_velocity_11, joint_velocity_12,
    ]

    accel_filt_queue = [accel_x_filtered, accel_y_filtered, accel_z_filtered]
    gyro_filt_queue = [gyro_x_filtered, gyro_y_filtered, gyro_z_filtered]
    projected_gravity_filt_queue = [projected_gravity_x_filtered, projected_gravity_y_filtered, projected_gravity_z_filtered]
    joint_position_filt_queue = [
        joint_position_1_filtered, joint_position_2_filtered, joint_position_3_filtered,
        joint_position_4_filtered, joint_position_5_filtered, joint_position_6_filtered,
        joint_position_7_filtered, joint_position_8_filtered, joint_position_9_filtered,
        joint_position_10_filtered, joint_position_11_filtered, joint_position_12_filtered,
    ]
    joint_velocity_filt_queue = [
        joint_velocity_1_filtered, joint_velocity_2_filtered, joint_velocity_3_filtered,
        joint_velocity_4_filtered, joint_velocity_5_filtered, joint_velocity_6_filtered,
        joint_velocity_7_filtered, joint_velocity_8_filtered, joint_velocity_9_filtered,
        joint_velocity_10_filtered, joint_velocity_11_filtered, joint_velocity_12_filtered,
    ]

    next_time_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    while is_running:
        next_time_ns += control_rate_ns

        imu_state = unitree_driver.get_imu_state()
        motor_state = unitree_driver.get_motor_state()
        
        # Filter:
        lowpass_filter.add_data(imu_state, motor_state)
        filtered_data = lowpass_filter.apply_filter()

        # Calculate Projected Gravity:
        quaternion = np.asarray(imu_state.quaternion)
        normalized_quaternion = quaternion / np.linalg.norm(quaternion)
        projected_gravity = lowpass_filter.rotate(
            vec=np.array([0, 0, -1]),
            quat=lowpass_filter.quat_inv(normalized_quaternion)
        )

        data_mappings = [
            (accel_queue, imu_state.accelerometer),
            (gyro_queue, imu_state.gyroscope),
            (projected_gravity_queue, projected_gravity),
            (joint_position_queue, motor_state.q),
            (joint_velocity_queue, motor_state.qd),
            (accel_filt_queue, filtered_data.accelerometer),
            (gyro_filt_queue, filtered_data.gyroscope),
            (projected_gravity_filt_queue, filtered_data.projected_gravity),
            (joint_position_filt_queue, filtered_data.joint_position),
            (joint_velocity_filt_queue, filtered_data.joint_velocity)
        ]

        # Map Data to Queues:
        for queue_list, data_array in data_mappings:
            for queue, value in zip(queue_list, data_array):
                queue.append(value)

        # Plot Sensor vs Filtered Data:
        for queue in zip(accel_queue, accel_filt_queue):
            accel_plt.set_xdata(np.arange(len(queue[0])))
            accel_plt.set_ydata(queue[0])
            accel_filt_plt.set_xdata(np.arange(len(queue[1])))
            accel_filt_plt.set_ydata(queue[1])

        for queue in zip(gyro_queue, gyro_filt_queue):
            gyro_plt.set_xdata(np.arange(len(queue[0])))
            gyro_plt.set_ydata(queue[0])
            gyro_filt_plt.set_xdata(np.arange(len(queue[1])))
            gyro_filt_plt.set_ydata(queue[1])

        for queue in zip(projected_gravity_queue, projected_gravity_filt_queue):
            projected_gravity_plt.set_xdata(np.arange(len(queue[0])))
            projected_gravity_plt.set_ydata(queue[0])
            projected_gravity_filt_plt.set_xdata(np.arange(len(queue[1])))
            projected_gravity_filt_plt.set_ydata(queue[1])

        for queue in zip(joint_position_queue, joint_position_filt_queue):
            joint_position_plt.set_xdata(np.arange(len(queue[0])))
            joint_position_plt.set_ydata(queue[0])
            joint_position_filt_plt.set_xdata(np.arange(len(queue[1])))
            joint_position_filt_plt.set_ydata(queue[1])
        
        for queue in zip(joint_velocity_queue, joint_velocity_filt_queue):
            joint_velocity_plt.set_xdata(np.arange(len(queue[0])))
            joint_velocity_plt.set_ydata(queue[0])
            joint_velocity_filt_plt.set_xdata(np.arange(len(queue[1])))
            joint_velocity_filt_plt.set_ydata(queue[1])

        # Update Plots:
        for ax in axs:
            ax.relim()
            ax.autoscale_view()
            ax.legend()

        plt.draw()
        fig.canvas.draw()
        fig.canvas.flush_events()
        
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
