from absl import app, flags
import os
import dataclasses
import pickle

import numpy as np

import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'filename', None, 'Filename to pickle relative to the data directory.', short_name='f',
)

def main(argv=None):
    if FLAGS.filename is None:
        raise ValueError('Please provide a filename for the pickle file.')
    
    directory = os.path.join(
        os.path.dirname(__file__),
        'data',
    )
    filepath = os.path.join(
        directory,
        f'{FLAGS.filename}.pkl',
    )
    with open(filepath, 'rb') as file:
        data = pickle.load(file)

    # Extract the data
    data = np.asarray(data)

    accelerometer = data[:, 0]
    gyroscope = data[:, 1]
    quaternion = data[:, 2]
    rpy = data[:, 3]

    # Plot and compare Accelerometer Data:
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(accelerometer[:, 0], label='Accelerometer X')
    axs[0].set_title('Accelerometer X Data')
    axs[0].set_ylabel('Acceleration (m/s^2)')
    axs[0].legend()

    axs[1].plot(accelerometer[:, 1], label='Accelerometer Y')
    axs[1].set_title('Accelerometer Y Data')
    axs[1].set_ylabel('Acceleration (m/s^2)')
    axs[1].legend()

    axs[2].plot(accelerometer[:, 2], label='Accelerometer Z')
    axs[2].set_title('Accelerometer Z Data')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Acceleration (m/s^2)')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f'data/accelerometer_{FLAGS.filename}.pdf')

    # Plot and compare Gyroscope Data:
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(gyroscope[:, 0], label='Gyroscope X')
    axs[0].set_title('Gyroscope X Data')
    axs[0].set_ylabel('Angular Velocity (rad/s)')
    axs[0].legend()

    axs[1].plot(gyroscope[:, 1], label='Gyroscope Y')
    axs[1].set_title('Gyroscope Y Data')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[1].legend()

    axs[2].plot(gyroscope[:, 2], label='Gyroscope Z')
    axs[2].set_title('Gyroscope Z Data')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Angular Velocity (rad/s)')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f'data/gyroscope_{FLAGS.filename}.pdf')

    # Plot and compare Quaternion Data:
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    axs[0].plot(quaternion[:, 0], label='Quaternion W')
    axs[0].set_title('Quaternion W Data')
    axs[0].set_ylabel('Quaternion')
    axs[0].legend()

    axs[1].plot(quaternion[:, 1], label='Quaternion X')
    axs[1].set_title('Quaternion X Data')
    axs[1].set_ylabel('Quaternion')
    axs[1].legend()

    axs[2].plot(quaternion[:, 2], label='Quaternion Y')
    axs[2].plot(hardware_quaternion[:, 2], label='Hardware Quaternion Y')
    axs[2].set_title('Quaternion Y Data')
    axs[2].set_ylabel('Quaternion')
    axs[2].legend()

    axs[3].plot(quaternion[:, 3], label='Quaternion Z')
    axs[3].set_title('Quaternion Z Data')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Quaternion')
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(f'data/quaternion_{FLAGS.filename}.pdf')

    # Plot and compare RPY Data:
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(rpy[:, 0], label='RPY Roll')
    axs[0].set_title('RPY Roll Data')
    axs[0].set_ylabel('Roll (rad)')
    axs[0].legend()
    
    axs[1].plot(rpy[:, 1], label='RPY Pitch')
    axs[1].set_title('RPY Pitch Data')
    axs[1].set_ylabel('Pitch (rad)')
    axs[1].legend()

    axs[2].plot(rpy[:, 2], label='RPY Yaw')
    axs[2].set_title('RPY Yaw Data')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Yaw (rad)')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f'data/rpy_{FLAGS.flag}.pdf')


if __name__ == "__main__":
    app.run(main)
