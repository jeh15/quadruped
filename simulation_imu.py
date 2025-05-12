import os
from absl import app
import pickle
import dataclasses as dataclass

import jax
import numpy as np
import numpy.typing as npt

import mujoco
from mujoco import mjx

from src.envs import unitree_go2_height_control as unitree_go2

jax.config.update("jax_enable_x64", True)


@dataclass.dataclass
class IMUState:
    accelerometer: npt.NDArray[np.float64]
    gyroscope: npt.NDArray[np.float64]
    quaternion: npt.NDArray[np.float64]


@dataclass.dataclass
class MotorState:
    joint_positions: npt.NDArray[np.float64]
    joint_velocities: npt.NDArray[np.float64]
    joint_torques: npt.NDArray[np.float64]


def main(argv=None):
    # Load Env for Sensor Data:
    env = unitree_go2.UnitreeGo2Env()

    # Load Model for Simulation:
    model = env.sys.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 3)
    mujoco.mj_forward(model, data)

    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)

    step_fn = jax.jit(mjx.step)

    control_rate = 0.02
    num_steps = int(control_rate / model.opt.timestep)

    imu_history = []
    motor_history = []

    for i in range(500):
        for _ in range(num_steps):
            mjx_data = step_fn(mjx_model, mjx_data)

        # Get IMU Data:
        imu_data = IMUState(
            accelerometer=env.get_accelerometer(mjx_data),
            gyroscope=env.get_gyro(mjx_data),
            quaternion=mjx_data.qpos[3:7],
        )
        imu_history.append(imu_data)

        # Get Motor Data:
        motor_data = MotorState(
            joint_positions=mjx_data.qpos[7:],
            joint_velocities=mjx_data.qvel[6:],
            joint_torques=mjx_data.actuator_force,
        )
        motor_history.append(motor_data)

    # Pickle Data:
    data_directory = os.path.join(
        os.path.dirname(__file__),
        'data',
    )
    os.makedirs(data_directory, exist_ok=True)
    imu_data_file = os.path.join(
        data_directory,
        'simulation_imu_data.pkl',
    )
    motor_data_file = os.path.join(
        data_directory,
        'simulation_motor_data.pkl',
    )

    with open(imu_data_file, 'wb') as f:
        pickle.dump(imu_history, f)

    with open(motor_data_file, 'wb') as f:
        pickle.dump(motor_history, f)


if __name__ == '__main__':
    app.run(main)
