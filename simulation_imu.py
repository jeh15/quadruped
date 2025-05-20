from absl import app
import os

import time
import pickle

import jax
import numpy as np

import mujoco
import mujoco.viewer

from src.envs import unitree_go2_height_control_v4 as unitree_go2

jax.config.update("jax_enable_x64", True)


def main(argv=None):
    # Load Env for Sensor Data:
    env = unitree_go2.UnitreeGo2Env()

    # Load Model for Simulation:
    model = env.sys.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 3)
    mujoco.mj_forward(model, data)

    model_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx_rsl_rl.xml',
    )

    model = mujoco.MjModel.from_xml_path(
        model_path,
    )
    model.opt.timestep = 0.004

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    data.qpos[2] = 0.5

    control_rate = 0.02
    num_steps = int(control_rate / model.opt.timestep)

    imu_history = []
    termination_flag = False
    rotations = np.array([
        [1, 0, -1, 0],
        [1, 0, 1, 0],
        [1, -1, 0, 0],
        [0, 1, 0, 0],
    ])
    i = 0

    names = ['upright', 'down', 'left', 'flipped']

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5
        while viewer.is_running() and not termination_flag:
            if i > 3:
                termination_flag = True
                break

            # Iterate Through Rotations:
            imu_states = []
            rotation = rotations[i]
            data.qpos[2] = 0.5
            data.qpos[3:7] = rotation

            # Forward Kinematics:
            mujoco.mj_forward(model, data)

            is_running = True
            while is_running:
                
                accelerometer = env.get_accelerometer(data)
                gyroscope = env.get_gyro(data)
                gravity = env.get_gravity(data)

                imu_state = np.concatenate(
                    [accelerometer, gyroscope, gravity],
                    axis=0,
                )
                imu_states.append(imu_state)

                data.qpos[2] = 0.5
                data.qpos[3:7] = rotation

                mujoco.mj_step(model, data)

                viewer.sync()
                
                if data.time >= 10.0:
                    data.time = 0.0
                    i += 1
                    is_running = False

            imu_states = np.array(imu_states)
            # Pickle Data:
            data_directory = os.path.join(
                os.path.dirname(__file__),
                'data',
            )
            os.makedirs(data_directory, exist_ok=True)
            imu_data_file = os.path.join(
                data_directory,
                f'simulation_{names[i-1]}.pkl',
            )

            with open(imu_data_file, 'wb') as f:
                pickle.dump(imu_states, f)


if __name__ == '__main__':
    app.run(main)
