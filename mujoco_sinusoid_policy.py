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

from src.envs import sinusoid_test as unitree_go2
from src.algorithms.ppo.load_utilities import load_policy


jax.config.update("jax_enable_x64", True)
pygame.init()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)
    

def controller(
    action: npt.ArrayLike,
    default_control: npt.ArrayLike,
    ctrl_lb: npt.ArrayLike,
    ctrl_ub: npt.ArrayLike,
    action_scale: float,
) -> np.ndarray:
    motor_targets = default_control + action * action_scale
    # motor_targets = np.clip(motor_targets, ctrl_lb, ctrl_ub)
    return motor_targets


def main(argv=None):
    # Set up Logger:
    logging.use_absl_handler()
    log_directory = os.path.join(
        os.path.dirname(__file__),
        'logs',
    )
    logging.get_absl_handler().use_absl_log_file(program_name='simulation_test', log_dir=log_directory) 
    logging.set_verbosity(logging.INFO)

    # Load from Env:
    env = unitree_go2.UnitreeGo2Env(filename='unitree_go2/scene_mjx_fixed.xml')
    model_mjx = env.sys.mj_model
    
    # High Fidelity Model:
    model_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx_fixed.xml',
    )

    model = mujoco.MjModel.from_xml_path(
        model_path,
    )
    model.opt.timestep = 0.004

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
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
        action_scale=env.action_scale,
    )

    # Test:
    command = np.array([0.0])
    action = model_mjx.key_ctrl.flatten()
    observation = np.zeros(env.num_observations)

    # Setup Joystick:
    joysticks = {}

    key = jax.random.key(0)
    termination_flag = False

    global_steps = 0

    joint_position_history = []
    joint_velocity_history = []
    action_history = []
    ctrl_history = []

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
                if joystick.get_button(11) == 1:
                    termination_flag = True
                command = -1 * joystick.get_axis(1)



            # Filter and Clip Command:
            command = np.array([
                command
            ])
            command = np.where(np.abs(command) < 0.1, 0.0, command)
            command = np.clip(command, -1.0, 1.0)

            if(command < 0):
                command = -0.1 * command
            else:
                command = 0.1 * command

            command = np.clip(command, -0.1, 0.1)

            # Print Tracking Reward:
            desired_foot_height = env.default_feet_position[:, -1] + command
            foot_position = env.get_feet_pos(data)
            error = np.sum(np.square(desired_foot_height - foot_position[:, -1]))
            tracking_reward = np.exp(-error / 0.01)

            print(f'Command: {command}')
            print(f'Desired Height: {desired_foot_height}')
            print(f'Actual Height: {foot_position[:, -1]}')
            print(f'Tracking Reward: {tracking_reward}')

            step_time = time.time()
            action_rng, key = jax.random.split(key)

            # Get Observation:
            observation = env.np_observation(
                mj_data=data,
                command=command,
                previous_action=action,
                add_noise=False,
            )
            action, _ = jax.block_until_ready(
                inference_fn(observation, action_rng),
            )
            ctrl = controller_fn(action)

            # data.ctrl = ctrl
            data.ctrl = ctrl

            # Log MuJoCo Data:
            joint_position = data.qpos
            joint_velocity = data.qvel

            # Append Data:
            joint_position_history.append(joint_position)
            joint_velocity_history.append(joint_velocity)
            action_history.append(action)
            ctrl_history.append(ctrl)

            for _ in range(num_steps):
                mujoco.mj_step(model, data)  # type: ignore

            viewer.sync()

            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            global_steps += 1


    # Save Data:
    joint_position_data = np.asarray(joint_position_history)
    joint_velocity_data = np.asarray(joint_velocity_history)
    action_data = np.asarray(action_history)
    ctrl_data = np.asarray(ctrl_history)

    np.savetxt(
        os.path.join(log_directory, 'simulation_joint_position_data.txt'),
        joint_position_data,
        delimiter=',',
    )
    np.savetxt(
        os.path.join(log_directory, 'simulation_joint_velocity_data.txt'),
        joint_velocity_data,
        delimiter=',',
    )
    np.savetxt(
        os.path.join(log_directory, 'simulation_action_data.txt'),
        action_data,
        delimiter=',',
    )
    np.savetxt(
        os.path.join(log_directory, 'simulation_ctrl_data.txt'),
        ctrl_data,
        delimiter=',',
    )


if __name__ == '__main__':
    app.run(main)
