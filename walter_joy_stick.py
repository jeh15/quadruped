from absl import app, flags
import functools
import time

import pygame

import jax
import numpy as np

import mujoco
import mujoco.viewer

from src.envs import walter
from src.load_utilities import load_policy

jax.config.update("jax_enable_x64", True)
pygame.init()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)


def controller(policy_action, limb_ids, limb_idx, wheel_idx, action_scale, torque_scale, default_pose):
    lb = np.array([-np.pi / 4, -np.pi / 4, -3.0, -3.0] * 4)
    ub = np.array([np.pi / 4, np.pi / 4, 3.0, 3.0] * 4)
    position_targets = default_pose[limb_ids-2] + policy_action[limb_idx] * action_scale
    torque_targets = policy_action[wheel_idx] * torque_scale
    motor_targets = np.array([
        position_targets[0], position_targets[1], torque_targets[0], torque_targets[1],
        position_targets[2], position_targets[3], torque_targets[2], torque_targets[3],
        position_targets[4], position_targets[5], torque_targets[4], torque_targets[5],
        position_targets[6], position_targets[7], torque_targets[6], torque_targets[7],
    ])
    motor_targets = np.clip(motor_targets, lb, ub)
    return motor_targets


def main(argv=None):
    # Load from Env:
    env = walter.WalterEnv()
    model = env.sys.mj_model

    data = mujoco.MjData(model)  # type: ignore
    mujoco.mj_resetData(model, data)  # type: ignore
    control_rate = 0.02
    num_steps = int(control_rate / model.opt.timestep)

    # Load Policy:
    make_policy, params = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
    )
    inference_function = make_policy(params)
    inference_fn = jax.jit(inference_function)

    # Controller:
    controller_fn = functools.partial(
        controller,
        limb_ids=np.asarray(env.limb_ids),
        limb_idx=np.asarray(env.limb_idx),
        wheel_idx=np.asarray(env.wheel_idx),
        action_scale=env._action_scale,
        torque_scale=env._torque_scale,
        default_pose=env.default_pose,
    )

    # Test:
    data.qpos = model.key_qpos.flatten()
    command = np.array([0.0, 0.0])
    action = model.key_ctrl.flatten()
    observation = np.zeros(env.history_length * env.num_observations)

    # Setup Joystick:
    joysticks = {}

    key = jax.random.key(0)
    termination_flag = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not termination_flag:

            # Move all of this to an event handler:
            for event in pygame.event.get():
                if event.type == pygame.JOYDEVICEADDED:
                    joy = pygame.joystick.Joystick(event.device_index)
                    joysticks[joy.get_instance_id()] = joy
                    print(f"Joystick {joy.get_instance_id()} connencted")

                if event.type == pygame.JOYDEVICEREMOVED:
                    del joysticks[event.instance_id]
                    print(f"Joystick {event.instance_id} disconnected")

            for joystick in joysticks.values():
                forward_command = -1 * joystick.get_axis(1)
                rotation_command = -1 * joystick.get_axis(2)

                # Filter and Clip Command:
                command = np.array([
                    forward_command, rotation_command,
                ])
                command = np.where(np.abs(command) < 0.1, 0.0, command)
                command = np.clip(command, -0.75, 0.75)

            step_time = time.time()
            action_rng, key = jax.random.split(key)
            observation = env.numpy_observation(
                mj_data=data,
                command=command,
                previous_action=action,
                observation_history=observation,
            )
            action, _ = inference_fn(observation, action_rng)
            ctrl = controller_fn(action)
            data.ctrl = ctrl

            for _ in range(num_steps):
                mujoco.mj_step(model, data)  # type: ignore

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)  # type: ignore

            viewer.sync()

            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == '__main__':
    app.run(main)