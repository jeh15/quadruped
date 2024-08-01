from absl import app, flags
import os
from enum import Enum
import copy

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

from brax.io import html

from src.envs import unitree_go1
from src.algorithms.ppo.load_utilities import load_policy

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)


class Feet(Enum):
    front_left = 0
    hind_left = 1
    front_right = 2
    hind_right = 3


def main(argv=None):
    # Load from Env:
    env = unitree_go1.UnitreeGo1Env(kick_vel=0.0)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Load Policy:
    make_policy, params = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        restore_iteration=FLAGS.checkpoint_iteration,
        environment=env,
    )
    inference_function = make_policy(params)
    inference_fn = jax.jit(inference_function)

    # Initialize Simulation:
    key = jax.random.key(0)

    # Sweep through velocities:
    velocities = [1.0, 1.5, 2.0, 2.5, 3.0]
    for velocity in velocities:
        key, subkey = jax.random.split(key)
        state = reset_fn(subkey)
        state.info['command'] = jnp.array([velocity, 0.0, 0.0])

        num_steps = 1000
        steady_state_ratio = 0.8
        states = []
        contacts = []
        forward_velocity = []
        for i in range(num_steps):
            # Stop Command Sampling:
            state.info['command'] = jnp.array([velocity, 0.0, 0.0])
            key, subkey = jax.random.split(key)
            action, _ = inference_fn(state.obs, subkey)
            state = step_fn(state, action)
            states.append(state.pipeline_state)
            # Get Steady State:
            steady_state_condition = (
                (i > int((1.0-steady_state_ratio) * num_steps))
                & (i <= int(steady_state_ratio * num_steps))
            )
            if steady_state_condition:
                contacts.append(state.info['last_contact'])
                forward_velocity.append(state.pipeline_state.qd[0])

        contacts = np.asarray(contacts)
        forward_velocity = np.asarray(forward_velocity)
        print(f'Velocity: {np.mean(forward_velocity)}')

        gait = {}
        mode = {
            'timing': [],
            'stance': {
                'length': [],
                'average': 0,
                'indicies': [],
            },
            'flight': {
                'length': [],
                'average': 0,
                'indicies': [],
            },
        }

        for foot in Feet:
            gait.update({foot.name: copy.deepcopy(mode)})

        for foot in Feet:
            initial_mode = 'stance' if contacts[0, foot.value] else 'flight'
            previous_mode = initial_mode
            mode_length = 0
            for idx, contact in enumerate(contacts[1:, foot.value]):
                current_mode = 'stance' if contact else 'flight'
                if current_mode == previous_mode:
                    mode_length += 1
                else:
                    start_idx = idx - mode_length
                    gait[foot.name]['timing'].append(previous_mode)
                    gait[foot.name][previous_mode]['length'].append(
                        mode_length,
                    )
                    gait[foot.name][previous_mode]['indicies'].append(
                        start_idx,
                    )
                    mode_length = 0
                previous_mode = current_mode
            # Remove trivial first element and calculate average:
            gait[foot.name].update(timing=gait[foot.name]['timing'][1:])
            gait[foot.name][initial_mode].update(
                length=gait[foot.name][initial_mode]['length'][1:],
            )
            gait[foot.name][initial_mode].update(
                indicies=gait[foot.name][initial_mode]['indicies'][1:],
            )
            gait[foot.name]['stance'].update(
                average=np.mean(gait[foot.name]['stance']['length']),
            )
            gait[foot.name]['flight'].update(
                average=np.mean(gait[foot.name]['flight']['length']),
            )

        for foot in Feet:
            duty_cycle = gait[foot.name]["stance"]["average"] / (gait[foot.name]["stance"]["average"] + gait[foot.name]["flight"]["average"])
            print(
                f'Average Stance Length ({foot.name}): {gait[foot.name]["stance"]["average"]:.3f} \t Average Flight Length ({foot.name}): {gait[foot.name]["flight"]["average"]:.3f} \t Duty Cycle ({foot.name}): {duty_cycle:.3f}',
            )
        print('\n')

        # Visualize:
        fig, axs = plt.subplots(1, 1)
        start_step = 5
        end_step = 15
        for foot in Feet:
            ys = [1, 1.1, 1.2, 1.3]
            colors = ['b', 'g', 'r', 'c']
            for stance_idx, stance_length in zip(
                gait[foot.name]['stance']['indicies'][start_step:end_step],
                gait[foot.name]['stance']['length'][start_step:end_step]
            ):
                axs.hlines(
                    y=ys[foot.value],
                    xmin=stance_idx*env.step_dt,
                    xmax=(stance_idx+stance_length)*env.step_dt,
                    color=colors[foot.value],
                    linestyle='solid',
                    linewidth=4,
                )

        axs.set_yticks(ys)
        axs.set_yticklabels([foot.name for foot in Feet])
        axs.set_xlabel('Time (s)')
        axs.set_ylim(0.9, 1.4)
        fig.suptitle(f'Gait Visualization at {np.mean(forward_velocity):.3f} m/s')

        plt.savefig(f'gait_velocity_{velocity}.png')

        # Generate HTML:
        html_string = html.render(
            sys=env.sys.tree_replace({'opt.timestep': env.step_dt}),
            states=states,
            height="100vh",
            colab=False,
        )

        html_path = os.path.join(
            os.path.dirname(__file__),
            f"visualization/visualization_{velocity}.html",
        )

        with open(html_path, "w") as f:
            f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)
