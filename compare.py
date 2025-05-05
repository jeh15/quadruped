from absl import app
import os
import pickle

import jax
import numpy as np

from brax.io import mjcf
from brax.mjx import pipeline

import matplotlib.pyplot as plt


jax.config.update('jax_enable_x64', True)


def main(argv=None):
    filename = 'models/unitree_go2/scene_mjx_fixed.xml'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    sys = mjcf.load(filepath)
    sys = sys.tree_replace({'opt.timestep': 0.004})

    control_rate = 0.02
    control_steps = int(control_rate / sys.opt.timestep)

    file_path = os.path.join(
        os.path.dirname(__file__),
        'data/unitree_data.pkl',
    )
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    data = np.asarray(data)
    
    q_measured = data[:, :, :12]
    qd_measured = data[:, :, 12:24]
    torque_measured = data[:, :, 24:36]
    setpoints = data[:, :, 36:]

    # Get the number of time steps and trials
    num_motors = 12
    num_trials, num_time_steps, _ = data.shape

    # Shuffle Data:
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    random_idx = jax.random.randint(subkey, (), minval=0, maxval=num_trials-1)

    q_initial_trial = q_measured[random_idx]
    qd_initial_trial = qd_measured[random_idx]
    ctrl_initial_trial = setpoints[random_idx]
    state = jax.jit(pipeline.init)(sys, q_initial_trial[0], qd_initial_trial[0], ctrl_initial_trial[0])

    q_history = [q_initial_trial[0]]
    qd_history = [qd_initial_trial[0]]
    prior_state_history = [state]

    for i, setpoint in enumerate(ctrl_initial_trial):
        for _ in range(control_steps):
            state = jax.jit(pipeline.step)(sys, state, setpoint)
        prior_state_history.append(state)
        q_history.append(state.q)
        qd_history.append(state.qd)

    q_history = np.asarray(q_history)
    qd_history = np.asarray(qd_history)

    fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(10, 5))
    fig.suptitle('Initial Joint Position Comparison')
    ax[0].plot(q_measured[random_idx, :, 0], color='orange', linestyle='--', linewidth=1.0, label='Data')
    ax[1].plot(q_measured[random_idx, :, 1], color='cornflowerblue', linestyle='--', linewidth=1.0, label='Data')
    ax[2].plot(q_measured[random_idx, :, 2],  color='lightcoral', linestyle='--', linewidth=1.0, label='Data')
    ax[0].plot(q_history[:, 0], color='orange', linewidth=1.0, label='Simulation')
    ax[1].plot(q_history[:, 1], color='cornflowerblue', linewidth=1.0, label='Simulation')
    ax[2].plot(q_history[:, 2], color='lightcoral', linewidth=1.0, label='Simulation')
    ax[0].legend()
    ax[0].set_title('Abduction')
    ax[1].set_title('Hip')
    ax[2].set_title('Knee')
    ax[0].set_ylabel('Position (rad)')
    ax[1].set_ylabel('Position (rad)')
    ax[2].set_ylabel('Position (rad)')

    plt.savefig('initial_joint_position_comparison.pdf')

    fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(10, 5))
    fig.suptitle('Initial Joint Velocity Comparison')
    ax[0].plot(qd_measured[random_idx, :, 0], color='orange', linestyle='--', linewidth=1.0, label='Data')
    ax[1].plot(qd_measured[random_idx, :, 1], color='cornflowerblue', linestyle='--', linewidth=1.0, label='Data')
    ax[2].plot(qd_measured[random_idx, :, 2],  color='lightcoral', linestyle='--', linewidth=1.0, label='Data')
    ax[0].plot(qd_history[:, 0], color='orange', linewidth=1.0, label='Simulation')
    ax[1].plot(qd_history[:, 1], color='cornflowerblue', linewidth=1.0, label='Simulation')
    ax[2].plot(qd_history[:, 2], color='lightcoral', linewidth=1.0, label='Simulation')
    ax[0].legend()
    ax[0].set_title('Abduction')
    ax[1].set_title('Hip')
    ax[2].set_title('Knee')
    ax[0].set_ylabel('Velocity (rad/s)')
    ax[1].set_ylabel('Velocity (rad/s)')
    ax[2].set_ylabel('Velocity (rad/s)')

    plt.savefig('initial_joint_velocity_comparison.pdf')

if __name__ == '__main__':
    app.run(main)