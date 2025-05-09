from absl import app
import os
import pickle

import jax
import numpy as np

import flax.struct

from brax.io import mjcf
from brax.mjx import pipeline

import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)


@flax.struct.dataclass
class minibatch:
    q: jax.Array
    qd: jax.Array
    ctrl: jax.Array


def main(argv=None):
    filename = 'models/unitree_go2/go2_regression_model.xml'
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
        'data/unitree_data_1.pkl',
    )
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    data = np.asarray(data)

    # Get the number of time steps and trials
    num_motors = 12
    num_dof = 3
    num_trials, num_time_steps, _ = data.shape

    # Parse Data:
    q_measured = data[:, :, :12]
    qd_measured = data[:, :, 12:24]
    torque_measured = data[:, :, 24:36]
    setpoints = data[:, :, 36:]

    # Concatenate Legs into different trials:
    process_fn = lambda x: np.concatenate(
            np.split(
                np.reshape(x, (num_trials, num_time_steps, 4, 3)),
                indices_or_sections=4,
                axis=2,
            ),
            axis=0,
        ).squeeze()

    q_measured = process_fn(q_measured)
    qd_measured = process_fn(qd_measured)
    torque_measured = process_fn(torque_measured)
    setpoints = process_fn(setpoints)

    # # Load Regression Data:
    # regression_filepath = os.path.join(
    #     os.path.dirname(__file__),
    #     'data/param_regression_history_kp.pkl',
    # )

    # with open(regression_filepath, 'rb') as file:
    #     regression_data = pickle.load(file)

    # # dof_damping = np.concatenate(regression_data['dof_damping'])[-1]
    # kp = np.concatenate(regression_data['kp'])[-1]

    # # dof_damping_history = np.concatenate(regression_data['dof_damping'])
    # kp_history = np.concatenate(regression_data['kp'])

    # # Load Loss Data:
    # loss_filepath = os.path.join(
    #     os.path.dirname(__file__),
    #     'data/loss_history_kp.pkl',
    # )
    
    # with open(loss_filepath, 'rb') as file:
    #     loss_data = pickle.load(file)

    # Run initial comparison of random trial:
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    random_idx = jax.random.randint(subkey, (), minval=0, maxval=4 * num_trials-1)
    q_initial_trial = q_measured[random_idx, :]
    qd_initial_trial = qd_measured[random_idx, :]
    ctrl_initial_trial = setpoints[random_idx, :]
    state = jax.jit(pipeline.init)(sys, q_initial_trial[0], qd_initial_trial[0])
    q_history_init = []
    qd_history_init = []
    torque_history_init = []
    prior_state_history = []
    for ctrl in ctrl_initial_trial:
        for _ in range(control_steps):
            state = jax.jit(pipeline.step)(sys, state, ctrl)
        prior_state_history.append(state)
        q_history_init.append(state.q)
        qd_history_init.append(state.qd)
        torque_history_init.append(state.actuator_force)


    # Run random trial with regressed params:
    # gain = sys.actuator_gainprm.at[:, 0].set(kp)
    # bias = sys.actuator_biasprm.at[:, 1].set(-kp)
    # sys = sys.replace(
    #     # dof_damping=dof_damping,
    #     actuator_gainprm=gain,
    #     actuator_biasprm=bias,
    # )
    # q_test = q_measured[random_idx, :]
    # qd_test = qd_measured[random_idx, :]
    # ctrl_test = setpoints[random_idx, :]
    # state = jax.jit(pipeline.init)(sys, q_test[0], qd_test[0])
    # q_history = []
    # qd_history = []
    # torque_history = []
    # posterior_state_history = []
    # for ctrl in ctrl_test:
    #     for _ in range(control_steps):
    #         state = jax.jit(pipeline.step)(sys, state, ctrl)
    #     posterior_state_history.append(state)
    #     q_history.append(state.q)
    #     qd_history.append(state.qd)
    #     torque_history.append(state.actuator_force)

    # q_history = np.asarray(q_history)
    # qd_history = np.asarray(qd_history)
    # torque_history = np.asarray(torque_history)
    q_history_init = np.asarray(q_history_init)
    qd_history_init = np.asarray(qd_history_init)
    torque_history_init = np.asarray(torque_history_init)

    # fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(10, 5))
    # # ax[0].plot(dof_damping_history[:, 0], color='orange', label='Abduction', linewidth=1.0)
    # # ax[0].plot(dof_damping_history[:, 1], color='cornflowerblue', label='Hip', linewidth=1.0)
    # # ax[0].plot(dof_damping_history[:, 2], color='lightcoral', label='Knee', linewidth=1.0)
    # ax[0].legend()
    # ax[0].set_title('Damping Regression')
    # ax[0].set_xlabel('Iterations')
    # ax[0].set_ylabel('Damping Parameter')

    # ax[1].plot(kp_history [:, 0], color='orange', label='Abduction', linewidth=1.0)
    # ax[1].plot(kp_history[:, 1], color='cornflowerblue', label='Hip', linewidth=1.0)
    # ax[1].plot(kp_history[:, 2], color='lightcoral', label='Knee', linewidth=1.0)
    # ax[1].legend()
    # ax[1].set_title('Motor Gain Regression')
    # ax[1].set_xlabel('Iterations')
    # ax[1].set_ylabel('Motor Gain Parameter')

    # ax[-1].plot(loss_data, linewidth=1.0)
    # ax[-1].set_title('Loss')
    # ax[-1].set_xlabel('Iterations')
    # ax[-1].set_ylabel('Loss')

    # plt.savefig('data/regress_params.pdf')

    # fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 5))
    # fig.suptitle('Regressed Comparison')

    # ax[0].plot(q_measured[random_idx, :, 0], color='orange', linestyle='--', linewidth=3.0)
    # ax[0].plot(q_measured[random_idx, :, 1], color='cornflowerblue', linestyle='--', linewidth=3.0)
    # ax[0].plot(q_measured[random_idx, :, 2],  color='lightcoral', linestyle='--', linewidth=3.0)
    # ax[0].plot(q_history[:, 0], color='orange', label='Abduction', linewidth=3.0)
    # ax[0].plot(q_history[:, 1], color='cornflowerblue', label='Hip', linewidth=3.0)
    # ax[0].plot(q_history[:, 2], color='lightcoral', label='Knee', linewidth=3.0)
    # ax[0].legend()
    # ax[0].set_title('Position')
    # ax[0].set_ylabel('Position')

    # ax[1].plot(qd_measured[random_idx, :, 0], color='orange', linestyle='--', linewidth=3.0)
    # ax[1].plot(qd_measured[random_idx, :, 1], color='cornflowerblue', linestyle='--', linewidth=3.0)
    # ax[1].plot(qd_measured[random_idx, :, 2], color='lightcoral', linestyle='--', linewidth=3.0)
    # ax[1].plot(qd_history[:, 0], color='orange', label='Abduction', linewidth=3.0)
    # ax[1].plot(qd_history[:, 1], color='cornflowerblue', label='Hip', linewidth=3.0)
    # ax[1].plot(qd_history[:, 2], color='lightcoral', label='Knee', linewidth=3.0)
    # ax[1].legend()
    # ax[1].set_title('Velocity')
    # ax[1].set_xlabel('Time')
    # ax[1].set_ylabel('Velocity')

    # plt.savefig('data/regressed.pdf')

    # fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 5))
    # fig.suptitle('Initial Comparison')

    # ax[0].plot(q_measured[random_idx, :, 0], color='orange', linestyle='--', linewidth=3.0)
    # ax[0].plot(q_measured[random_idx, :, 1], color='cornflowerblue', linestyle='--', linewidth=3.0)
    # ax[0].plot(q_measured[random_idx, :, 2],  color='lightcoral', linestyle='--', linewidth=3.0)
    # ax[0].plot(q_history_init[:, 0], color='orange', label='Abduction', linewidth=3.0)
    # ax[0].plot(q_history_init[:, 1], color='cornflowerblue', label='Hip', linewidth=3.0)
    # ax[0].plot(q_history_init[:, 2], color='lightcoral', label='Knee', linewidth=3.0)
    # ax[0].legend()
    # ax[0].set_title('Position')
    # ax[0].set_ylabel('Position')

    # ax[1].plot(qd_measured[random_idx, :, 0], color='orange', linestyle='--', linewidth=3.0)
    # ax[1].plot(qd_measured[random_idx, :, 1], color='cornflowerblue', linestyle='--', linewidth=3.0)
    # ax[1].plot(qd_measured[random_idx, :, 2], color='lightcoral', linestyle='--', linewidth=3.0)
    # ax[1].plot(qd_history_init[:, 0], color='orange', label='Abduction', linewidth=3.0)
    # ax[1].plot(qd_history_init[:, 1], color='cornflowerblue', label='Hip', linewidth=3.0)
    # ax[1].plot(qd_history_init[:, 2], color='lightcoral', label='Knee', linewidth=3.0)
    # ax[1].legend()
    # ax[1].set_title('Velocity')
    # ax[1].set_xlabel('Time')
    # ax[1].set_ylabel('Velocity')

    # plt.savefig('data/initial.pdf')

    fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(10, 5))
    ax[0].plot(q_measured[random_idx, :, 0], color='orange', label='Data', linewidth=1.0)
    ax[0].plot(q_history_init[:, 0], color='cornflowerblue', label='Simulation', linewidth=1.0)
    ax[0].set_title('Abduction')
    ax[0].set_ylabel('Position (rad)')

    ax[1].plot(q_measured[random_idx, :, 1], color='orange', label='Data', linewidth=1.0)
    ax[1].plot(q_history_init[:, 1], color='cornflowerblue', label='Simulation', linewidth=1.0)
    ax[1].legend()
    ax[1].set_title('Hip')
    ax[1].set_ylabel('Position (rad)')

    ax[2].plot(q_measured[random_idx, :, 2], color='orange', label='Data', linewidth=1.0)
    ax[2].plot(q_history_init[:, 2], color='cornflowerblue', label='Simulation', linewidth=1.0)
    ax[2].set_title('Knee')
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel('Position (rad)')

    plt.savefig('data/initial_joint_position_comparison.pdf')

    fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(10, 5))
    ax[0].plot(qd_measured[random_idx, :, 0], color='orange', label='Data', linewidth=1.0)
    ax[0].plot(qd_history_init[:, 0], color='cornflowerblue', label='Simulation', linewidth=1.0)
    ax[0].set_title('Abduction')
    ax[0].set_ylabel('Velocity (rad/s)')

    ax[1].plot(qd_measured[random_idx, :, 1], color='orange', label='Data', linewidth=1.0)
    ax[1].plot(qd_history_init[:, 1], color='cornflowerblue', label='Simulation', linewidth=1.0)
    ax[1].legend()
    ax[1].set_title('Hip')
    ax[1].set_ylabel('Velocity (rad/s)')

    ax[2].plot(qd_measured[random_idx, :, 2], color='orange', label='Data', linewidth=1.0)
    ax[2].plot(qd_history_init[:, 2], color='cornflowerblue', label='Simulation', linewidth=1.0)
    ax[2].set_title('Knee')
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel('Velocity (rad/s)')

    plt.savefig('data/initial_joint_velocity_comparison.pdf')

    # fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(10, 5))
    # ax[0].plot(q_measured[random_idx, :, 0], color='orange', label='Data', linewidth=1.0)
    # ax[0].plot(q_history[:, 0], color='cornflowerblue', label='Simulation', linewidth=1.0)
    # ax[0].set_title('Abduction')
    # ax[0].set_ylabel('Position (rad)')

    # ax[1].plot(q_measured[random_idx, :, 1], color='orange', label='Data', linewidth=1.0)
    # ax[1].plot(q_history[:, 1], color='cornflowerblue', label='Simulation', linewidth=1.0)
    # ax[1].legend()
    # ax[1].set_title('Hip')
    # ax[1].set_ylabel('Position (rad)')

    # ax[2].plot(q_measured[random_idx, :, 2], color='orange', label='Data', linewidth=1.0)
    # ax[2].plot(q_history[:, 2], color='cornflowerblue', label='Simulation', linewidth=1.0)
    # ax[2].set_title('Knee')
    # ax[2].set_xlabel('Step')
    # ax[2].set_ylabel('Position (rad)')

    # plt.savefig('data/regressed_joint_position_comparison.pdf')

    # fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(10, 5))
    # ax[0].plot(qd_measured[random_idx, :, 0], color='orange', label='Data', linewidth=1.0)
    # ax[0].plot(qd_history[:, 0], color='cornflowerblue', label='Simulation', linewidth=1.0)
    # ax[0].set_title('Abduction')
    # ax[0].set_ylabel('Velocity (rad/s)')

    # ax[1].plot(qd_measured[random_idx, :, 1], color='orange', label='Data', linewidth=1.0)
    # ax[1].plot(qd_history[:, 1], color='cornflowerblue', label='Simulation', linewidth=1.0)
    # ax[1].legend()
    # ax[1].set_title('Hip')
    # ax[1].set_ylabel('Velocity (rad/s)')

    # ax[2].plot(qd_measured[random_idx, :, 2], color='orange', label='Data', linewidth=1.0)
    # ax[2].plot(qd_history[:, 2], color='cornflowerblue', label='Simulation', linewidth=1.0)
    # ax[2].set_title('Knee')
    # ax[2].set_xlabel('Step')
    # ax[2].set_ylabel('Velocity (rad/s)')

    # plt.savefig('data/regressed_joint_velocity_comparison.pdf')

    fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(10, 5))
    ax[0].plot(torque_measured[random_idx, :, 0], color='orange', label='Data', linewidth=1.0)
    ax[0].plot(torque_history_init[:, 0], color='cornflowerblue', label='Simulation', linewidth=1.0)
    ax[0].set_title('Abduction')
    ax[0].set_ylabel('Torque (N*m)')

    ax[1].plot(torque_measured[random_idx, :, 1], color='orange', label='Data', linewidth=1.0)
    ax[1].plot(torque_history_init[:, 1], color='cornflowerblue', label='Simulation', linewidth=1.0)
    ax[1].legend()
    ax[1].set_title('Hip')
    ax[1].set_ylabel('Torque (N*m)')

    ax[2].plot(torque_measured[random_idx, :, 2], color='orange', label='Data', linewidth=1.0)
    ax[2].plot(torque_history_init[:, 2], color='cornflowerblue', label='Simulation', linewidth=1.0)
    ax[2].set_title('Knee')
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel('Torque (N*m)')

    plt.savefig('data/initial_joint_torque_comparison.pdf')

    # fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(10, 5))
    # ax[0].plot(torque_measured[random_idx, :, 0], color='orange', label='Data', linewidth=1.0)
    # ax[0].plot(torque_history[:, 0], color='cornflowerblue', label='Simulation', linewidth=1.0)
    # ax[0].set_title('Abduction')
    # ax[0].set_ylabel('Torque (N*m)')

    # ax[1].plot(torque_measured[random_idx, :, 1], color='orange', label='Data', linewidth=1.0)
    # ax[1].plot(torque_history[:, 1], color='cornflowerblue', label='Simulation', linewidth=1.0)
    # ax[1].legend()
    # ax[1].set_title('Hip')
    # ax[1].set_ylabel('Torque (N*m)')

    # ax[2].plot(torque_measured[random_idx, :, 2], color='orange', label='Data', linewidth=1.0)
    # ax[2].plot(torque_history[:, 2], color='cornflowerblue', label='Simulation', linewidth=1.0)
    # ax[2].set_title('Knee')
    # ax[2].set_xlabel('Step')
    # ax[2].set_ylabel('Torque (N*m)')

    # plt.savefig('data/regressed_joint_torque_comparison.pdf')


if __name__ == '__main__':
    app.run(main)
