from absl import app

import pickle

import numpy as np
import matplotlib.pyplot as plt


def main(argv=None):
    # Load the data:
    with open('params_False.pkl', 'rb') as f:
        states_false = pickle.load(f)

    with open('params_True.pkl', 'rb') as f:
        states_true = pickle.load(f)

    # Extract the q and qd values:
    q_false = list(map(lambda x: x.q, states_false))
    q_true = list(map(lambda x: x.q, states_true))
    qd_false = list(map(lambda x: x.qd, states_false))
    qd_true = list(map(lambda x: x.qd, states_true))

    # Convert to numpy arrays:
    q_false = np.asarray(q_false)
    q_true = np.asarray(q_true)
    qd_false = np.asarray(qd_false)
    qd_true = np.asarray(qd_true)

    # Plot the data:
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(q_false[:, 0], label='Abduction', color='red')
    ax[0].plot(q_false[:, 1], label='Hip', color='red')
    ax[0].plot(q_false[:, 2], label='Knee', color='red')
    ax[0].plot(q_true[:, 0], label='Abduction', color='blue')
    ax[0].plot(q_true[:, 1], label='Hip', color='blue')
    ax[0].plot(q_true[:, 2], label='Knee', color='blue')
    ax[0].set_title('Positions')
    ax[0].legend()

    ax[1].plot(qd_false[:, 0], label='Abduction', color='red')
    ax[1].plot(qd_false[:, 1], label='Hip', color='red')
    ax[1].plot(qd_false[:, 2], label='Knee', color='red')
    ax[1].plot(qd_true[:, 0], label='Abduction', color='blue')
    ax[1].plot(qd_true[:, 1], label='Hip', color='blue')
    ax[1].plot(qd_true[:, 2], label='Knee', color='blue')
    ax[1].set_title('Velocities')
    ax[1].legend()

    plt.show()


if __name__ == '__main__':
    app.run(main)
