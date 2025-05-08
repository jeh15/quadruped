from absl import app
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import flax.struct


@flax.struct.dataclass
class minibatch:
    q: jax.Array
    qd: jax.Array
    ctrl: jax.Array


def main(argv=None):
    # Load Data:
    file_path = os.path.join(
        os.path.dirname(__file__),
        'data/param_regression_history_motor.pkl',
    )
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    dof_damping = np.concatenate(data['dof_damping'])[-1]
    kp = np.concatenate(data['kp'])[-1]

    print(f'Damping: {dof_damping}')
    print(f'Stiffness: {kp}')

    pass

    
if __name__ == "__main__":
    app.run(main)
