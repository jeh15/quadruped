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
        'data/param_regression.pkl',
    )
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    frictionloss = np.concatenate(data['frictionloss'])[-1]
    armature = np.concatenate(data['armature'])[-1]

    print(f'Friction Loss: {frictionloss}')
    print(f'Armature: {armature}')

    
if __name__ == "__main__":
    app.run(main)
