from absl import app
import os
import functools
import time
import pickle

import numpy as np


def main(argv=None):
    # Load Data:
    file_path = os.path.join(
        os.path.dirname(__file__),
        'data/unitree_data.pkl',
    )
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    data = np.asarray(data)

    
if __name__ == "__main__":
    app.run(main)
