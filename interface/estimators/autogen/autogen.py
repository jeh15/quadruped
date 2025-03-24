from absl import app
from absl import flags

import os

import numpy as np
import scipy

import unitree_api


FLAGS = flags.FLAGS
flags.DEFINE_string("filepath", None, "Bazel file path to the autogen folder (This should be automatically determined by the genrule).")

class AutoGet():
    def __init__(self, network_name: str, control_rate: int = 500):
        self.network_name = network_name
        self.control_rate = control_rate

        self.driver = unitree_api.UnitreeDriver(
            network_name=self.network_name,
            control_rate=self.control_rate,
        )

        self.driver.initialize()

    def initialize(self):
        # Calibrate IMU:
        gyroscope = []
        accelerometer = []
        quaternion = []
    
    def generate_defines(self):
        cc_code = f"""#pragma once
#include <array>
#include <string_view>

using namespace std::string_view_literals;

namespace interface::constants::estimator {{
    
}}
        """

        filepath = os.path.join(FLAGS.filepath, "autogen_defines.h")
        with open(filepath, "w") as f:
            f.write(cc_code)


def main(argv=None):
    # Generate defines for IMU initialization:
    autogen = AutoGet(network_name="eno2", control_rate=500)
    autogen.generate_defines()


if __name__ == "__main__":
    app.run(main)
