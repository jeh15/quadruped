from absl import app
from absl import flags

import os

import mujoco

from python.runfiles import Runfiles


FLAGS = flags.FLAGS
flags.DEFINE_string("filepath", None, 'Bazel filepath to the autogen folder (This should be automatically determined by the genrule).')


class AutoGen():
    def __init__(self, mj_model: mujoco.MjModel):
        self.mj_model = mj_model

        # Estimator Constants:
        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv
        self.na = self.mj_model.na
        self.nu = self.mj_model.nu
        self.state_size = self.mj_model.nq + self.mj_model.nv + self.mj_model.na
        self.sensor_size = self.mj_model.nsensor
        self.sensordata_size = 0
        for i in range(self.sensor_size):
            self.sensordata_size += self.mj_model.sensor_dim[i]

    def generate_defines(self):
        cc_code = f"""#pragma once

namespace constants::estimator {{
    // Estimator Constants:
    constexpr int nq = {self.nq};
    constexpr int nv = {self.nv};
    constexpr int na = {self.na};
    constexpr int nu = {self.nu};
    constexpr int state_size = {self.state_size};
    constexpr int sensor_size = {self.sensor_size};
    constexpr int sensordata_size = {self.sensordata_size};
}}
        """
        filepath = os.path.join(FLAGS.filepath, "estimator_defines.h")
        with open(filepath, "w") as f:
            f.write(cc_code)


def main(argv):
    # Initialize Mujoco Model: (TODO(jeh15): This is hardcoded for now but can be passed as a flag from the genrule)
    r = Runfiles.Create()
    mj_model = mujoco.MjModel.from_xml_path(
        r.Rlocation(
            path="mujoco-models/models/unitree_go2/go2_estimation.xml",
        )
    )

    # Generate Defines:
    autogen = AutoGen(mj_model)
    autogen.generate_defines()


if __name__ == "__main__":
    app.run(main)
