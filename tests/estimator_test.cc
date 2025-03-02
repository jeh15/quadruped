#include <filesystem>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "GLFW/glfw3.h"

#include "interface/unitree_go2/simulation_interface.h"
#include "interface/unitree_go2/estimator.h"
#include "operational-space-control/unitree_go2/autogen/autogen_defines.h"

using rules_cc::cc::runfiles::Runfiles;


int main(int argc, char** argv) {
    // Use Runfiles to get the path to the model:
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path estimator_model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/go2_estimation.xml");

    std::filesystem::path mock_model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/scene_estimation.xml");

    // Estimator Args:
    EstimatorArgs estimator_args = {
        .xml_path = estimator_model_path,
        .control_rate = 1000,
    };

    // Unitree Driver Args:
    MockUnitreeDriverArgs driver_args = {
        .xml_path = mock_model_path,
        .control_rate_us = 1000,
    };

    // Initialize Driver and Estimator:
    absl::Status result;
    std::shared_ptr<MockUnitreeDriver> unitree_driver = std::make_shared<MockUnitreeDriver>(driver_args.xml_path, driver_args.control_rate_us);
    result.Update(unitree_driver->initialize());

    EstimatorInterface<MockUnitreeDriver> estimator_interface(unitree_driver, estimator_args.xml_path, estimator_args.control_rate);

    // Iterate over estimator state:
    int ndstate = estimator_interface.estimator.DimensionProcess();
    double* state = estimator_interface.estimator.State();
    std::cout << "Numbers of states: " << ndstate << std::endl;
    for(int i = 0; i <= ndstate; i++) {
        std::cout << state[i] << std::endl;
    }

    /* 
        State Structure:
        Base Position: x, y, z
        Base Orientation: w, wx, wy, wz
        Joint Positions x 4: abduction, hip, knee
        Base Linear Velocity: x, y, z
        Base Angular Velocity: wx, wy, wz
        Joint Velocities x 4: abduction, hip, knee
    */

    // Clean up estimator:
    estimator_interface.clean_up();

    return 0;
}