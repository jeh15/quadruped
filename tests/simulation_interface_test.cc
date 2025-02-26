#include <filesystem>
#include "rules_cc/cc/runfiles/runfiles.h"

#include "interface/unitree_go2/simulation_interface.h"

using rules_cc::cc::runfiles::Runfiles;


int main(int argc, char** argv) {
    // Use Runfiles to get the path to the model:
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path model_path = 
        runfiles->Rlocation("quadruped/models/unitree_go2/go2_mjx_torque.xml");

    std::filesystem::path mock_model_path = 
        runfiles->Rlocation("quadruped/models/unitree_go2/scene_mjx_torque.xml");

    // OSC Args:
    OperationalSpaceControllerArgs osc_args = {
        .xml_path = model_path,
        .control_rate = 1000,
        .osqp_settings = osqp::OsqpSettings(),
    };

    // MC Args:
    MockMotorControllerArgs mc_args = {
        .xml_path = mock_model_path,
        .control_rate = 2000,
    };

    // Initialize Interface Driver:
    UnitreeGo2Interface unitree_driver(osc_args, mc_args);

    return 0;
}