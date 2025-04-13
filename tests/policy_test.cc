#include <filesystem>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "GLFW/glfw3.h"

#include "interface/unitree_go2/mock_unitree_driver.h"
#include "interface/unitree_go2/policy_interface.h"

#include "unitree-api/containers.h"
#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"

using rules_cc::cc::runfiles::Runfiles;


// Visualization:
mjvCamera cam;
mjvPerturb pert;
mjvOption opt;
mjvScene scn;
mjrContext con;


int main(int argc, char** argv) {
    // Use Runfiles to get the path to the model:
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path onnx_model_path = 
        runfiles->Rlocation("unitree-interface/onnx_models/breezy-dawn-149.onnx");


    std::filesystem::path mock_model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/scene_collision.xml");

    absl::Status result;

    // Initialize Mock Unitree Driver:
    std::shared_ptr<MockUnitreeDriver> mock_unitree_driver = 
        std::make_shared<MockUnitreeDriver>(mock_model_path, 2000);
    result.Update(mock_unitree_driver->initialize());
    ABSL_CHECK(result.ok()) << result.message();

    // Initialize Policy Interface:
    PolicyInterface policy_interface(
        onnx_model_path,
        mock_unitree_driver
    );
    result.Update(policy_interface.initialize());
    ABSL_CHECK(result.ok()) << result.message();

    return 0;
}

