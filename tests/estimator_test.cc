#include <filesystem>

#include "absl/status/status.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"

#include "interface/unitree_go2/mock_unitree_driver.h"
#include "interface/estimators/imu_estimator.h"

#include "operational-space-control/unitree_go2/autogen/autogen_defines.h"
#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"

using rules_cc::cc::runfiles::Runfiles;


using namespace interface::containers::mock_unitree_driver;


int main(int argc, char** argv) {
    // Use Runfiles to get the path to the model:
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path mock_model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/scene_estimation.xml");

    // Unitree Driver Args:
    MockUnitreeDriverArgs driver_args = {
        .xml_path = mock_model_path,
        .control_rate_us = 1000,
    };

    // Initialize Driver and Estimator:
    absl::Status result;
    std::shared_ptr<MockUnitreeDriver> unitree_driver = std::make_shared<MockUnitreeDriver>(driver_args.xml_path, driver_args.control_rate_us);
    result.Update(unitree_driver->initialize());

    std::cout << "Unitree Driver Initialized" << std::endl;

    int estimator_control_rate = 1000;
    IMUEstimator<MockUnitreeDriver> estimator_interface(unitree_driver, estimator_control_rate);

    // Initialize Estimator:
    result.Update(estimator_interface.initialize());

    std::cout << "Estimator Initialized" << std::endl;

    // Print New State:
    auto state = estimator_interface.get_state();
    std::cout << "Estimator State: " << std::endl;
    std::cout << "Body Position: " << state.body_position.transpose() << std::endl;
    std::cout << "Body Rotation: " << state.body_rotation.w() << " " << state.body_rotation.vec().transpose() << std::endl;
    std::cout << "Joint Position: " << state.joint_position.transpose() << std::endl;
    std::cout << "Linear Body Velocity: " << state.linear_body_velocity.transpose() << std::endl;
    std::cout << "Angular Body Velocity: " << state.angular_body_velocity.transpose() << std::endl;
    std::cout << "Joint Velocity: " << state.joint_velocity.transpose() << std::endl;
    std::cout << "Linear Body Acceleration: " << state.linear_body_acceleration.transpose() << std::endl;
    std::cout << "Contact Mask: " << state.contact_mask.transpose() << std::endl;

    return 0;
};
