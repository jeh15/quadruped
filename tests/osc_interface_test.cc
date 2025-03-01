#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "Eigen/Dense"
#include "osqp++.h"

#include "interface/unitree_go2/logger.h"
#include "interface/unitree_go2/osc_interface.h"

using rules_cc::cc::runfiles::Runfiles;


int main(int argc, char** argv) {
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path model_path = 
        runfiles->Rlocation("unitree-interface/models/unitree_go2/go2_mjx_torque.xml");

    // OSC Args:
    OsqpSettings osqp_settings;
    osqp_settings.verbose = false;
    osqp_settings.polish = true;
    osqp_settings.polish_refine_iter = 3;
    osqp_settings.eps_abs = 1e-3;

    OperationalSpaceControllerArgs osc_args = {
        .xml_path = model_path,
        .control_rate = 1000,
        .osqp_settings = osqp_settings,
    };

    // MC Args:
    MotorControllerArgs mc_args = {
        .network_name = "eno2",
        .control_rate = 2000,
    };

    // Logger Args:
    StateLoggerArgs log_args = {
        .log_filepath = "hardware_test.log",
        .logging_rate = 100000,
        .enable_logging = true,
    };

    // Initialize Interface Driver:
    absl::Status result;
    UnitreeGo2Interface unitree_driver(osc_args, mc_args, log_args);
    result.Update(unitree_driver.initialize());

    // Update Taskspace Targets:
    Eigen::Matrix<double, model::site_ids_size, 6, Eigen::RowMajor> taskspace_targets = Eigen::Matrix<double, constants::model::site_ids_size, 6, Eigen::RowMajor>::Zero();
    taskspace_targets << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    
    std::ignore = unitree_driver.update_taskspace_targets(taskspace_targets);

    // Initialize Threads:
    result.Update(unitree_driver.initialize_threads());

    std::cout << "Press Enter to Activate the Operational Space Controller: " << std::endl;
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    while(true) {
        auto now = Clock::now();
        auto elapsed_time_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start);
        if(elapsed_time_seconds.count() > 15) {
            break;
        }
    }

    // Activate Operational Space Controller:
    State state;
    double kd = 5.0;
    Eigen::Vector<double, 3> zero_taskspace = Eigen::Vector<double, 3>::Zero();
    result.Update(unitree_driver.activate_operational_space_controller());
    ABSL_CHECK(result.ok()) << result.message();
    std::cout << "Press Enter to Terminate Process: " << std::endl;
    start = Clock::now();
    while(true) {
        // auto now = Clock::now();
        // auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
        // auto time = elapsed_time.count() * 1.0e-3;
        // Sinusoidal Z Targets:
        state = unitree_driver.get_state();
        Eigen::Vector<double, 3> linear_velocity_feedback = kd * (zero_taskspace -  state.linear_body_velocity);
        Eigen::Vector<double, 3> angular_velocity_feedback = kd * (zero_taskspace -  state.angular_body_velocity);
        for(int i = 0; i < 3; i++) {
            taskspace_targets(0, i) = linear_velocity_feedback[i];
            taskspace_targets(0, i + 3) = angular_velocity_feedback[i];
        }
        // std::ignore = unitree_driver.update_taskspace_targets(taskspace_targets);
        std::cout << "Linear Velocity Feedback: " << linear_velocity_feedback.transpose() << std::endl;
        std::cout << "Angular Velocity Feedback: " << angular_velocity_feedback.transpose() << std::endl;
    }

    // Stop threads and clean up:
    result.Update(unitree_driver.stop_threads());
    result.Update(unitree_driver.clean_up());
    ABSL_CHECK(result.ok()) << result.message();

    return 0;
}
