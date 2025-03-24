#include <filesystem>
#include <cmath>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "GLFW/glfw3.h"

#include "interface/unitree_go2/mock_unitree_driver.h"
#include "interface/estimators/imu_estimator.h"

#include "operational-space-control/unitree_go2/autogen/autogen_defines.h"
#include "unitree-api/containers.h"
#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"

using rules_cc::cc::runfiles::Runfiles;


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

    std::filesystem::path estimator_model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/go2.xml");

    std::filesystem::path mock_model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/scene.xml");

    // Estimator Args:
    interface::containers::estimator::EstimatorArgs estimator_args = {
        .xml_path = estimator_model_path,
        .control_rate_us = 1000,
    };

    // Logger Args
    interface::containers::logger::LoggerArgs estimator_logger_args = {
        .filepath = "estimator.log",
        .log_rate_us = 1000,
        .enable_logging = true,
    };

    interface::containers::logger::LoggerArgs driver_logger_args = {
        .filepath = "mock_driver.log",
        .log_rate_us = 10000,
        .enable_logging = false,
    };

    // Unitree Driver Args:
    MockUnitreeDriverArgs driver_args = {
        .xml_path = mock_model_path,
        .control_rate_us = 1000,
    };

    // Initialize Driver and Estimator:
    absl::Status result;
    std::shared_ptr<MockUnitreeDriver> unitree_driver = std::make_shared<MockUnitreeDriver>(driver_args.xml_path, driver_args.control_rate_us, driver_logger_args);
    result.Update(unitree_driver->initialize());
    ABSL_CHECK(result.ok()) << result.message();

    EstimatorInterface<MockUnitreeDriver> estimator_interface(unitree_driver, estimator_args, estimator_logger_args);
    result.Update(estimator_interface.initialize());
    ABSL_CHECK(result.ok()) << result.message();

    // Expose mj_model and mj_data for visualization:
    auto mj_model = unitree_driver->mj_model;
    auto mj_data = unitree_driver->mj_data;

    // Visualization:
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultPerturb(&pert);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
    mjv_makeScene(mj_model, &scn, 1000);
    mjr_makeContext(mj_model, &con, mjFONTSCALE_100);

    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // Initialize Estimator and Unitree Driver Threads:
    result.Update(estimator_interface.initialize_estimator_thread());
    result.Update(unitree_driver->initialize_control_thread());
    ABSL_CHECK(result.ok()) << result.message();

    double visualization_timer = unitree_driver->mj_data->time;
    double visualization_start_time = visualization_timer;
    double visualization_interval = 0.01;
    double simulation_time = 2.0;
    while(unitree_driver->mj_data->time < simulation_time) {
        visualization_timer = unitree_driver->mj_data->time - visualization_start_time;

        // Update Motor Commands:
        unitree::containers::MotorCommand motor_commands;
        motor_commands.q_setpoint = {
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8, 
        };
        motor_commands.qd_setpoint = {
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        };
        motor_commands.torque_feedforward = {
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        };
        motor_commands.stiffness = {
            60.0, 60.0, 60.0,
            60.0, 60.0, 60.0,
            60.0, 60.0, 60.0,
            60.0, 60.0, 60.0,
        };
        motor_commands.damping = {
            5.0, 5.0, 5.0,
            5.0, 5.0, 5.0,
            5.0, 5.0, 5.0,
            5.0, 5.0, 5.0,
        };
        // unitree_driver->update_command(motor_commands);

        mj_data = unitree_driver->mj_data;
        if(visualization_timer > visualization_interval) {
            // Print State:
            auto state = estimator_interface.get_state();
            std::cout << "Body Position: " << state.body_position.transpose() << std::endl;
            std::cout << "Body Rotation: " << state.body_rotation.transpose() << std::endl;
            std::cout << "Linear Body Velocity: " << state.linear_body_velocity.transpose() << std::endl;
            std::cout << "Angular Body Velocity: " << state.angular_body_velocity.transpose() << std::endl;
            std::cout << "Motor Position: " << state.motor_position.transpose() << std::endl;
            std::cout << "Motor Velocity: " << state.motor_velocity.transpose() << std::endl;

            visualization_start_time = unitree_driver->mj_data->time;

            mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            // swap OpenGL buffers (blocking call due to v-sync)
            glfwSwapBuffers(window);

            // process pending GUI events, call GLFW callbacks
            glfwPollEvents();
        }
    }

    // Clean up visualization:
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Stop Threads and Clean up:
    result.Update(estimator_interface.stop_estimator_thread());
    result.Update(unitree_driver->stop_control_thread());
    result.Update(estimator_interface.clean_up());
    result.Update(unitree_driver->clean_up());
    ABSL_CHECK(result.ok()) << result.message();

    return 0;
}