#include <filesystem>
#include <cmath>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "GLFW/glfw3.h"

#include "interface/unitree_go2/mock_unitree_driver.h"
#include "interface/unitree_go2/interface.h"

#include "operational-space-control/unitree_go2/constants.h"
#include "unitree-api/containers.h"
#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"
#include "interface/unitree_go2/constants.h"

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

    std::filesystem::path osc_model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/go2.xml");

    std::filesystem::path mock_model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/scene_go2.xml");

    // Unitree Driver Args:
    interface::containers::mock_unitree_driver::MockUnitreeDriverArgs driver_args = {
        .xml_path = mock_model_path,
        .control_rate_us = 1000,
    };

    // Estimator Args: (This is useless...)
    interface::containers::estimator::EstimatorArgs estimator_args = {
        .control_rate_us = 1000,
    };

    // OSC Args:
    interface::containers::controller::OperationalSpaceControllerArgs osc_args = {
        .xml_path = osc_model_path,
        .control_rate_us = 1000,
    };

    // Safety Controller Args:
    interface::containers::controller::SafetyControllerArgs safety_args = {
        .stiffness = 0.0,
        .damping = 5.0,
    };

    // Logger Args:
    interface::containers::logger::LoggerArgs logger_args = {
        .filepath = "simulation.log",
        .log_rate_us = 1000,
        .enable_logging = true,
    };


    // Initialize Driver:
    absl::Status result;
    std::shared_ptr<MockUnitreeDriver> unitree_driver = std::make_shared<MockUnitreeDriver>(
        driver_args.xml_path, driver_args.control_rate_us
    );
    result.Update(unitree_driver->initialize());
    ABSL_CHECK(result.ok()) << result.message();

    // Initialize Interface:
    UnitreeGo2Interface<MockUnitreeDriver> interface(
        unitree_driver, estimator_args, osc_args, safety_args, logger_args
    );
    result.Update(interface.initialize());
    ABSL_CHECK(result.ok()) << result.message();

    // // Expose mj_model and mj_data for visualization:
    // auto mj_model = unitree_driver->mj_model;
    // auto mj_data = unitree_driver->mj_data;

    // // Visualization:
    // glfwInit();
    // GLFWwindow* window = glfwCreateWindow(800, 600, "Demo", NULL, NULL);
    // glfwMakeContextCurrent(window);
    // glfwSwapInterval(1);

    // // initialize visualization data structures
    // mjv_defaultCamera(&cam);
    // mjv_defaultPerturb(&pert);
    // mjv_defaultOption(&opt);
    // mjr_defaultContext(&con);
    // mjv_makeScene(mj_model, &scn, 1000);
    // mjr_makeContext(mj_model, &con, mjFONTSCALE_100);

    // // get framebuffer viewport
    // mjrRect viewport = {0, 0, 0, 0};
    // glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // // Initialize Estimator and Unitree Driver Threads:
    // result.Update(estimator_interface.initialize_estimator_thread());
    // result.Update(unitree_driver->initialize_control_thread());
    // ABSL_CHECK(result.ok()) << result.message();

    // double visualization_timer = unitree_driver->mj_data->time;
    // double visualization_start_time = visualization_timer;
    // double visualization_interval = 0.01;
    // double simulation_time = 2.0;
    // while(unitree_driver->mj_data->time < simulation_time) {
    //     visualization_timer = unitree_driver->mj_data->time - visualization_start_time;

    //     // Update Motor Commands:
    //     unitree::containers::MotorCommand motor_commands;
    //     motor_commands.q_setpoint = {
    //         0.0, 0.9, -1.8,
    //         0.0, 0.9, -1.8,
    //         0.0, 0.9, -1.8,
    //         0.0, 0.9, -1.8, 
    //     };
    //     motor_commands.qd_setpoint = {
    //         0.0, 0.0, 0.0,
    //         0.0, 0.0, 0.0,
    //         0.0, 0.0, 0.0,
    //         0.0, 0.0, 0.0,
    //     };
    //     motor_commands.torque_feedforward = {
    //         0.0, 0.0, 0.0,
    //         0.0, 0.0, 0.0,
    //         0.0, 0.0, 0.0,
    //         0.0, 0.0, 0.0,
    //     };
    //     motor_commands.stiffness = {
    //         60.0, 60.0, 60.0,
    //         60.0, 60.0, 60.0,
    //         60.0, 60.0, 60.0,
    //         60.0, 60.0, 60.0,
    //     };
    //     motor_commands.damping = {
    //         5.0, 5.0, 5.0,
    //         5.0, 5.0, 5.0,
    //         5.0, 5.0, 5.0,
    //         5.0, 5.0, 5.0,
    //     };
    //     // unitree_driver->update_command(motor_commands);

    //     mj_data = unitree_driver->mj_data;
    //     if(visualization_timer > visualization_interval) {
    //         // Print State:
    //         auto state = estimator_interface.get_state();
    //         std::cout << "Body Position: " << state.body_position.transpose() << std::endl;
    //         std::cout << "Body Rotation: " << state.body_rotation.transpose() << std::endl;
    //         std::cout << "Linear Body Velocity: " << state.linear_body_velocity.transpose() << std::endl;
    //         std::cout << "Angular Body Velocity: " << state.angular_body_velocity.transpose() << std::endl;
    //         std::cout << "Motor Position: " << state.motor_position.transpose() << std::endl;
    //         std::cout << "Motor Velocity: " << state.motor_velocity.transpose() << std::endl;

    //         visualization_start_time = unitree_driver->mj_data->time;

    //         mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
    //         mjr_render(viewport, &scn, &con);

    //         // swap OpenGL buffers (blocking call due to v-sync)
    //         glfwSwapBuffers(window);

    //         // process pending GUI events, call GLFW callbacks
    //         glfwPollEvents();
    //     }
    // }

    // Clean up visualization:
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Stop Threads and Clean up:
    result.Update(interface.stop_threads());
    result.Update(interface.clean_up());
    result.Update(unitree_driver->clean_up());
    ABSL_CHECK(result.ok()) << result.message();

    return 0;
}