#include <filesystem>

#include "absl/status/status.h"
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

// Visualization:
mjvCamera cam;
mjvPerturb pert;
mjvOption opt;
mjvScene scn;
mjrContext con;


using namespace interface::containers::mock_unitree_driver;


int main(int argc, char** argv) {
    // Use Runfiles to get the path to the model:
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path mock_model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/scene_go2.xml");

    // Unitree Driver Args:
    MockUnitreeDriverArgs driver_args = {
        .xml_path = mock_model_path,
        .control_rate_us = 1000,
    };

    // Initialize Driver and Estimator:
    absl::Status result;
    std::shared_ptr<MockUnitreeDriver> unitree_driver = std::make_shared<MockUnitreeDriver>(driver_args.xml_path, driver_args.control_rate_us);
    result.Update(unitree_driver->initialize());

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

    unitree_driver->update_command(motor_commands);

    result.Update(unitree_driver->initialize_thread());

    std::cout << "Unitree Driver Initialized" << std::endl;

    // Expose mj_model and mj_data for visualization:
    auto mj_model = unitree_driver->mj_model;
    auto mj_data = unitree_driver->get_mj_data();

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

    mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();

    // Initialize Estimator:
    int estimator_control_rate = 1000;
    IMUEstimator<MockUnitreeDriver> estimator_interface(unitree_driver, estimator_control_rate);
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

    // Initialize Estimator Thread:
    result.Update(estimator_interface.initialize_thread());
    double visualization_timer = unitree_driver->get_mj_data()->time;
    double visualization_start_time = visualization_timer;
    double visualization_interval = 0.01;
    double simulation_time = 10.0;
    auto current_time = unitree_driver->get_mj_data()->time;
    while(current_time < simulation_time) {
        mj_data = unitree_driver->get_mj_data();
        current_time = mj_data->time;
        visualization_timer = current_time - visualization_start_time;

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

        unitree_driver->update_command(motor_commands);

        if(visualization_timer > visualization_interval) {
            // Print State:
            auto state = estimator_interface.get_state();
            // Convert Quaternion to Vector4
            interface::aliases::common::Vector4<float> body_rotation {
                state.body_rotation.w(), 
                state.body_rotation.x(), 
                state.body_rotation.y(), 
                state.body_rotation.z()
            };
            std::cout << "Body Position: " << state.body_position.transpose() << std::endl;
            std::cout << "Body Rotation: " << body_rotation.transpose() << std::endl;
            std::cout << "Linear Body Velocity: " << state.linear_body_velocity.transpose() << std::endl;
            std::cout << "Angular Body Velocity: " << state.angular_body_velocity.transpose() << std::endl;
            std::cout << "Motor Position: " << state.joint_position.transpose() << std::endl;
            std::cout << "Motor Velocity: " << state.joint_velocity.transpose() << std::endl;

            visualization_start_time = mj_data->time;

            mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            // swap OpenGL buffers (blocking call due to v-sync)
            glfwSwapBuffers(window);

            // process pending GUI events, call GLFW callbacks
            glfwPollEvents();
        }
    }

    std::cout << "Cleaning up" << std::endl;

    // Clean up visualization:
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Stop Threads and Clean up:
    result.Update(estimator_interface.stop_thread());
    result.Update(unitree_driver->stop_thread());
    result.Update(unitree_driver->clean_up());
    ABSL_CHECK(result.ok()) << result.message();


    return 0;
};
