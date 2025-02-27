#include <filesystem>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "osqp++.h"
#include "GLFW/glfw3.h"

#include "interface/unitree_go2/simulation_interface.h"
#include "operational-space-control/unitree_go2/autogen/autogen_defines.h"

using rules_cc::cc::runfiles::Runfiles;


mjvCamera cam;                      // abstract camera
mjvPerturb pert;                    // perturbation object
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context


int main(int argc, char** argv) {
    // Use Runfiles to get the path to the model:
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path model_path = 
        runfiles->Rlocation("unitree-interface/models/unitree_go2/go2_mjx_torque.xml");

    std::filesystem::path mock_model_path = 
        runfiles->Rlocation("unitree-interface/models/unitree_go2/scene_mjx_torque.xml");

    // OSC Args:
    OsqpSettings osqp_settings;
    osqp_settings.verbose = false;
    osqp_settings.polish = true;
    osqp_settings.polish_refine_iter = 3;
    osqp_settings.eps_abs = 1e-3;

    // OSC Args:
    OperationalSpaceControllerArgs osc_args = {
        .xml_path = model_path,
        .control_rate = 1000,
        .osqp_settings = osqp_settings,
    };

    // MC Args:
    MockMotorControllerArgs mc_args = {
        .xml_path = mock_model_path,
        .control_rate = 2000,
    };

    // Logger Args:
    StateLoggerArgs log_args = {
        .log_filepath = "simulation_test.log",
        .logging_rate = 100000,
        .enable_logging = true,
    };

    // Initialize Interface Driver:
    absl::Status result;
    UnitreeGo2Interface unitree_driver(osc_args, mc_args, log_args);
    result.Update(unitree_driver.initialize());

    auto mj_model = unitree_driver.motor_controller.mj_model;

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
    mjv_makeScene(mj_model, &scn, 1000);                     // space for 1000 objects
    mjr_makeContext(mj_model, &con, mjFONTSCALE_100);     // model-specific context


    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // Update Taskspace Targets:
    Eigen::Matrix<double, model::site_ids_size, 6, Eigen::RowMajor> taskspace_targets = Eigen::Matrix<double, constants::model::site_ids_size, 6, Eigen::RowMajor>::Zero();
    taskspace_targets << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    
    std::ignore = unitree_driver.update_taskspace_targets(taskspace_targets);

    // These initialize or throw:
    result.Update(unitree_driver.initialize_threads());
    ABSL_CHECK(result.ok()) << result.message();

    // Main Loop:
    State state;
    double kd = 2.0;
    double magnitude = 0.5;
    double frequency = 0.5;

    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();

    int visualize_iter = 0;
    while(unitree_driver.motor_controller.mj_data->time < 30) {
        // auto torque_command = unitree_driver.get_torque_command();
        // std::cout << "Torque Command: " << torque_command.transpose() << std::endl;

        auto now = Clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
        auto time = elapsed_time.count() * 1.0e-3;

        // Sinusoidal Z Targets:
        state = unitree_driver.get_state();
        double velocity_target = magnitude * frequency * std::cos(frequency * time);
        double acceleration_target = -magnitude * frequency * frequency * std::sin(frequency * time);
        double velocity_error = velocity_target - state.linear_body_velocity(2);
        double command = acceleration_target + kd * velocity_error;
        taskspace_targets(0, 2) = command;
        std::ignore = unitree_driver.update_taskspace_targets(taskspace_targets);

        auto mj_data = unitree_driver.motor_controller.mj_data;
        if(visualize_iter % 10 == 0) {
            mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            // swap OpenGL buffers (blocking call due to v-sync)
            glfwSwapBuffers(window);

            // process pending GUI events, call GLFW callbacks
            glfwPollEvents();
        }
        visualize_iter++;
    }

    result.Update(unitree_driver.stop_threads());
    result.Update(unitree_driver.clean_up());
    ABSL_CHECK(result.ok()) << result.message();

    return 0;
}