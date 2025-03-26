#include <filesystem>
#include <cmath>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "GLFW/glfw3.h"
#include "osqp++.h"

#include "interface/unitree_go2/mock_unitree_driver.h"
#include "interface/unitree_go2/interface.h"
#include "interface/estimators/imu_estimator.h"

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

    // Unitree Driver Args and Initialization:
    interface::containers::mock_unitree_driver::MockUnitreeDriverArgs driver_args = {
        .xml_path = mock_model_path,
        .control_rate_us = 1000,
    };
    std::shared_ptr<MockUnitreeDriver> unitree_driver = std::make_shared<MockUnitreeDriver>(
        driver_args.xml_path, driver_args.control_rate_us
    );

    // Estimator Args and Initialization:
    interface::containers::estimator::EstimatorArgs estimator_args = {
        .control_rate_us = 1000,
    };
    std::shared_ptr<IMUEstimator<MockUnitreeDriver>> estimator = std::make_shared<IMUEstimator<MockUnitreeDriver>>(
        unitree_driver, estimator_args.control_rate_us
    );

    // OSC Args and Initialization:
    osqp::OsqpSettings osqp_settings;
    osqp_settings.verbose = false;
    osqp_settings.polish = true;
    osqp_settings.polish_refine_iter = 3;

    interface::containers::controller::OperationalSpaceControllerArgs osc_args = {
        .xml_path = osc_model_path,
        .control_rate_us = 1000,
        .osqp_settings = osqp_settings,
    };
    std::shared_ptr<OperationalSpaceController> operational_space_controller = std::make_shared<OperationalSpaceController>(
        osc_args.xml_path, osc_args.control_rate_us, osc_args.osqp_settings
    );

    // Safety Controller Args:
    interface::containers::controller::SafetyControllerArgs safety_args = {
        .stiffness = 0.0,
        .damping = 5.0,
    };

    // Logger Args:
    interface::containers::logger::LoggerArgs logger_args = {
        .filepath = "simulation.log",
        .log_rate_us = 1000,
        .enable_logging = false,
    };

    // Initialize Interface Driver:
    absl::Status result;
    UnitreeGo2Interface interface = UnitreeGo2Interface(
        unitree_driver, estimator, operational_space_controller, safety_args, logger_args
    );

    // Initialize Unitree Driver:
    result.Update(unitree_driver->initialize());
    ABSL_CHECK(result.ok()) << result.message();

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

    cam.azimuth = 135.0;
    mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();

    // Initialize Estimator Only to test State Estimation:
    result.Update(estimator->initialize());
    ABSL_CHECK(result.ok()) << result.message();

    // Set initial Position Estimate using Forward Kinematics:
    common::Vector3<float> inital_position = common::Vector3<float>(
        static_cast<float>(mj_data->qpos[0]), static_cast<float>(mj_data->qpos[1]), static_cast<float>(mj_data->qpos[2])
    );
    result.Update(estimator->update_position_estimate(inital_position));
    ABSL_CHECK(result.ok()) << result.message();

    auto state = estimator->get_state();
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
    result.Update(estimator->initialize_thread());
    ABSL_CHECK(result.ok()) << result.message();

    // Initialize Interface:
    result.Update(interface.initialize());
    ABSL_CHECK(result.ok()) << result.message();

    // Unitree Driver Initialization Check:
    std::cout << "Unitree Driver Initialized: " << unitree_driver->is_initialized() << std::endl;
    std::cout << "Unitree Driver Thread Initialized: " << unitree_driver->is_thread_initialized() << std::endl;
    
    // Estimator Initialization Check:
    std::cout << "Estimator Initialized: " << estimator->is_initialized() << std::endl;
    std::cout << "Estimator Thread Initialized: " << estimator->is_thread_initialized() << std::endl;

    // OSC Initialization Check:
    std::cout << "OSC Initialized: " << operational_space_controller->is_initialized() << std::endl;
    std::cout << "OSC Optimization Initialized: " << operational_space_controller->is_optimization_initialized() << std::endl;

    // Initialize in PD Control Mode:
    // std::ignore = interface.default_controller_values(60.0, 5.0);
    // interface::containers::controller::ControlMode mode = 
    // interface::containers::controller::ControlMode::Default;
    // std::ignore = interface.set_control_mode(mode);

    // Initialize in OSC Control Mode:
    interface::containers::controller::ControlMode mode = 
    interface::containers::controller::ControlMode::OperationalSpaceController;
    std::ignore = interface.set_control_mode(mode);

    // Initialize Threads:
    result.Update(interface.initialize_threads());
    ABSL_CHECK(result.ok()) << result.message();

    double visualization_timer = mj_data->time;
    double visualization_start_time = visualization_timer;
    double visualization_interval = 0.01;
    double simulation_time = 20.0;
    auto current_time = mj_data->time;
    while(current_time < simulation_time) {
        mj_data = unitree_driver->mj_data;
        current_time = mj_data->time;
        visualization_timer = current_time - visualization_start_time;

        // if(current_time > 7.0) {
        //     // Set OSC Control Mode:
        //     interface::containers::controller::ControlMode mode = 
        //     interface::containers::controller::ControlMode::OperationalSpaceController;
        //     std::ignore = interface.set_control_mode(mode);
        // }

        // Update Taskspace Targets:
        osc::aliases::TaskspaceTargets taskspace_targets = osc::aliases::TaskspaceTargets::Zero();

        // Velocity:
        // auto interface_state = interface.get_state();
        // interface::aliases::common::Vector3<double> linear_control = 75.0 * (interface::aliases::common::Vector3<double>::Zero() - interface_state.linear_body_velocity);
        // interface::aliases::common::Vector3<double> angular_control = 25.0 * (interface::aliases::common::Vector3<double>::Zero() - interface_state.angular_body_velocity);
        // Eigen::Vector<double, 6> cmd {linear_control(0), linear_control(1), linear_control(2), angular_control(0), angular_control(1), angular_control(2)};
        // taskspace_targets.row(0) = cmd;

        // Position and Velocity:
        auto estimator_state = estimator->get_state();
        interface::aliases::common::Vector3<double> position_error = inital_position.cast<double>() - estimator_state.body_position.cast<double>();
        interface::aliases::common::Vector3<double> velocity_error = interface::aliases::common::Vector3<double>::Zero() - estimator_state.linear_body_velocity.cast<double>();
        interface::aliases::common::Vector3<double> rotation_error = (Eigen::Quaternion<double>(1, 0, 0, 0) * estimator_state.body_rotation.conjugate().cast<double>()).vec();
        interface::aliases::common::Vector3<double> angular_velocity_error = interface::aliases::common::Vector3<double>::Zero() - estimator_state.angular_body_velocity.cast<double>();
        interface::aliases::common::Vector3<double> linear_control = 150.0 * (position_error) + 25.0 * (velocity_error);
        interface::aliases::common::Vector3<double> angular_control = 50.0 * (rotation_error) + 10.0 * (angular_velocity_error);
        Eigen::Vector<double, 6> cmd {linear_control(0), linear_control(1), linear_control(2), angular_control(0), angular_control(1), angular_control(2)};
        taskspace_targets.row(0) = cmd;

        result.Update(interface.update_taskspace_targets(taskspace_targets));

        if(visualization_timer > visualization_interval) {
            // Print State:
            auto estimator_state = estimator->get_state();
            // interface::aliases::common::Vector4<float> body_rotation {
            //     estimator_state.body_rotation.w(), 
            //     estimator_state.body_rotation.x(), 
            //     estimator_state.body_rotation.y(), 
            //     estimator_state.body_rotation.z()
            // };
            // std::cout << "Estimator State: " << std::endl;
            // std::cout << "Body Position: " << estimator_state.body_position.transpose() << std::endl;
            // std::cout << "Body Rotation: " << body_rotation.transpose() << std::endl;
            // std::cout << "Linear Body Velocity: " << estimator_state.linear_body_velocity.transpose() << std::endl;
            // std::cout << "Angular Body Velocity: " << estimator_state.angular_body_velocity.transpose() << std::endl;
            // std::cout << "Motor Position: " << estimator_state.joint_position.transpose() << std::endl;
            // std::cout << "Motor Velocity: " << estimator_state.joint_velocity.transpose() << std::endl;
            
            // std::cout << "Safety Stop: " << interface.is_safety_stop() << std::endl;

            // auto ctrl = interface.get_torque_command();
            // std::cout << "Control: " << ctrl.transpose() << std::endl;

            // Compare mj_data to estimator:
            std::cout << "Position Estimate: " << estimator_state.body_position.transpose() << std::endl;
            std::cout << "Position mj_data: " << mj_data->qpos[0] << " " << mj_data->qpos[1] << " " << mj_data->qpos[2] << std::endl;
            std::cout << "Velocity Estimate: " << estimator_state.linear_body_velocity.transpose() << std::endl;
            std::cout << "Velocity mj_data: " << mj_data->qvel[0] << " " << mj_data->qvel[1] << " " << mj_data->qvel[2] << std::endl;

            // Compare Estimator State to Interface State:
            // auto interface_state = interface.get_state();
            // std::cout << "Interface State: " << std::endl;
            // std::cout << "Body Rotation: " << interface_state.body_rotation.transpose() << std::endl;
            // std::cout << "Linear Body Velocity: " << interface_state.linear_body_velocity.transpose() << std::endl;
            // std::cout << "Angular Body Velocity: " << interface_state.angular_body_velocity.transpose() << std::endl;
            // std::cout << "Motor Position: " << interface_state.motor_position.transpose() << std::endl;
            // std::cout << "Motor Velocity: " << interface_state.motor_velocity.transpose() << std::endl;

            visualization_start_time = mj_data->time;

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
    result.Update(interface.stop_threads());
    result.Update(interface.clean_up());
    result.Update(unitree_driver->clean_up());
    ABSL_CHECK(result.ok()) << result.message();

    return 0;
}