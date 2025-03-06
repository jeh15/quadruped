#pragma once

#include <iostream>
#include <string>
#include <array>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <numbers>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"

#include "Eigen/Dense"
#include "osqp++.h"

#include "unitree-api/unitree_driver.h"
#include "unitree-api/containers.h"


#include "operational-space-control/unitree_go2/operational_space_controller.h"
#include "operational-space-control/unitree_go2/constants.h"
#include "operational-space-control/unitree_go2/containers.h"

#include "interface/unitree_go2/logger.h"
#include "interface/estimators/imu_estimator.h"
#include "interfrace/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"

namespace osc = operational_space_controller;


template <typename RobotDriver = UnitreeDriver>
class UnitreeGo2Interface {
    public:
        UnitreeGo2Interface(
            std::shared_ptr<RobotDriver> unitree_driver,
            OperationalSpaceControllerArgs osc_args, 
            EstimatorArgs estimator_args, 
            LoggerArgs log_args
        ) : 
            unitree_driver(unitree_driver),
            estimator(unitree_driver, estimator_args.control_rate_us),
            operational_space_controller(osc_args.control_rate_us, osc_args.osqp_settings),
            logger(log_args.log_filepath, log_args.logging_rate),
            enable_logging(log_args.enable_logging),
            xml_path(osc_args.xml_path) {}
        ~UnitreeGo2Interface() {}

        absl::Status initialize() {
            absl::Status result;
            // Initialize Unitree Driver:
            result.Update(unitree_driver->initialize());
            // Initialize Estimator:
            result.Update(estimator.initialize());
            // Initialize Operational Space Controller:
            result.Update(initialize_operational_space_controller());

            // Initialize Logger:
            if(enable_logging)
                result.Update(logger.initialize());

            ABSL_CHECK(result.ok()) << result.message();
            
            return absl::OkStatus();
        }

        absl::Status initialize_operational_space_controller() {
            if(!estimator->is_initialized())
                return absl::FailedPreconditionError("State Estimator not initialized. State Estimator needs to be initialized first to set the initial state of the Operational Space Controller.");

            absl::Status result;
            result.Update(update_state());
            result.Update(operational_space_controller.initialize(xml_path, state));
            result.Update(operational_space_controller.initialize_optimization());
            if (!result.ok())
                return result;

            operational_space_controller_initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_thread() {
            if(!operational_space_controller_initialized)
                return absl::FailedPreconditionError("Operational Space Controller and/or Estimator not initialized");
            
            thread = std::thread(&UnitreeGo2Interface::control_loop, this);
            thread_initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_threads() {
            // Initialize all threads:
            absl::Status result;
            result.Update(estimator.initialize_thread());
            result.Update(operational_space_controller.initialize_thread());
            result.Update(unitree_driver->initialize_thread());
            result.Update(initialize_thread());
            if(enable_logging)
                result.Update(logger.initialize_thread());

            ABSL_CHECK(result.ok()) << result.message();

            return absl::OkStatus();
        }

        absl::Status stop_thread() {
            if(!thread_initialized)
                return absl::FailedPreconditionError("Control Thread not initialized");

            running = false;
            thread.join();
            return absl::OkStatus();
        }

        absl::Status stop_threads() {
            absl::Status result;
            result.Update(stop_thread());
            result.Update(operational_space_controller.stop_thread());
            result.Update(estimator.stop_estimator_thread());
            result.Update(unitree_driver->stop_thread());
            if(enable_logging)
                result.Update(logger.stop_thread());

            return result;
        }

        absl::Status clean_up() {
            absl::Status result;
            result.Update(operational_space_controller.clean_up());    
            if(!result.ok())
                return result;

            return absl::OkStatus();
        }

        absl::Status activate_operational_space_controller() {
            if(!thread_initialized)
                return absl::FailedPreconditionError("Control Thread not initialized. Initial Control Commands must come from Default Control.");
            
            LOG(INFO) << "Activating Operational Space Controller";
            activate_control = true;
            return absl::OkStatus();
        }

        absl::Status update_taskspace_targets(const osc::aliases::TaskspaceTargets& new_taskspace_targets) {
            if (!operational_space_controller_initialized)
                return absl::FailedPreconditionError("Operational Space Controller not initialized");
            
            std::lock_guard<std::mutex> lock(mutex);
            taskspace_targets = new_taskspace_targets;
            return absl::OkStatus();
        }

        osc::containers::State get_state() {
            std::lock_guard<std::mutex> lock(mutex);
            return state;
        }

        MotorVector<double> get_torque_command() {
            std::lock_guard<std::mutex> lock(mutex);
            return operational_space_controller.get_torque_command();
        }

    private:
        /* Shared Variables */
        osc::containers::State state;
        osc::aliases::TaskspaceTargets taskspace_targets = osc::aliases::TaskspaceTargets::Zero();
        /* Components */
        OperationalSpaceController operational_space_controller;
        IMUEstimator<RobotDriver> estimator;
        std::shared_ptr<RobotDriver> unitree_driver;
        ControllerLogger logger;
        bool enable_logging;
        bool operational_space_controller_initialized = false;
        const std::filesystem::path xml_path;
        const int control_rate_us = unitree_driver->get_control_rate(); // This should match the control rate of the motor controller.
        /* Setpoint Integration */
        const double alpha = 0.9;
        const double timestep = std::chrono::duration<double>(std::chrono::microseconds(control_rate_us)).count();
        /* Initial PD Controller */
        float stiffness_value = 5.0;
        float damping_value = 5.0;
        float stiffness_delta = 0.01;
        float damping_delta = 0.0;
        /* Thread Variables */
        std::atomic<bool> running{true};
        std::atomic<bool> activate_control{false};
        std::thread thread;
        std::mutex mutex;
        bool thread_initialized = false;

        // Update State comes from Estimator now...
        // absl::Status update_state() {
        //     // Get Current State for Unitree Go2 Motor Driver:
        //     unitree::containers::LowState low_state = unitree_driver.get_low_state();
        //     unitree::containers::IMUState imu_state = unitree_driver.get_imu_state();
        //     unitree::containers::MotorState motor_state = unitree_driver.get_motor_state();

        //     // Create contact mask:
        //     ContactMask contact_mask = ContactMask::Zero();
        //     Eigen::Vector<short, 4> foot_force = Eigen::Map<Eigen::Vector<short, 4>>(low_state.foot_force.data())(foot_idx_map);
        //     for(int i = 0; i < 4; i++) {
        //         contact_mask(i) = foot_force(i) > contact_threshold;
        //     }

        //     // Reformat data to match Mujoco Model: 
        //     MotorVectorFloat motor_position = Eigen::Map<MotorVectorFloat>(motor_state.q.data())(motor_idx_map);
        //     MotorVectorFloat motor_velocity = Eigen::Map<MotorVectorFloat>(motor_state.qd.data())(motor_idx_map);
        //     MotorVectorFloat motor_acceleration = Eigen::Map<MotorVectorFloat>(motor_state.qdd.data())(motor_idx_map);
        //     MotorVectorFloat motor_torque_estimate = Eigen::Map<MotorVectorFloat>(motor_state.torque_estimate.data())(motor_idx_map);
        //     QuaternionFloat body_rotation = Eigen::Map<QuaternionFloat>(imu_state.quaternion.data());
        //     Vector3Float angular_body_velocity = Eigen::Map<Vector3Float>(imu_state.gyroscope.data());
        //     Vector3Float linear_body_acceleration = Eigen::Map<Vector3Float>(imu_state.accelerometer.data());

        //     // Unitree does not provide linear velocity:
        //     smooth_acceleration = alpha * linear_body_acceleration + (1 - alpha) * previous_smooth_acceleration;
        //     linear_body_velocity = previous_linear_body_velocity + smooth_acceleration * control_rate_us * 1.0e-6f;
        //     smooth_velocity = alpha * linear_body_velocity + (1 - alpha) * previous_smooth_velocity;

        //     state.motor_position = motor_position.cast<double>();
        //     state.motor_velocity = motor_velocity.cast<double>();
        //     state.motor_acceleration = motor_acceleration.cast<double>();
        //     state.torque_estimate = motor_torque_estimate.cast<double>();
        //     state.body_rotation = body_rotation.cast<double>();
        //     state.angular_body_velocity = angular_body_velocity.cast<double>();
        //     state.linear_body_velocity = smooth_velocity.cast<double>();
        //     state.linear_body_acceleration = linear_body_acceleration.cast<double>();
        //     state.contact_mask = contact_mask;

        //     return absl::OkStatus();
        // }

        unitree::containers::MotorCommand update_motor_command(
            MotorVector<float>& torque_commands,
            MotorVector<float>& velocity_setpoints = MotorVector<float>::Zero(),
            MotorVector<float>& position_setpoints = MotorVector<float>::Zero(),
        ) {
            // Run safety controller on commands:
            absl::Status result;
            result.Update(safety_controller.torque_saturator(torque_commands, state));
            result.Update(safety_controller.setpoint_saturator(position_setpoints, velocity_setpoints));
            if(!result.ok()) {
                result.Update(safety_controller.setpoint_override());
            }
            // Alternatively, kill process: However, this will not stop the inertia of the robot.
            // ABSL_CHECK(result.ok()) << result.message();
            
            // Get Safety Controller State
            SafetyControllerState safety_controller_state = safety_controller.get_state();
            bool stop_control = safety_controller.stop_control();

            // Create Motor Command: Cast to float for motor controller:
            std::array<float, osc::constants::model::nu_size> q_setpoint;
            std::array<float, osc::constants::model::nu_size> qd_setpoint;
            std::array<float, osc::constants::model::nu_size> torque_feedforward;
            std::array<float, osc::constants::model::nu_size> stiffness;
            std::array<float, osc::constants::model::nu_size> damping;

            Eigen::Map<MotorVector<float>>(q_setpoint.data()) = safety_controller_state.position_setpoint.cast<float>();
            Eigen::Map<MotorVector<float>>(qd_setpoint.data()) = safety_controller_state.velocity_setpoint.cast<float>();
            Eigen::Map<MotorVector<float>>(torque_feedforward.data()) = afety_controller_state.torque_command.cast<float>();
            Eigen::Map<MotorVector<float>>(stiffness.data()) = MotorVector<float>::Constant(static_cast<float>(safety_controller_state.stiffness));
            Eigen::Map<MotorVector<float>>(damping.data()) = MotorVector<float>::Constant(static_cast<float>(safety_controller_state.damping));
            
            unitree::containers::MotorCommand motor_command = {
                .q_setpoint = q_setpoint,
                .qd_setpoint = qd_setpoint,
                .torque_feedforward = torque_feedforward,
                .stiffness = stiffness,
                .damping = damping,
            };

            return motor_command;
        }

        unitree::containers::MotorCommand default_motor_command(const float stiffness_value = 5.0, const float damping_value = 5.0) {
            /*
                Hold default position.
            */

            // Clamp values:
            std::clamp(stiffness_value, 0.0f, 120.0f);
            std::clamp(damping_value, 0.0f, 5.0f);

            std::array<float, osc::constants::model::nu_size> q_setpoint = {
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
            };
            std::array<float, osc::constants::model::nu_size> qd_setpoint = { 0 };
            std::array<float, osc::constants::model::nu_size> torque_feedforward = { 0 };
            std::array<float, osc::constants::model::nu_size> stiffness = { 
                stiffness_value, stiffness_value, stiffness_value,
                stiffness_value, stiffness_value, stiffness_value,
                stiffness_value, stiffness_value, stiffness_value,
                stiffness_value, stiffness_value, stiffness_value,
            };
            std::array<float, osc::constants::model::nu_size> damping = {
                damping_value, damping_value, damping_value,
                damping_value, damping_value, damping_value,
                damping_value, damping_value, damping_value,
                damping_value, damping_value, damping_value,
            };
            std::array<float, osc::constants::model::nu_size> kp = { 
                5.0, 5.0, 5.0,
                5.0, 5.0, 5.0,
                5.0, 5.0, 5.0,
                5.0, 5.0, 5.0,
             };
            std::array<float, osc::constants::model::nu_size> kd = { 
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
             };
            
            unitree::containers::MotorCommand motor_command = {
                .q_setpoint = q_setpoint,
                .qd_setpoint = qd_setpoint,
                .torque_feedforward = torque_feedforward,
                .stiffness = stiffness,
                .damping = damping,
            };

            return motor_command;
        }
        
        void control_loop() {
            using Clock = std::chrono::steady_clock;
            auto next_time = Clock::now();
            while(running) {
                // Calculate next time:
                next_time += std::chrono::microseconds(control_rate_us);
                /* Lock Guard Scope */
                {
                    std::lock_guard<std::mutex> lock(mutex);

                    // Get Robot State from Motor Controller and Update State Struct: Shared Variable (state)
                    absl::Status result = update_state();

                    // Update Operational Space Controller mj_model with State: Shared Variable (state)
                    operational_space_controller.update_state(state);

                    // Update Operational Space Controller with Taskspace Targets: Shared Variable (taskspace_targets)
                    operational_space_controller.update_taskspace_targets(taskspace_targets);

                    if(enable_logging)
                        result.Update(logger.update_state(state));
                }

                // Get Torque Command: (OSC Locks this)
                ActuatorCommand torque_command = operational_space_controller.get_torque_command()(motor_idx_map);
                ActuatorCommandFloat torque_command_f = torque_command.cast<float>();

                // Get Solution to get Joint Accelerations and Torques:
                // OptimizationSolution solution = operational_space_controller.get_solution();
                // ActuatorCommand joint_accelerations = solution(Eigen::seqN(0, optimization::dv_size))(motor_idx_map);
                // ActuatorCommand torque_command = solution(Eigen::seqN(optimization::dv_idx, optimization::u_size))(motor_idx_map);

                // Integrate to get velocity setpoints:
                // ActuatorCommand velocity_desired = state.motor_velocity + joint_accelerations * control_rate_us * 1.0e-6;
                // ActuatorCommand velocity_setpoint = alpha * velocity_desired + (1 - alpha) * state.motor_velocity;


                // Create Motor Command:
                unitree::containers::MotorCommand motor_command;
                if(activate_control) {
                    motor_command = update_motor_command(torque_command_f);
                }
                else {
                    motor_command = default_motor_command(stiffness_value, damping_value);
                    stiffness_value += stiffness_delta;
                    damping_value += damping_delta;
                }

                // Send Motor Command: (Motor Controller Locks this)
                unitree_driver.update_command(motor_command);

                // Check for overrun and sleep until next time:
                auto now = Clock::now();
                if (now < next_time) {
                    std::this_thread::sleep_until(next_time);
                } 
                else {
                    // Log overrun:
                    auto overrun = std::chrono::duration_cast<std::chrono::microseconds>(now - next_time);
                    std::cout << "Interface Control Loop Execution Time Exceeded Control Rate: " 
                        << overrun.count() << "us" << std::endl;
                    next_time = now;
                }
            }
        }
};
