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

#include "interface/unitree_go2/safety_controller.h"
#include "interface/unitree_go2/logger.h"
#include "interface/estimators/imu_estimator.h"
#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"
#include "interface/unitree_go2/constants.h"

namespace osc = operational_space_controller;

using namespace interface::aliases::common;
using namespace interface::containers::controller;
using namespace interface::containers::logger;
using namespace interface::containers::estimator;


template <typename RobotDriver = UnitreeDriver>
class UnitreeGo2Interface {
    public:
        UnitreeGo2Interface(
            std::shared_ptr<RobotDriver> unitree_driver,
            EstimatorArgs estimator_args,
            OperationalSpaceControllerArgs osc_args,
            SafetyControllerArgs safety_args,
            LoggerArgs log_args
        ) : 
            unitree_driver(unitree_driver),
            estimator(unitree_driver, estimator_args.control_rate_us),
            operational_space_controller(osc_args.control_rate_us, osc_args.osqp_settings),
            safety_controller(safety_args.stiffness, safety_args.damping),
            logger(log_args.filepath, log_args.log_rate_us),
            enable_logging(log_args.enable_logging),
            xml_path(osc_args.xml_path) {}
        ~UnitreeGo2Interface() {}

        absl::Status initialize() {
            absl::Status result;
            // Initialize Unitree Driver if not initialized:
            if(!unitree_driver->is_initialized())
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
            if(!estimator.is_initialized())
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
            result.Update(estimator.stop_thread());
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

        absl::Status set_control_mode(const ControlMode new_control_mode) {
            std::lock_guard<std::mutex> lock(mutex);
            control_mode = new_control_mode;
            return absl::OkStatus();
        }

        ControlMode get_control_mode() {
            std::lock_guard<std::mutex> lock(mutex);
            return control_mode;
        }

        bool is_safety_stop() {
            return safety_stop;
        }

    private:
        /* Shared Variables */
        osc::containers::State state;
        osc::aliases::TaskspaceTargets taskspace_targets = osc::aliases::TaskspaceTargets::Zero();
        /* Control Mode */
        ControlMode control_mode = ControlMode::Damping;
        /* Components */
        std::shared_ptr<RobotDriver> unitree_driver;
        IMUEstimator<RobotDriver> estimator;
        OperationalSpaceController operational_space_controller;
        SafetyController safety_controller;
        ControllerLogger logger;
        bool enable_logging;
        const std::filesystem::path xml_path;
        bool operational_space_controller_initialized = false;
        bool safety_stop = false;
        const int control_rate_us = unitree_driver->get_control_rate(); // This should match the control rate of the motor controller.
        /* Setpoint Integration */
        const double alpha = 0.9;
        const double timestep = std::chrono::duration<double>(std::chrono::microseconds(control_rate_us)).count();
        MotorVector<double> previous_motor_velocity = MotorVector<double>::Zero();
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

        absl::Status update_state() {
            EstimatorState estimator_state = estimator.get_state();
            // Convert Quaternion to Vector4
            Vector4<float> body_rotation {
                estimator_state.body_rotation.w(), 
                estimator_state.body_rotation.x(), 
                estimator_state.body_rotation.y(), 
                estimator_state.body_rotation.z()
            };

            state.motor_position = estimator_state.joint_position.cast<double>();
            state.motor_velocity = estimator_state.joint_velocity.cast<double>();
            // Unitree returns vector of zeros, estimator is not tracking this...
            state.motor_acceleration = MotorVector<double>::Zero();
            state.torque_estimate = estimator_state.torque_estimate.cast<double>();
            state.body_rotation = body_rotation.cast<double>();
            state.linear_body_velocity = estimator_state.linear_body_velocity.cast<double>();
            state.angular_body_velocity = estimator_state.angular_body_velocity.cast<double>();
            state.linear_body_acceleration = estimator_state.linear_body_acceleration.cast<double>();
            state.contact_mask = estimator_state.contact_mask.cast<double>();

            return absl::OkStatus();
        }

        unitree::containers::MotorCommand update_motor_command(
            const MotorVector<double>& torque_commands,
            const MotorVector<double>& velocity_setpoints = MotorVector<double>::Zero(),
            const MotorVector<double>& position_setpoints = MotorVector<double>::Zero()
        ) {
            // Run safety controller on commands:
            absl::Status result;
            result.Update(safety_controller.torque_saturator(torque_commands, state));
            result.Update(safety_controller.setpoint_saturator(position_setpoints, velocity_setpoints));
            if(!result.ok()) {
                result.Update(safety_controller.setpoint_override());
            }
            
            // Get Safety Controller State
            SafetyControllerState safety_controller_state = safety_controller.get_state();
            safety_stop = safety_controller.safety_stop();
            if(safety_stop) {
                // Set to Damping Mode:
                control_mode = ControlMode::Damping;
                return interface::constants::controller::damping_motor_command;
            }

            // Create Motor Command: Cast to float for motor controller:
            std::array<float, osc::constants::model::nu_size> q_setpoint;
            std::array<float, osc::constants::model::nu_size> qd_setpoint;
            std::array<float, osc::constants::model::nu_size> torque_feedforward;
            std::array<float, osc::constants::model::nu_size> stiffness;
            std::array<float, osc::constants::model::nu_size> damping;

            Eigen::Map<MotorVector<float>>(q_setpoint.data()) = safety_controller_state.position_setpoint.cast<float>();
            Eigen::Map<MotorVector<float>>(qd_setpoint.data()) = safety_controller_state.velocity_setpoint.cast<float>();
            Eigen::Map<MotorVector<float>>(torque_feedforward.data()) = safety_controller_state.torque_command.cast<float>();
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

                // Get Solution to get Joint Accelerations and Torques:
                osc::aliases::OptimizationSolution solution = operational_space_controller.get_solution();
                MotorVector<double> joint_accelerations = Eigen::Map<MotorVector<double>>(
                    solution(Eigen::seqN(0, osc::constants::optimization::dv_size)).data()
                );
                MotorVector<double> torque_command = Eigen::Map<MotorVector<double>>(
                    solution(Eigen::seqN(osc::constants::optimization::dv_idx, osc::constants::optimization::u_size)).data()
                );

                // Integrate to get velocity setpoints:
                MotorVector<double> velocity_desired = state.motor_velocity + joint_accelerations * timestep;
                MotorVector<double> velocity_setpoint = alpha * velocity_desired + (1.0 - alpha) * previous_motor_velocity;
                previous_motor_velocity = state.motor_velocity;

                // Create Motor Command:
                unitree::containers::MotorCommand motor_command;
                switch(control_mode) {
                    case ControlMode::Damping:
                        motor_command = interface::constants::controller::damping_motor_command;
                        break;
                    case ControlMode::Default:
                        motor_command = default_motor_command(stiffness_value, damping_value);
                        stiffness_value += stiffness_delta;
                        damping_value += damping_delta;
                        break;
                    case ControlMode::OperationalSpaceController:
                        motor_command = update_motor_command(torque_command, velocity_setpoint);
                        break;
                }

                // Send Motor Command:
                unitree_driver->update_command(motor_command);

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
