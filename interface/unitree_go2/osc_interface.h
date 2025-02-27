#pragma once

#include <iostream>
#include <string>
#include <array>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"

#include "Eigen/Dense"
#include "osqp++.h"

#include "operational-space-control/unitree_go2/operational_space_controller.h"
#include "operational-space-control/unitree_go2/autogen/autogen_defines.h"
#include "unitree-api/lowlevelapi.h"
#include "unitree-api/lowlevelapi_types.h"


namespace {
    using TaskspaceTargetsMatrix = Eigen::Matrix<double, constants::model::site_ids_size, 6, Eigen::RowMajor>;
    using TorqueCommand = Eigen::Vector<double, constants::model::nu_size>;
    using MotorVector = Eigen::Vector<double, constants::model::nu_size>;
    using MotorVectorFloat = Eigen::Vector<float, constants::model::nu_size>;
    using Quaternion = Eigen::Vector<double, 4>;
    using QuaternionFloat = Eigen::Vector<float, 4>;
    using Vector3 = Eigen::Vector<double, 3>;
    using Vector3Float = Eigen::Vector<float, 3>;
    using ContactMask = Eigen::Vector<double, constants::model::contact_site_ids_size>;
}

struct OperationalSpaceControllerArgs {
    std::filesystem::path xml_path;
    int control_rate = 1000;
    osqp::OsqpSettings osqp_settings = osqp::OsqpSettings();
};

struct MotorControllerArgs {
    std::string network_name;
    int control_rate = 2000;
};

class UnitreeGo2Interface {
    public:
        UnitreeGo2Interface(OperationalSpaceControllerArgs osc_args, MotorControllerArgs mc_args) : 
            operational_space_controller(osc_args.control_rate, osc_args.osqp_settings),
            motor_controller(mc_args.control_rate), 
            xml_path(osc_args.xml_path),
            network_name(mc_args.network_name),
            control_rate_us(mc_args.control_rate) {}
        ~UnitreeGo2Interface() {}

        absl::Status initialize() {
            // Initialize Motor Controller and Operational Space Controller:
            absl::Status result;
            result.Update(initialize_motor_controller());
            result.Update(initialize_operational_space_controller());
            ABSL_CHECK(result.ok()) << result.message();
            
            return absl::OkStatus();
        }

        absl::Status initialize_operational_space_controller() {
            if(!motor_controller_initialized)
                return absl::FailedPreconditionError("Motor Controller not initialized. Motor Controller needs to be initialized first to set the initial state of the Operational Space Controller.");

            // Load mujoco model and use initial state from the motor controller:
            absl::Status result;
            result.Update(operational_space_controller.initialize(xml_path, initial_state));
            result.Update(operational_space_controller.initialize_optimization());
            if (!result.ok())
                return result;

            operational_space_controller_initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_motor_controller() {
            absl::Status result;
            result.Update(motor_controller.initialize(network_name));
            result.Update(update_state());
            if(!result.ok())
                return result;

            initial_state = get_state();
            motor_controller_initialized = true;

            return absl::OkStatus();
        }

        absl::Status initialize_operational_space_controller_thread() {
            absl::Status result = operational_space_controller.initialize_control_thread();
            if(!result.ok())
                return result;
            
            return absl::OkStatus();
        }

        absl::Status initialize_motor_controller_thread() {
            absl::Status result;
            result.Update(motor_controller.initialize_control_thread());
            if(!result.ok())
                return result;

            return absl::OkStatus();
        }

        absl::Status initialize_control_thread() {
            if(!operational_space_controller_initialized || !motor_controller_initialized)
                return absl::FailedPreconditionError("Operational Space Controller and/or Motor Controller not initialized");
            
            thread = std::thread(&UnitreeGo2Interface::control_loop, this);
            control_thread_initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_threads() {
            // Initialize all threads:
            absl::Status result;
            result.Update(initialize_operational_space_controller_thread());
            result.Update(initialize_motor_controller_thread());
            result.Update(initialize_control_thread());
            ABSL_CHECK(result.ok()) << result.message();

            return absl::OkStatus();
        }

        absl::Status stop_control_thread() {
            if(!control_thread_initialized)
                return absl::FailedPreconditionError("Control Thread not initialized");

            running = false;
            thread.join();
            return absl::OkStatus();
        }

        absl::Status stop_child_threads() {
            absl::Status result;
            result.Update(operational_space_controller.stop_control_thread());
            result.Update(motor_controller.stop_control_thread());
            if(!result.ok())
                return result;

            return absl::OkStatus();
        }

        absl::Status stop_threads() {
            // Stop all threads:
            absl::Status result;
            result.Update(stop_control_thread());
            result.Update(stop_child_threads());
            if (!result.ok())
                return result;

            return absl::OkStatus();
        }

        absl::Status clean_up() {
            absl::Status result;
            result.Update(operational_space_controller.clean_up());    
            if(!result.ok())
                return result;

            return absl::OkStatus();
        }

        absl::Status activate_operational_space_controller() {
            if(!control_thread_initialized)
                return absl::FailedPreconditionError("Control Thread not initialized. Initial Control Commands must come from Default Control.");

            activate_control = true;
            return absl::OkStatus();
        }

        absl::Status update_taskspace_targets(const TaskspaceTargetsMatrix& new_taskspace_targets) {
            if (!operational_space_controller_initialized)
                return absl::FailedPreconditionError("Operational Space Controller not initialized");
            
            std::lock_guard<std::mutex> lock(mutex);
            taskspace_targets = new_taskspace_targets;
            return absl::OkStatus();
        }

        State get_state() {
            std::lock_guard<std::mutex> lock(mutex);
            return state;
        }

        TorqueCommand get_torque_command() {
            std::lock_guard<std::mutex> lock(mutex);
            return operational_space_controller.get_torque_command();
        }

    private:
        /* Shared Variables */
        State state;
        TaskspaceTargetsMatrix taskspace_targets = TaskspaceTargetsMatrix::Zero();
        /* Operational Space Controller and Motor Controller */
        OperationalSpaceController operational_space_controller;
        MotorController motor_controller;
        State initial_state;
        bool operational_space_controller_initialized = false;
        bool motor_controller_initialized = false;
        const std::filesystem::path xml_path;
        const std::string network_name;
        const int control_rate_us; // This should match the control rate of the motor controller.
        /* Index mappings for Robot and Mujoco Model: mj_model : [FL FR Hl HR] | robot : [FR FL HR HL] */
        const std::array<int, constants::model::nu_size> motor_idx_map{3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
        const std::array<int, 4> foot_idx_map{1, 0, 3, 2};
        const short contact_threshold = 24;
        float stiffness_value = 5.0;
        float damping_value = 5.0;
        float stiffness_delta = 0.01;
        float damping_delta = 0.0;
        /* Thread Variables */
        std::atomic<bool> running{true};
        std::atomic<bool> activate_control{false};
        std::thread thread;
        std::mutex mutex;
        bool control_thread_initialized = false;

        absl::Status update_state() {
            // Get Current State for Unitree Go2 Motor Driver:
            lowleveltypes::LowState low_state = motor_controller.get_low_state();
            lowleveltypes::IMUState imu_state = motor_controller.get_imu_state();
            lowleveltypes::MotorState motor_state = motor_controller.get_motor_state();

            // Create contact mask:
            ContactMask contact_mask = ContactMask::Zero();
            Eigen::Vector<short, 4> foot_force = Eigen::Map<Eigen::Vector<short, 4>>(low_state.foot_force.data())(foot_idx_map);
            for(int i = 0; i < 4; i++) {
                contact_mask(i) = foot_force(i) < contact_threshold;
            }

            // Reformat data to match Mujoco Model: 
            MotorVectorFloat motor_position = Eigen::Map<MotorVectorFloat>(motor_state.q.data())(motor_idx_map);
            MotorVectorFloat motor_velocity = Eigen::Map<MotorVectorFloat>(motor_state.qd.data())(motor_idx_map);
            MotorVectorFloat motor_acceleration = Eigen::Map<MotorVectorFloat>(motor_state.qdd.data())(motor_idx_map);
            MotorVectorFloat motor_torque_estimate = Eigen::Map<MotorVectorFloat>(motor_state.torque_estimate.data())(motor_idx_map);
            QuaternionFloat body_rotation = Eigen::Map<QuaternionFloat>(imu_state.quaternion.data());
            Vector3Float body_velocity = Eigen::Map<Vector3Float>(imu_state.gyroscope.data());
            Vector3Float body_acceleration = Eigen::Map<Vector3Float>(imu_state.accelerometer.data());

            state.motor_position = motor_position.cast<double>();
            state.motor_velocity = motor_velocity.cast<double>();
            state.motor_acceleration = motor_acceleration.cast<double>();
            state.torque_estimate = motor_torque_estimate.cast<double>();
            state.body_rotation = body_rotation.cast<double>();
            state.body_velocity = body_velocity.cast<double>();
            state.body_acceleration = body_acceleration.cast<double>();
            state.contact_mask = contact_mask;

            return absl::OkStatus();
        }

        lowleveltypes::MotorCommand update_motor_command(const TorqueCommand& torque_command) {
            /*
                Motor Command Struct:
                
                Turning off position based feedback terms.
                Using velocity feedback terms for damping.
                Only using built-in Unitree Control Loop.
            */
            std::array<float, constants::model::nu_size> q_setpoint = {
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
            };
            std::array<float, constants::model::nu_size> qd_setpoint = { 0 };
            std::array<float, constants::model::nu_size> torque_feedforward;
            for(int i = 0; i < constants::model::nu_size; i++) {
                torque_feedforward[i] = torque_command(i);
            }
            std::array<float, constants::model::nu_size> stiffness = { 0 };
            std::array<float, constants::model::nu_size> damping = {
                5.0, 5.0, 5.0,
                5.0, 5.0, 5.0,
                5.0, 5.0, 5.0,
                5.0, 5.0, 5.0,
            };
            std::array<float, constants::model::nu_size> kp = { 0 };
            std::array<float, constants::model::nu_size> kd = { 0 };
            
            lowleveltypes::MotorCommand motor_command = {
                .q_setpoint = q_setpoint,
                .qd_setpoint = qd_setpoint,
                .torque_feedforward = torque_feedforward,
                .stiffness = stiffness,
                .damping = damping,
                .kp = kp,
                .kd = kd,
            };

            return motor_command;
        }

        lowleveltypes::MotorCommand default_motor_command(const float stiffness_value = 5.0, const float damping_value = 5.0) {
            /*
                Hold default position.
            */

            // Clamp values:
            std::clamp(stiffness_value, 0.0f, 120.0f);
            std::clamp(damping_value, 0.0f, 5.0f);

            std::array<float, constants::model::nu_size> q_setpoint = {
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
            };
            std::array<float, constants::model::nu_size> qd_setpoint = { 0 };
            std::array<float, constants::model::nu_size> torque_feedforward = { 0 };
            std::array<float, constants::model::nu_size> stiffness = { 
                stiffness_value, stiffness_value, stiffness_value,
                stiffness_value, stiffness_value, stiffness_value,
                stiffness_value, stiffness_value, stiffness_value,
                stiffness_value, stiffness_value, stiffness_value,
            };
            std::array<float, constants::model::nu_size> damping = {
                damping_value, damping_value, damping_value,
                damping_value, damping_value, damping_value,
                damping_value, damping_value, damping_value,
                damping_value, damping_value, damping_value,
            };
            std::array<float, constants::model::nu_size> kp = { 
                5.0, 5.0, 5.0,
                5.0, 5.0, 5.0,
                5.0, 5.0, 5.0,
                5.0, 5.0, 5.0,
             };
            std::array<float, constants::model::nu_size> kd = { 
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
             };
            
            lowleveltypes::MotorCommand motor_command = {
                .q_setpoint = q_setpoint,
                .qd_setpoint = qd_setpoint,
                .torque_feedforward = torque_feedforward,
                .stiffness = stiffness,
                .damping = damping,
                .kp = kp,
                .kd = kd,
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
                }

                // Get Torque Command: (OSC Locks this)
                TorqueCommand torque_command = operational_space_controller.get_torque_command()(motor_idx_map);

                // Create Motor Command:
                lowleveltypes::MotorCommand motor_command;
                if(activate_control) {
                    motor_command = update_motor_command(torque_command);
                }
                else {
                    motor_command = default_motor_command(stiffness_value, damping_value);
                    stiffness_value += stiffness_delta;
                    damping_value += damping_delta;
                }

                // Send Motor Command: (Motor Controller Locks this)
                motor_controller.update_command(motor_command);

                // Check for overrun and sleep until next time:
                auto now = Clock::now();
                if (now < next_time) {
                    std::this_thread::sleep_until(next_time);
                } else {
                    // Log overrun:
                    auto overrun = std::chrono::duration_cast<std::chrono::microseconds>(now - next_time);
                    std::cout << "Interface Control Loop Execution Time Exceeded Control Rate: " 
                        << overrun.count() << "us" << std::endl;
                    next_time = now;
                }
            }
        }
};
