#pragma once

#include <iostream>
#include <string>
#include <array>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <fstream>
#include <memory>

#include "absl/status/status.h"
#include "absl/log/log.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/log/initialize.h"
#include "absl/strings/string_view.h"
#include "absl/log/absl_check.h"

#include "Eigen/Dense"
#include "osqp++.h"

#include "operational-space-control/unitree_go2/operational_space_controller.h"
#include "operational-space-control/unitree_go2/autogen/autogen_defines.h"
#include "unitree-api/containers.h"

#include "mujoco/mujoco.h"


namespace {
    using TaskspaceTargetsMatrix = Eigen::Matrix<double, constants::model::site_ids_size, 6, Eigen::RowMajor>;
    using TorqueCommand = Eigen::Vector<double, constants::model::nu_size>;
    using MotorVector = Eigen::Vector<double, constants::model::nu_size>;
    using MotorVectorFloat = Eigen::Vector<float, constants::model::nu_size>;
    using Quaternion = Eigen::Vector<double, 4>;
    using QuaternionFloat = Eigen::Vector<float, 4>;
    using Vector3 = Eigen::Vector<double, 3>;
    using Vector3Float = Eigen::Vector<float, 3>;
}

struct MockUnitreeDriverArgs {
    std::filesystem::path xml_path;
    int control_rate_us = 2000;
};


class MockUnitreeDriver {
    public:
        MockUnitreeDriver(std::filesystem::path xml_path, int control_rate_us) : xml_path(xml_path), control_rate_us(control_rate_us) {}
        ~MockUnitreeDriver() {}

        // Mujoco Model and Data public for visualization and testing:
        mjModel* mj_model;
        mjData* mj_data;

        absl::Status initialize() {
            char error[1000];
            mj_model = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
            if( !mj_model ) {
                printf("%s\n", error);
                return absl::InternalError("Failed to load Mujoco Model");
            }

            // Physics timestep:
            double timestep = std::chrono::duration<double>(std::chrono::microseconds(control_rate_us)).count();

            mj_model->opt.timestep = timestep;
            mj_data = mj_makeData(mj_model);

            // Initialize mj_data:
            mj_data->qpos = mj_model->key_qpos;
            mj_data->qvel = mj_model->key_qvel;
            mj_data->ctrl = mj_model->key_ctrl;

            mj_forward(mj_model, mj_data);

            initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_control_thread() {
            if(!initialized)
                return absl::FailedPreconditionError("Unitree Driver not initialized");

            thread = std::thread(&MockUnitreeDriver::control_loop, this);
            control_thread_initialized = true;
            return absl::OkStatus();
        }

        absl::Status stop_control_thread() {
            if(!initialized || !control_thread_initialized)
                return absl::FailedPreconditionError("Unitree Driver not initialized");

            running = false;
            thread.join();
            return absl::OkStatus();
        }

        absl::Status clean_up() {
            if(!initialized)
                return absl::FailedPreconditionError("Unitree Driver not initialized. Nothing to clean up.");

            mj_deleteData(mj_data);
            mj_deleteModel(mj_model);
            return absl::OkStatus();
        }

        void update_command(const unitree::containers::MotorCommand& motor_cmd) {
            std::lock_guard<std::mutex> lock(mutex);
            for(const auto& [key, value] : unitree::containers::MotorID) {
                float q_setpoint = std::clamp(motor_cmd.q_setpoint[value], motor_limits.q_lb[value], motor_limits.q_ub[value]);
                float qd_setpoint = std::clamp(motor_cmd.qd_setpoint[value], motor_limits.qd_lb[value], motor_limits.qd_ub[value]);
                float torque_feedforward = std::clamp(motor_cmd.torque_feedforward[value], motor_limits.tau_lb[value], motor_limits.tau_ub[value]);
                float stiffness = std::clamp(motor_cmd.stiffness[value], 0.0f, 100.0f);
                float damping = std::clamp(motor_cmd.damping[value], 0.0f, 100.0f);
                motor_commands.q_setpoint[value] = q_setpoint;
                motor_commands.qd_setpoint[value] = qd_setpoint; 
                motor_commands.torque_feedforward[value] = torque_feedforward;
                motor_commands.stiffness[value] = stiffness;
                motor_commands.damping[value] = damping;
            }
        }

        unitree::containers::LowState get_low_state() {
            unitree::containers::LowState low_state;
            float distance_threshold = 1.0e-3;
            for(int i = 0; i < 4; i++) {
                auto contact = mj_data->contact[i];
                if(contact.dist < distance_threshold) {
                    low_state.foot_force[i] = 20;
                }
                else {
                    low_state.foot_force[i] = 0;
                }
            }
            return low_state;
        }

        unitree::containers::IMUState get_imu_state() {
            unitree::containers::IMUState imu_state;
            constexpr int quaternion_start = 3;
            constexpr int quaternion_size = 4;
            constexpr int gyroscope_start = 3;
            constexpr int vector3_size = 3;
            for(int i = 0; i < quaternion_size; i++) {
                imu_state.quaternion[i] = static_cast<float>(mj_data->qpos[quaternion_start + i]);
            }
            for(int i = 0; i < vector3_size ; i++) {
                imu_state.gyroscope[i] = static_cast<float>(mj_data->qvel[gyroscope_start + i]);
                imu_state.accelerometer[i] = static_cast<float>(mj_data->qacc[i]);
                // Unused
                imu_state.rpy[i] = 0.0f;
            }
            return imu_state;
        }

        unitree::containers::MotorState get_motor_state() {
            unitree::containers::MotorState motor_state;
            constexpr int position_offset = 7;
            constexpr int velocity_offset = 6;
            for(int i = 0; i < constants::model::nu_size; i++) {
                motor_state.q[i] = static_cast<float>(mj_data->qpos[position_offset + i]);
                motor_state.qd[i] = static_cast<float>(mj_data->qvel[velocity_offset + i]);
                motor_state.qdd[i] = static_cast<float>(mj_data->qacc[velocity_offset + i]);
                motor_state.torque_estimate[i] = mj_data->qfrc_actuator[velocity_offset + i];
            }
            return motor_state;
        }

        int get_control_rate() {
            return control_rate_us;
        }

        bool is_initialized() {
            return initialized;
        }

        bool is_control_thread_initialized() {
            return control_thread_initialized;
        }

        private:
        std::filesystem::path xml_path;
        // Motor Structs:
        const struct {
            std::array<float, unitree::containers::num_motors> q_lb = {
                -1.0472, -1.5708, -2.7227,
                -1.0472, -1.5708, -2.7227,
                -1.0472, -0.5236, -2.7227,
                -1.0472, -0.5236, -2.7227
            };
            std::array<float, unitree::containers::num_motors> q_ub = {
                1.0472, 3.4907, -0.83776,
                1.0472, 3.4907, -0.83776,
                1.0472, 4.5379, -0.83776,
                1.0472, 4.5379, -0.83776,
            };
            std::array<float, unitree::containers::num_motors> qd_lb = {
                -10.0, -10.0, -10.0,
                -10.0, -10.0, -10.0,
                -10.0, -10.0, -10.0,
                -10.0, -10.0, -10.0
            };
            std::array<float, unitree::containers::num_motors> qd_ub = {
                10.0, 10.0, 10.0,
                10.0, 10.0, 10.0,
                10.0, 10.0, 10.0,
                10.0, 10.0, 10.0
            };
            std::array<float, unitree::containers::num_motors> tau_lb = {
                -23.7, -23.7, -45.3,
                -23.7, -23.7, -45.3,
                -23.7, -23.7, -45.3,
                -23.7, -23.7, -45.3
            };
            std::array<float, unitree::containers::num_motors> tau_ub = {
                23.7, 23.7, 45.3,
                23.7, 23.7, 45.3,
                23.7, 23.7, 45.3,
                23.7, 23.7, 45.3
            };
        } motor_limits;
        unitree::containers::MotorCommand motor_commands;
        // Control Thread:
        int control_rate_us;
        std::atomic<bool> running{true};
        std::mutex mutex;
        std::thread thread;
        bool initialized = false;
        bool control_thread_initialized = false;

        void control_loop() {
            using Clock = std::chrono::steady_clock;
            auto next_execution_time = Clock::now();
            // Thread Loop:
            while(running) {
                // Calculate next execution time first
                next_execution_time += std::chrono::microseconds(control_rate_us);

                /* Lock Guard Scope */
                {   
                    std::lock_guard<std::mutex> lock(mutex);
                    // Iterate over motors and update mj_data ctrl:
                    constexpr int position_offset = 7;
                    constexpr int velocity_offset = 6;
                    for(int i = 0; i < constants::model::nu_size; i++) {
                        float q_error = motor_commands.q_setpoint[i] - static_cast<float>(mj_data->qpos[position_offset + i]);
                        float qd_error = motor_commands.qd_setpoint[i] - static_cast<float>(mj_data->qvel[velocity_offset + i]);
                        float torque_feedforward = std::clamp(motor_commands.torque_feedforward[i], motor_limits.tau_lb[i], motor_limits.tau_ub[i]);
                        float torque_input = torque_feedforward + motor_commands.stiffness[i] * (q_error) + motor_commands.damping[i] * (qd_error);
                        float torque_cmd = std::clamp(torque_input, motor_limits.tau_lb[i], motor_limits.tau_ub[i]);
                        mj_data->ctrl[i] = torque_cmd;
                    }

                    // Step Simulation Model:
                    mj_step(mj_model, mj_data);
                }
                // Check for overrun and sleep until next execution time
                auto now = Clock::now();
                if(now < next_execution_time) {
                    std::this_thread::sleep_until(next_execution_time);
                } 
                else {
                    // Log overrun
                    auto overrun = std::chrono::duration_cast<std::chrono::microseconds>(now - next_execution_time);
                    std::cout << "Mock Unitree Driver Loop Execution Time Exceeded Control Rate: " 
                            << overrun.count() << "us" << std::endl;
                    // Reset next execution time to prevent cascading delays
                    next_execution_time = now;
                }
            }
        } 
};
