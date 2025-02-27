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
#include "unitree-api/lowlevelapi_types.h"

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
    using ContactMask = Eigen::Vector<double, constants::model::contact_site_ids_size>;
}

struct OperationalSpaceControllerArgs {
    std::filesystem::path xml_path;
    int control_rate = 1000;
    osqp::OsqpSettings osqp_settings = osqp::OsqpSettings();
};

struct MockMotorControllerArgs {
    std::filesystem::path xml_path;
    int control_rate = 2000;
};

// Data Struct for compatibility with Mujoco:
struct LowState {
    std::array<float, 4> foot_force = { 0 };
};

class MockMotorController {
    public:
        MockMotorController(int control_rate) : control_rate_us(control_rate) {}
        ~MockMotorController() {}

        // Mujoco Model and Data public for visualization and testing:
        mjModel* mj_model;
        mjData* mj_data;

        absl::Status initialize(const std::filesystem::path xml_path) {
            char error[1000];
            mj_model = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
            if( !mj_model ) {
                printf("%s\n", error);
                return absl::InternalError("Failed to load Mujoco Model");
            }

            // Physics timestep:
            double timestep = std::chrono::duration<double>(std::chrono::microseconds(control_rate_us)).count();
            
            // Debug Message:
            std::cout << "Timestep: " << timestep << std::endl;

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
                return absl::FailedPreconditionError("Motor Controller not initialized");

            thread = std::thread(&MockMotorController::control_loop, this);
            control_thread_initialized = true;
            return absl::OkStatus();
        }

        absl::Status stop_control_thread() {
            if(!initialized || !control_thread_initialized)
                return absl::FailedPreconditionError("Motor Controller not initialized");

            running = false;
            thread.join();
            return absl::OkStatus();
        }

        absl::Status clean_up() {
            if(!initialized)
                return absl::FailedPreconditionError("Motor Controller not initialized. Nothing to clean up.");

            mj_deleteData(mj_data);
            mj_deleteModel(mj_model);
            return absl::OkStatus();
        }

        void update_command(const lowleveltypes::MotorCommand& motor_cmd) {
            std::lock_guard<std::mutex> lock(mutex);
            for(const auto& [key, value] : lowleveltypes::MotorID) {
                float q_setpoint = std::clamp(motor_cmd.q_setpoint[value], motor_limits.q_lb[value], motor_limits.q_ub[value]);
                float qd_setpoint = std::clamp(motor_cmd.qd_setpoint[value], motor_limits.qd_lb[value], motor_limits.qd_ub[value]);
                float torque_feedforward = std::clamp(motor_cmd.torque_feedforward[value], motor_limits.tau_lb[value], motor_limits.tau_ub[value]);
                float stiffness = std::clamp(motor_cmd.stiffness[value], 0.0f, 100.0f);
                float damping = std::clamp(motor_cmd.damping[value], 0.0f, 100.0f);
                float kp = std::clamp(motor_cmd.kp[value], 0.0f, 100.0f);
                float kd = std::clamp(motor_cmd.kd[value], 0.0f, 100.0f);
                motor_commands.q_setpoint[value] = q_setpoint;
                motor_commands.qd_setpoint[value] = qd_setpoint; 
                motor_commands.torque_feedforward[value] = torque_feedforward;
                motor_commands.stiffness[value] = stiffness;
                motor_commands.damping[value] = damping;
                motor_commands.kp[value] = kp;
                motor_commands.kd[value] = kd;
            }
        }

        LowState get_low_state() {
            LowState low_state;
            for(int i = 0; i < 4; i++) {
                auto contact = mj_data->contact[i];
                low_state.foot_force[i] = static_cast<float>(contact.dist);
            }
            return low_state;
        }

        lowleveltypes::IMUState get_imu_state() {
            lowleveltypes::IMUState imu_state;
            constexpr int quaternion_start = 3;
            constexpr int quaternion_size = 4;
            constexpr int gyroscope_start = 3;
            constexpr int vector3_size = 3;
            for(int i = 0; i < quaternion_size; i++) {
                imu_state.quaternion[i] = static_cast<float>(mj_data->qpos[quaternion_start + i]);
            }
            for(int i = 0; i < vector3_size ; i++) {
                imu_state.gyroscope[i] = static_cast<float>(mj_data->qvel[gyroscope_start + i]);
                // Unused
                imu_state.accelerometer[i] = 0.0f;
                imu_state.rpy[i] = 0.0f;
            }
            return imu_state;
        }

        lowleveltypes::MotorState get_motor_state() {
            lowleveltypes::MotorState motor_state;
            constexpr int position_offset = 7;
            constexpr int velocity_offset = 6;
            for(int i = 0; i < constants::model::nu_size; i++) {
                motor_state.q[i] = static_cast<float>(mj_data->qpos[position_offset + i]);
                motor_state.qd[i] = static_cast<float>(mj_data->qvel[velocity_offset + i]);
                motor_state.torque_estimate[i] = mj_data->qfrc_actuator[velocity_offset + i];
                // Unused
                motor_state.qdd[i] = 0.0f;
            }
            return motor_state;
        }

        private:
        // Motor Structs:
        const struct {
            std::array<float, lowleveltypes::num_motors> q_lb = {
                -1.0472, -1.5708, -2.7227,
                -1.0472, -1.5708, -2.7227,
                -1.0472, -0.5236, -2.7227,
                -1.0472, -0.5236, -2.7227
            };
            std::array<float, lowleveltypes::num_motors> q_ub = {
                1.0472, 3.4907, -0.83776,
                1.0472, 3.4907, -0.83776,
                1.0472, 4.5379, -0.83776,
                1.0472, 4.5379, -0.83776,
            };
            std::array<float, lowleveltypes::num_motors> qd_lb = {
                -10.0, -10.0, -10.0,
                -10.0, -10.0, -10.0,
                -10.0, -10.0, -10.0,
                -10.0, -10.0, -10.0
            };
            std::array<float, lowleveltypes::num_motors> qd_ub = {
                10.0, 10.0, 10.0,
                10.0, 10.0, 10.0,
                10.0, 10.0, 10.0,
                10.0, 10.0, 10.0
            };
            std::array<float, lowleveltypes::num_motors> tau_lb = {
                -23.7, -23.7, -45.3,
                -23.7, -23.7, -45.3,
                -23.7, -23.7, -45.3,
                -23.7, -23.7, -45.3
            };
            std::array<float, lowleveltypes::num_motors> tau_ub = {
                23.7, 23.7, 45.3,
                23.7, 23.7, 45.3,
                23.7, 23.7, 45.3,
                23.7, 23.7, 45.3
            };
        } motor_limits;
        lowleveltypes::MotorCommand motor_commands;
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
                        float torque_input = torque_feedforward + motor_commands.kp[i] * (q_error) + motor_commands.kd[i] * (qd_error);
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
                    std::cout << "Mock Motor Control Loop Execution Time Exceeded Control Rate: " 
                            << overrun.count() << "us" << std::endl;
                    // Reset next execution time to prevent cascading delays
                    next_execution_time = now;
                }
            }
        } 
};

class UnitreeGo2Interface {
    public:
        UnitreeGo2Interface(OperationalSpaceControllerArgs osc_args, MockMotorControllerArgs mc_args) : 
            operational_space_controller(osc_args.control_rate, osc_args.osqp_settings),
            motor_controller(mc_args.control_rate), 
            xml_path(osc_args.xml_path),
            mock_robot_model_xml_path(mc_args.xml_path),
            control_rate_us(mc_args.control_rate) {}
        ~UnitreeGo2Interface() {}

        /* Operational Space Controller and Motor Controller */
        OperationalSpaceController operational_space_controller;
        MockMotorController motor_controller;

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
            if(!result.ok())
                return result;

            operational_space_controller_initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_motor_controller() {
            absl::Status result;
            result.Update(motor_controller.initialize(mock_robot_model_xml_path));
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
            absl::Status result;
            result.Update(stop_control_thread());
            result.Update(stop_child_threads());
            if(!result.ok())
                return result;

            return absl::OkStatus();
        }

        absl::Status clean_up() {
            absl::Status result;
            result.Update(operational_space_controller.clean_up());
            result.Update(motor_controller.clean_up());     
            if(!result.ok())
                return result;

            return absl::OkStatus();
        }

        absl::Status update_taskspace_targets(const TaskspaceTargetsMatrix& new_taskspace_targets) {
            if(!operational_space_controller_initialized)
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
        State initial_state;
        bool operational_space_controller_initialized = false;
        bool motor_controller_initialized = false;
        const std::filesystem::path xml_path;
        const std::filesystem::path mock_robot_model_xml_path;
        const int control_rate_us; // This should match the control rate of the motor controller.
        /* Index mappings for Mock Robot and Mujoco match */
        const std::array<int, constants::model::nu_size> motor_idx_map{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        const std::array<int, 4> foot_idx_map{0, 1, 2, 3};
        const float contact_threshold = 1e-3;
        /* Thread Variables */
        std::atomic<bool> running{true};
        std::thread thread;
        std::mutex mutex;
        bool control_thread_initialized = false;

        absl::Status update_state() {
            // Get Current State for Unitree Go2 Motor Driver:
            LowState low_state = motor_controller.get_low_state();
            lowleveltypes::IMUState imu_state = motor_controller.get_imu_state();
            lowleveltypes::MotorState motor_state = motor_controller.get_motor_state();

            // Create contact mask:
            ContactMask contact_mask = ContactMask::Zero();
            Eigen::Vector<float, 4> foot_force = Eigen::Map<Eigen::Vector<float, 4>>(low_state.foot_force.data())(foot_idx_map);
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
                Using Mock Controller that only has access to my PD Loop.
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
            std::array<float, constants::model::nu_size> damping = { 0 };
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

                // Create Motor Command: (No Lock Needed)
                lowleveltypes::MotorCommand motor_command = update_motor_command(torque_command);

                // Send Motor Command: (Motor Controller Locks this)
                motor_controller.update_command(motor_command);

                // Check for overrun and sleep until next time:
                auto now = Clock::now();
                if(now < next_time) {
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
