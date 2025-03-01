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

#include "interface/unitree_go2/logger.h"

#include "operational-space-control/unitree_go2/operational_space_controller.h"
#include "operational-space-control/unitree_go2/autogen/autogen_defines.h"
#include "unitree-api/lowlevelapi.h"
#include "unitree-api/lowlevelapi_types.h"


namespace {
    using TaskspaceTargetsMatrix = Eigen::Matrix<double, constants::model::site_ids_size, 6, Eigen::RowMajor>;
    using ActuatorCommand = Eigen::Vector<double, constants::model::nu_size>;
    using ActuatorCommandFloat = Eigen::Vector<float, constants::model::nu_size>;
    using MotorVector = Eigen::Vector<double, constants::model::nu_size>;
    using MotorVectorFloat = Eigen::Vector<float, constants::model::nu_size>;
    using Quaternion = Eigen::Vector<double, 4>;
    using QuaternionFloat = Eigen::Vector<float, 4>;
    using Vector3 = Eigen::Vector<double, 3>;
    using Vector3Float = Eigen::Vector<float, 3>;
    using ContactMask = Eigen::Vector<double, constants::model::contact_site_ids_size>;
    using OptimizationSolution = Eigen::Vector<double, constants::optimization::design_vector_size>;

    template <typename T> int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

    template<typename Derived>
    void clamp_vector(Eigen::MatrixBase<Derived>& vector, typename Derived::Scalar min, typename Derived::Scalar max) {
        vector = vector.cwiseMin(max).cwiseMax(min);
    }

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
        UnitreeGo2Interface(OperationalSpaceControllerArgs osc_args, MotorControllerArgs mc_args, StateLoggerArgs log_args) : 
            operational_space_controller(osc_args.control_rate, osc_args.osqp_settings),
            motor_controller(mc_args.control_rate),
            logger(log_args.log_filepath, log_args.logging_rate),
            enable_logger(log_args.enable_logging),
            xml_path(osc_args.xml_path),
            network_name(mc_args.network_name),
            control_rate_us(mc_args.control_rate) {}
        ~UnitreeGo2Interface() {}

        absl::Status initialize() {
            // Initialize Motor Controller and Operational Space Controller:
            absl::Status result;
            result.Update(initialize_motor_controller());
            result.Update(initialize_operational_space_controller());

            // Initialize Logger:
            if(enable_logger)
                result.Update(logger.initialize());

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
            result.Update(operational_space_controller.initialize_control_thread());
            result.Update(motor_controller.initialize_control_thread());
            result.Update(initialize_control_thread());
            if(enable_logger)
                result.Update(logger.initialize_log_thread());

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

        absl::Status stop_threads() {
            absl::Status result;
            result.Update(stop_control_thread());
            result.Update(logger.stop_log_thread());
            result.Update(operational_space_controller.stop_control_thread());
            result.Update(motor_controller.stop_control_thread());
            if(!result.ok())
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
            
            LOG(INFO) << "Activating Operational Space Controller";
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

        ActuatorCommand get_torque_command() {
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
        StateLogger logger;
        State initial_state;
        bool enable_logger;
        bool operational_space_controller_initialized = false;
        bool motor_controller_initialized = false;
        const std::filesystem::path xml_path;
        const std::string network_name;
        const int control_rate_us; // This should match the control rate of the motor controller.
        /* Index mappings for Robot and Mujoco Model: mj_model : [FL FR Hl HR] | robot : [FR FL HR HL] */
        const std::array<int, constants::model::nu_size> motor_idx_map{3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
        const std::array<int, 4> foot_idx_map{1, 0, 3, 2};
        const short contact_threshold = 5;
        float stiffness_value = 5.0;
        float damping_value = 5.0;
        float stiffness_delta = 0.01;
        float damping_delta = 0.0;
        /* Approximate Linear Body Velocity */
        float alpha = 0.9;
        Vector3Float previous_smooth_acceleration = Vector3Float::Zero();
        Vector3Float smooth_acceleration = Vector3Float::Zero();
        Vector3Float previous_linear_body_velocity = Vector3Float::Zero();
        Vector3Float linear_body_velocity = Vector3Float::Zero();
        Vector3Float previous_smooth_velocity = Vector3Float::Zero();
        Vector3Float smooth_velocity = Vector3Float::Zero();
        /* Thread Variables */
        std::atomic<bool> running{true};
        std::atomic<bool> activate_control{false};
        std::thread thread;
        std::mutex mutex;
        bool control_thread_initialized = false;
        /* Safety Controller Variables */
        double kp_lb = 2.0;
        double kp_ub = 20.0;
        double kd_lb = 2.0;
        double kd_ub = 10.0;
        // Position Soft and Hard Limits:
        std::array<double, constants::model::nu_size> lower_lb = {
            -0.546, -0.5708, -1.7227,
            -0.546, -0.5708, -1.7227,
            -0.546, -0.0, -1.7227,
            -0.546, -0.0, -1.7227
        };
        std::array<double, constants::model::nu_size> lower_ub = {
            -0.8472, -1.3708, -2.5227,
            -0.8472, -1.3708, -2.5227,
            -0.8472, -0.3236, -2.5227,
            -0.8472, -0.3236, -2.5227
        };
        std::array<double, constants::model::nu_size> upper_lb = {
            0.546, 2.4907, 0.162,
            0.546, 2.4907, 0.162,
            0.546, 3.5379, 0.162,
            0.546, 3.5379, 0.162,
        };
        std::array<double, constants::model::nu_size> upper_ub = {
            0.8472, 3.2907, -0.63776,
            0.8472, 3.2907, -0.63776,
            0.8472, 4.3379, -0.63776,
            0.8472, 4.3379, -0.63776,
        };
        // Velocity Soft and Hard Limits:
        std::array<double, constants::model::nu_size> velocity_lb = {
            std::numbers::pi, std::numbers::pi, std::numbers::pi,
            std::numbers::pi, std::numbers::pi, std::numbers::pi,
            std::numbers::pi, std::numbers::pi, std::numbers::pi,
            std::numbers::pi, std::numbers::pi, std::numbers::pi,
        };
        std::array<double, constants::model::nu_size> velocity_ub = {
            2 * std::numbers::pi, 2 * std::numbers::pi, 2 * std::numbers::pi,
            2 * std::numbers::pi, 2 * std::numbers::pi, 2 * std::numbers::pi,
            2 * std::numbers::pi, 2 * std::numbers::pi, 2 * std::numbers::pi,
            2 * std::numbers::pi, 2 * std::numbers::pi, 2 * std::numbers::pi,
        };
        // Torque Saturation Limits:
        float torque_ub = 10.0;
        float torque_lb = -10.0;


        absl::Status initialize_filter() {
            // Initialize Filter:
            lowleveltypes::IMUState imu_state = motor_controller.get_imu_state();

            // Reformat data to match Mujoco Model:
            Vector3Float linear_body_acceleration = Eigen::Map<Vector3Float>(imu_state.accelerometer.data());

            // Initialize Previous Values:
            previous_smooth_acceleration = linear_body_acceleration;
            previous_linear_body_velocity = Vector3Float::Zero();
            previous_smooth_velocity = Vector3Float::Zero();

            return absl::OkStatus();
        }

        absl::Status update_state() {
            // Get Current State for Unitree Go2 Motor Driver:
            lowleveltypes::LowState low_state = motor_controller.get_low_state();
            lowleveltypes::IMUState imu_state = motor_controller.get_imu_state();
            lowleveltypes::MotorState motor_state = motor_controller.get_motor_state();

            // Create contact mask:
            ContactMask contact_mask = ContactMask::Zero();
            Eigen::Vector<short, 4> foot_force = Eigen::Map<Eigen::Vector<short, 4>>(low_state.foot_force.data())(foot_idx_map);
            for(int i = 0; i < 4; i++) {
                contact_mask(i) = foot_force(i) > contact_threshold;
            }

            // Reformat data to match Mujoco Model: 
            MotorVectorFloat motor_position = Eigen::Map<MotorVectorFloat>(motor_state.q.data())(motor_idx_map);
            MotorVectorFloat motor_velocity = Eigen::Map<MotorVectorFloat>(motor_state.qd.data())(motor_idx_map);
            MotorVectorFloat motor_acceleration = Eigen::Map<MotorVectorFloat>(motor_state.qdd.data())(motor_idx_map);
            MotorVectorFloat motor_torque_estimate = Eigen::Map<MotorVectorFloat>(motor_state.torque_estimate.data())(motor_idx_map);
            QuaternionFloat body_rotation = Eigen::Map<QuaternionFloat>(imu_state.quaternion.data());
            Vector3Float angular_body_velocity = Eigen::Map<Vector3Float>(imu_state.gyroscope.data());
            Vector3Float linear_body_acceleration = Eigen::Map<Vector3Float>(imu_state.accelerometer.data());

            // Unitree does not provide linear velocity:
            smooth_acceleration = alpha * linear_body_acceleration + (1 - alpha) * previous_smooth_acceleration;
            linear_body_velocity = previous_linear_body_velocity + smooth_acceleration * control_rate_us * 1.0e-6f;
            smooth_velocity = alpha * linear_body_velocity + (1 - alpha) * previous_smooth_velocity;

            state.motor_position = motor_position.cast<double>();
            state.motor_velocity = motor_velocity.cast<double>();
            state.motor_acceleration = motor_acceleration.cast<double>();
            state.torque_estimate = motor_torque_estimate.cast<double>();
            state.body_rotation = body_rotation.cast<double>();
            state.angular_body_velocity = angular_body_velocity.cast<double>();
            state.linear_body_velocity = smooth_velocity.cast<double>();
            state.linear_body_acceleration = linear_body_acceleration.cast<double>();
            state.contact_mask = contact_mask;

            return absl::OkStatus();
        }

        // TODO(jeh15): Log these values.
        ActuatorCommandFloat safety_controller(ActuatorCommandFloat& torque_command) {
            /*
                Saturates control input if past soft constraint 
                and terminates the process if past the hard constraint.
            */
            Eigen::Vector<double, constants::model::nu_size> position_command = Eigen::Vector<double, constants::model::nu_size>::Zero();
            Eigen::Vector<double, constants::model::nu_size> velocity_command = Eigen::Vector<double, constants::model::nu_size>::Zero();
            double kp = 0.0;
            double kd = 0.0;
            for(int i = 0; i < constants::model::nu_size; i++){
                double motor_position = state.motor_position[i];
                double motor_velocity = state.motor_velocity[i];
                if(motor_position > upper_lb[i]) {
                    ABSL_CHECK(motor_position < upper_ub[i]) << "Motor Position Exceeded Upper Bound";
                    kp = kp_lb + (abs(motor_position) - abs(upper_lb[i])) * (kp_ub - kp_lb) / ( abs(upper_ub[i]) - abs(upper_lb[i]));
                    position_command(i) = kp * (upper_lb[i] - motor_position);
                }
                else if(motor_position < lower_lb[i]) {
                    ABSL_CHECK(motor_position < upper_ub[i]) << "Motor Position Exceeded Lower Bound";
                    kp = kp_lb + (abs(motor_position) - abs(lower_lb[i])) * (kp_ub - kp_lb) / ( abs(lower_ub[i]) - abs(lower_lb[i]));
                    position_command(i) = kp * (lower_lb[i] - motor_position);
                }

                if(abs(motor_velocity) > velocity_lb[i]) {
                    ABSL_CHECK(abs(motor_velocity) < velocity_ub[i]) << "Motor Velocity Exceeded Limit";
                    kd = kd_lb + (motor_velocity - velocity_lb[i]) * (kd_ub - kd_lb) / (velocity_ub[i] - velocity_lb[i]);
                    double velocity_setpoint = sgn<double>(motor_velocity) * velocity_lb[i];
                    velocity_command(i) = kd * (velocity_setpoint - motor_velocity);
                }
            }

            // Add Safety Controller to Torque Command:
            ActuatorCommandFloat position_command_f = position_command.cast<float>();
            ActuatorCommandFloat velocity_command_f = velocity_command.cast<float>();
            torque_command = torque_command + position_command_f + velocity_command_f;
            
            // Saturate Torque Command:
            clamp_vector(torque_command , torque_lb, torque_ub);

            return torque_command;
        }

        lowleveltypes::MotorCommand update_motor_command(
            ActuatorCommandFloat& torque_command,
            const ActuatorCommandFloat& velocity_setpoint = ActuatorCommandFloat::Zero(),
            const ActuatorCommandFloat& position_setpoint = ActuatorCommandFloat::Zero(),
            const float stiffness_value = 0.0,
            const float damping_value = 5.0
        ) {
            /*
                Motor Command Struct:
                
                Turning off position based feedback terms.
                Using velocity feedback terms for damping.
                Only using built-in Unitree Control Loop.
            */

            // Run safety controller on torque command:
            torque_command = safety_controller(torque_command);

            std::array<float, constants::model::nu_size> q_setpoint;
            for(int i = 0; i < constants::model::nu_size; i++) {
                q_setpoint[i] = position_setpoint(i);
            }
            std::array<float, constants::model::nu_size> qd_setpoint;
            for(int i = 0; i < constants::model::nu_size; i++) {
                qd_setpoint[i] = velocity_setpoint(i);
            }
            std::array<float, constants::model::nu_size> torque_feedforward;
            for(int i = 0; i < constants::model::nu_size; i++) {
                torque_feedforward[i] = torque_command(i);
            }
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

                    if(enable_logger)
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
                lowleveltypes::MotorCommand motor_command;
                if(activate_control) {
                    motor_command = update_motor_command(torque_command_f);
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
