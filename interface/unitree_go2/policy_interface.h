#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <numeric>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"

#include <onnxruntime_cxx_api.h>

#include "Eigen/Dense"

#include "unitree-api/unitree_driver.h"
#include "unitree-api/containers.h"

#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"
#include "interface/unitree_go2/constants.h"
#include "interface/unitree_go2/utilities.h"

using namespace interface::aliases::common;
using namespace interface::containers::controller;


namespace {
    template <typename T>
    T vector_product(const std::vector<T>& v) {
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    }
}

//TODO(jeh15): Compile time constants.
template <typename RobotDriver = UnitreeDriver>
class PolicyInterface {
    public:
        PolicyInterface(
            std::filesystem::path onnx_model_path,
            std::shared_ptr<RobotDriver> unitree_driver
        ) : 
            onnx_model_path(onnx_model_path),
            unitree_driver(unitree_driver) {}
        ~PolicyInterface() {}

        absl::Status initialize() {
            absl::Status result;
            // Initialize Unitree Driver:
            if(!unitree_driver->is_initialized())
                result.Update(unitree_driver->initialize());
            // Initialize Policy:
            if(!session_initialized)
                result.Update(initialize_session());
            // Get Initial Position for Get Up Routine:
            if(!initial_position_initialized)
                result.Update(get_initial_position());

            ABSL_CHECK(result.ok()) << result.message();
            
            return absl::OkStatus();
        }

        absl::Status initialize_session() {
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_ptr = std::make_unique<Ort::Session>(*env, onnx_model_path.c_str(), session_options);
            if (!session_ptr) {
                return absl::InternalError("Policy Interface: Failed to create ONNX session");
            }

            // Initialize Inputs and Outputs:
            for (size_t i = 0; i < session_ptr->GetInputCount(); ++i) {
                input_nodes.push_back(session_ptr->GetInputNameAllocated(i, allocator));
                input_names.push_back(input_nodes.back().get());
                Ort::TypeInfo type_info = session_ptr->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                input_types.push_back(tensor_info.GetElementType());
                input_shapes.push_back(tensor_info.GetShape());
            }

            // Get Output Names and Shapes
            for (size_t i = 0; i < session_ptr->GetOutputCount(); ++i) {
                output_nodes.push_back(session_ptr->GetOutputNameAllocated(i, allocator));
                output_names.push_back(output_nodes.back().get());
                Ort::TypeInfo type_info = session_ptr->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                output_types.push_back(tensor_info.GetElementType());
                output_shapes.push_back(tensor_info.GetShape());
            }
            
            // Initialize Input and Output Vectors: (assumes 1 input and 1 output tensor)
            if (input_shapes.size() != 1 || output_shapes.size() != 1) {
                return absl::InternalError("Policy Interface: Expected 1 input and 1 output tensor");
            }
            input_tensor_size = vector_product(input_shapes[0]);
            output_tensor_size = vector_product(output_shapes[0]);
            policy_input.resize(input_tensor_size);
            policy_output.resize(output_tensor_size);
            observation.resize(input_tensor_size);

            session_initialized = true;
            return absl::OkStatus();
        }

        absl::Status get_initial_position() {
            if(!unitree_driver->is_initialized())
                return absl::FailedPreconditionError("Policy Interface: Unitree Driver not initialized");

            unitree::containers::MotorState motor_state = unitree_driver->get_motor_state();
            initial_position = Eigen::Map<MotorVector<float>>(motor_state.q.data());
            initial_position_initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_thread() {
            if(!session_initialized)
                return absl::FailedPreconditionError("Policy Interface: ONNX Session not initialized");
            if(!initial_position_initialized)
                return absl::FailedPreconditionError("Policy Interface: Initial Position not initialized");

            thread = std::thread(&PolicyInterface::control_loop, this);
            thread_initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_threads() {
            // Initialize all threads:
            absl::Status result;
            if(!thread_initialized)
                result.Update(initialize_thread());
            if(!unitree_driver->is_thread_initialized())
                result.Update(unitree_driver->initialize_thread());

            ABSL_CHECK(result.ok()) << result.message();

            return absl::OkStatus();
        }

        absl::Status stop_thread() {
            if(!thread_initialized)
                return absl::FailedPreconditionError("Policy Interface: Control Thread not initialized");

            running = false;
            thread.join();
            return absl::OkStatus();
        }

        absl::Status stop_threads() {
            absl::Status result;
            result.Update(stop_thread());
            result.Update(unitree_driver->stop_thread());

            return result;
        }

        absl::Status set_control_mode(ControlMode mode) {
            std::lock_guard<std::mutex> lock(mutex);
            control_mode = mode;
            return absl::OkStatus();
        }

        ControlMode get_control_mode() {
            std::lock_guard<std::mutex> lock(mutex);
            return control_mode;
        }

        absl::Status set_command(const Vector3<float>& new_command) {
            std::lock_guard<std::mutex> lock(mutex);
            command = new_command;
            return absl::OkStatus();
        }

        Vector3<float> get_command() {
            std::lock_guard<std::mutex> lock(mutex);
            return command;
        }
    
    private:
        /* Shared Variables */
        Vector3<float> command = Vector3<float>::Zero();
        /* ONNX Variables */
        std::filesystem::path onnx_model_path;
        std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXPolicy");
        std::unique_ptr<Ort::Session> session_ptr;
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<Ort::AllocatedStringPtr> input_nodes;
        std::vector<Ort::AllocatedStringPtr> output_nodes;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        std::vector<ONNXTensorElementDataType> input_types;
        std::vector<ONNXTensorElementDataType> output_types;
        std::vector<std::vector<int64_t>> input_shapes;
        std::vector<std::vector<int64_t>> output_shapes;
        /* Initialization Flags */
        bool session_initialized = false;
        bool thread_initialized = false;
        bool initial_position_initialized = false;
        /* Unitree Driver */
        std::shared_ptr<RobotDriver> unitree_driver;
        /* Thread Variables */
        std::atomic<bool> running{true};
        std::thread thread;
        std::mutex mutex;
        /* Policy Variables */
        size_t input_tensor_size;
        size_t output_tensor_size;
        std::vector<float> policy_input;
        std::vector<float> policy_output;
        Eigen::Vector<float, Eigen::Dynamic> observation;
        /* Default Command Values */
        MotorVector<float> initial_position;
        MotorVector<float> default_position = {
            0.0f, 0.9f, -1.8f,
            0.0f, 0.9f, -1.8f,
            0.0f, 0.9f, -1.8f,
            0.0f, 0.9f, -1.8f
        };
        std::array<float, unitree::containers::num_motors> q_setpoint = {
            0.0f, 0.9f, -1.8f,
            0.0f, 0.9f, -1.8f,
            0.0f, 0.9f, -1.8f,
            0.0f, 0.9f, -1.8f
        };
        std::array<float, unitree::containers::num_motors> qd_setpoint = {
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f
        };
        std::array<float, unitree::containers::num_motors> torque_feedforward = {
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f
        };
        std::array<float, unitree::containers::num_motors> stiffness = {
            35.0f, 35.0f, 35.0f,
            35.0f, 35.0f, 35.0f,
            35.0f, 35.0f, 35.0f,
            35.0f, 35.0f, 35.0f
        };
        std::array<float, unitree::containers::num_motors> damping = {
            0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f
        };
        ControlMode control_mode = ControlMode::Damping;
        const int control_rate_us = 20000;  // 50Hz
        const float action_scale = 0.5f;

        absl::Status inference_policy() {
            // Initialize Input and Output Tensors:
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
            );
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                policy_input.data(),
                policy_input.size(),
                input_shapes[0].data(),
                input_shapes[0].size()
            );
            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                policy_output.data(),
                policy_output.size(),
                output_shapes[0].data(),
                output_shapes[0].size()
            );

            // Inference:
            Ort::RunOptions run_options;
            session_ptr->Run(
                run_options,
                input_names.data(),
                &input_tensor,
                1,
                output_names.data(),
                &output_tensor,
                1
            );

            if(!output_tensor.HasValue()) {
                return absl::InternalError("Failed to get output tensor");
            }

            return absl::OkStatus();
        }

        absl::Status get_observation() {
            // Get Measurements:
            unitree::containers::IMUState imu_state = unitree_driver->get_imu_state();
            unitree::containers::MotorState motor_state = unitree_driver->get_motor_state();

            // IMU State and Motor State measurements:
            Vector3<float> accelerometer_measurement = Eigen::Map<Vector3<float>>(imu_state.accelerometer.data());
            Vector3<float> gyroscope_measurement = Eigen::Map<Vector3<float>>(imu_state.gyroscope.data());
            Vector4<float> quaternion_measurement = Eigen::Map<Vector4<float>>(imu_state.quaternion.data());
            MotorVector<float> joint_positions = Eigen::Map<MotorVector<float>>(motor_state.q.data());
            MotorVector<float> joint_velocities = Eigen::Map<MotorVector<float>>(motor_state.qd.data());
            
            // Projected Gravity:
            Eigen::Quaternion<float> quaternion(
                quaternion_measurement(0), quaternion_measurement(1), quaternion_measurement(2), quaternion_measurement(3)
            );
            quaternion.normalize();
            Eigen::Matrix3<float> rotation = quaternion.toRotationMatrix();
            Vector3<float> projected_gravity = rotation.transpose() * Vector3<float>(0.0f, 0.0f, -1.0f);
            
            // Set Last Actions from Policy Output:
            MotorVector<float> previous_actions = Eigen::Map<MotorVector<float>>(policy_output.data());
            
            // Velocity Commands:
            Vector3<float> commands = command;
            
            // Set Observation:
            observation << gyroscope_measurement,
                           projected_gravity,
                           joint_positions - default_position,
                           joint_velocities,
                           previous_actions,
                           commands;

            // Set Input Tensor:
            for(size_t i = 0; i < input_tensor_size; ++i) {
                policy_input[i] = observation(i);
            }

            return absl::OkStatus();
        }

        unitree::containers::MotorCommand get_motor_command() {
            MotorVector<float> actions = Eigen::Map<MotorVector<float>>(policy_output.data());
            MotorVector<float> position_setpoints = default_position + actions * action_scale;
            Eigen::Map<MotorVector<float>>(q_setpoint.data()) = position_setpoints;

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
                next_time += std::chrono::microseconds(control_rate_us);
                /* Lock Guard Scope */
                {
                    std::lock_guard<std::mutex> lock(mutex);

                    absl::Status result;

                    // Update Observation:
                    result.Update(get_observation());

                    // Inference Policy:
                    result.Update(inference_policy());

                    // Get Motor Command:
                    unitree::containers::MotorCommand motor_command;
                    switch(control_mode) {
                        case ControlMode::Damping:
                            motor_command = interface::constants::controller::damping_motor_command;
                            break;
                        case ControlMode::GetUp:
                            std::tie(motor_command, control_mode) = interface::utilities::get_up_routine(
                                initial_position,
                                default_position,
                                control_rate_us
                            );
                            break;
                        case ControlMode::Stand:
                            motor_command = interface::constants::controller::stand_motor_command;
                            break;
                        case ControlMode::Policy:
                            motor_command = get_motor_command();
                            break;
                        case ControlMode::OperationalSpaceController:
                            control_mode = ControlMode::Damping;
                            motor_command = interface::constants::controller::damping_motor_command;
                            break;
                    }

                    // Send Motor Command:
                    unitree_driver->update_command(motor_command);
                }

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