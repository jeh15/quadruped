#pragma once

#include <iostream>
#include <string>
#include <vector>
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

#include <onnxruntime_cxx_api.h>

#include "Eigen/Dense"

#include "unitree-api/unitree_driver.h"
#include "unitree-api/containers.h"

#include "interface/unitree_go2/safety_controller.h"
#include "interface/unitree_go2/logger.h"
#include "interface/estimators/imu_estimator.h"
#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"
#include "interface/unitree_go2/constants.h"

using namespace interface::aliases::common;
using namespace interface::containers::controller;
using namespace interface::containers::logger;
using namespace interface::containers::estimator;


template <typename T>
T vector_product(const std::vector<T>& v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

//TODO(jeh15): Compile time constants.
template <typename RobotDriver = UnitreeDriver>
class PolicyInterface {
    public:
        PolicyInterface(
            std::filesystem::path onnx_model_path,
            std::shared_ptr<RobotDriver> unitree_driver,
            LoggerArgs log_args
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

            ABSL_CHECK(result.ok()) << result.message();
            
            return absl::OkStatus();
        }

        absl::Status initialize_session() {
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_ptr = std::make_unique<Ort::Session>(env, onnx_model_path.c_str(), session_options);
            if (!session_ptr) {
                return absl::InternalError("Failed to create ONNX session");
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
    
    private:
        /* ONNX Variables */
        std::filesystem::path onnx_model_path;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXPolicy");
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
        std::array<float, unitree::containers::num_motors> q_setpoint = {
            0.0f, 0.9f, -1.8f,
            0.0f, 0.9f, -1.8f,
            0.0f, 0.9f, -1.8f,
            0.0f, 0.9f, -1.8f,
        };
        std::array<float, unitree::containers::num_motors> qd_setpoint = {0.0f};
        std::array<float, unitree::containers::num_motors> torque_feedforward = {0.0f};
        std::array<float, unitree::containers::num_motors> stiffness = {35.0f};
        std::array<float, unitree::containers::num_motors> damping = {0.5f};



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
            common::Vector3<float> accelerometer_measurement = Eigen::Map<common::Vector3<float>>(imu_state.accelerometer.data());
            common::Vector3<float> gyroscope_measurement = Eigen::Map<common::Vector3<float>>(imu_state.gyroscope.data());
            common::Quaternion<float> quaternion_measurement = Eigen::Map<common::Quaternion<float>>(imu_state.quaternion.data());
            common::MotorVector<float> joint_positions = Eigen::Map<common::MotorVector<float>>(motor_state.q.data());
            common::MotorVector<float> joint_velocities = Eigen::Map<common::MotorVector<float>>(motor_state.qd.data());
            
            // Projected Gravity:
            Eigen::Quaternion<float> quaternion(
                quaternion_measurement(0), quaternion_measurement(1), quaternion_measurement(2), quaternion_measurement(3)
            );
            quaternion.normalize();
            Eigen::Matrix3<float> rotation = quaternion.toRotationMatrix();
            common::Vector3<float> projected_gravity = rotation.T * common::Vector3<float>(0.0f, 0.0f, -1.0f);
            
            // Set Last Actions from Policy Output:
            common::MotorVector<float> previous_actions = Eigen::Map<common::MotorVector<float>>(policy_output.data());
            
            // Velocity Commands:
            common::Vector3<float> commands = common::Vector3<float>::Zero();
            
            // Set Observation:
            observation << gyroscope_measurement,
                           project_gravity,
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
            common::MotorVector<float> actions = Eigen::Map<common::MotorVector<float>>(policy_output.data());
            common::MotorVector<float> position_setpoints = default_position + actions * action_scale;
            Eigen::Map<common::MotorVector<float>>(q_setpoint.data()) = position_setpoints;
            
            unitree::containers::MotorCommand motor_command = {
                .q_setpoint = q_setpoint,
                .qd_setpoint = qd_setpoint,
                .torque_feedforward = torque_feedforward,
                .stiffness = stiffness,
                .damping = damping,
            };

            return motor_command;
        }
};