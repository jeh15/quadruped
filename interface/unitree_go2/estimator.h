#pragma once

#include <filesystem>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"

#include "mujoco.h"
#include "Eigen/Dense"

#include "interface/estimators/kalman.h"
#include "operational-space-control/unitree_go2/autogen/autogen_defines.h"
#include "unitree-api/lowlevelapi.h"

// TODO(jeh15): Number of sensors is hardcoded. Make these AutoGen.
namespace {
    using MotorVector = Eigen::Vector<double, constants::model::nu_size>;
    using MotorVectorFloat = Eigen::Vector<float, constants::model::nu_size>;
    using Quaternion = Eigen::Vector<double, 4>;
    using QuaternionFloat = Eigen::Vector<float, 4>;
    using Vector3 = Eigen::Vector<double, 3>;
    using Vector3Float = Eigen::Vector<float, 3>;
    using SensorVector = Eigen::Vector<double, 39>;
    using SensorVectorFloat = Eigen::Vector<float, 39>;

}

struct EstimatorArgs {
    std::filesystem::path xml_path;
    int control_rate = 1000;
};

class EstimatorInterface {
    public:
        EstimatorInterface(std::shared_ptr<UnitreeDriver> unitree_drv, const std::filesystem::path xml_path, const int control_rate) : unitree_driver(unitree_drv) {
            char error[1000];
            mj_model = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
            ABSL_CHECK(mj_model != nullptr) << "Failed to load model: " << error;

            control_rate_us = control_rate;
            mj_model->opt.timestep = control_rate * 1e-6;

            mj_data = mj_makeData(mj_model);
            ABSL_CHECK(mj_data != nullptr) << "Failed to make data";

            estimator = mjpc::Kalman(mj_model);
            ABSL_CHECK(estimator != nullptr) << "Failed to initialize estimator";
        }
        ~EstimatorInterface() {};

        absl::Status initialize() {
           if(!motor_controller->is_initialized())
                return absl::FailedPreconditionError("Unitree Driver not initialized");

            std::ignore = get_measurements();
            
            // Figure out this type:
            estimator.SetState(state?);

            return absl::OkStatus();
        }

        absl::Status initialize_estimator_thread() {
            thread = std::thread(&EstimatorInterface::estimator_loop, this);
            estimator_thread_initialized = true;
            return absl::OkStatus();
        }

        absl::Status stop_estimator_thread() {
            if(!estimator_thread_initialized)
                return absl::FailedPreconditionError("Estimator thread not initialized");

            running = false;
            thread.join();
            return absl::OkStatus();
        }

        absl::Status get_state(){
        }
        
    private:
        /* Mujoco */
        std::filesystem::path model_path;
        MjModel* mj_model;
        MjData* mj_data;
        /* Estimator */
        mjpc::Kalman estimator;
        MotorVector ctrl = MotorVector::Zero();
        SensorVector sensor = SensorVector::Zero();
        /* UnitreeDriver */
        std::shared_ptr<UnitreeDriver> unitree_driver;
        /* Thread */
        std::atomic<bool> running = true;
        std::thread thread;
        std::mutex mutex;
        bool estimator_thread_initialized = false;
        int control_rate_us;

        absl::Status update_state() {
            // Parse C Array from Estimator:
        }

        absl::Status get_measurements() {
            // Get Sensor Data:
            lowleveltypes::LowState low_state = unitree_driver->get_low_state();
            lowleveltypes::IMUState imu_state = unitree_driver->get_imu_state();
            lowleveltypes::MotorState motor_state = unitree_driver->get_motor_state();
            
            // Calculate Contact Mask:
            ContactMask contact_mask = ContactMask::Zero();
            Eigen::Vector<short, 4> foot_force = Eigen::Map<Eigen::Vector<short, 4>>(low_state.foot_force.data())(foot_idx_map);
            for(int i = 0; i < 4; i++) {
                contact_mask(i) = foot_force(i) > contact_threshold;
            }

            // Combine Control and Sensor Data:
            MotorVectorFloat q = Eigen::Map<MotorVectorFloat>(motor_state.q.data())(motor_idx_map);
            MotorVectorFloat qd = Eigen::Map<MotorVectorFloat>(motor_state.qd.data())(motor_idx_map);
            MotorVectorFloat torque_estimate = Eigen::Map<MotorVectorFloat>(motor_state.torque_estimate.data())(motor_idx_map);
            QuaternionFloat quaternion = Eigen::Map<QuaternionFloat>(imu_state.quaternion.data());
            Vector3Float gyroscop = Eigen::Map<Vector3Float>(imu_state.gyroscope.data());
            Vector3Float accelerometer = Eigen::Map<Vector3Float>(imu_state.accelerometer.data());
            
            SensorVectorFloat sensor_vector;
            sensor_vector << q, qd, torque_estimate, quaternion, gyroscop, accelerometer, contact_mask;

            // Update estimator measurements:
            ctrl = torque_estimate.cast<double>();
            sensor = sensor_vector.cast<double>();

            return absl::OkStatus();
        }

        void estimator_loop() {
            using Clock = std::chrono::steady_clock;
            auto start = Clock::now();
            auto next_time = Clock::now();
            while(running) {
                // Calculate next time:
                next_time += std::chrono::milliseconds(control_rate_us);
                /* Lock Guard Scope */
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    
                    // Get Measurements: (Updates ctrl and sensor)
                    std::ignore = get_measurements();

                    // Update Measurement:
                    estimator->UpdateMeasurement(ctrl.data(), sensor.data())

                    // Update prediction:
                    estimator->UpdatePredict();

                    // Get and Set State:
                    double* state = estimator->State();
                    update_state(state);
                }
            }
        }
};
