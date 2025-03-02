#pragma once

#include <filesystem>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"

#include "interface/estimators/kalman.h"
#include "interface/unitree_go2/simulation_interface.h"
#include "unitree-api/unitree_driver.h"
#include "unitree-api/containers.h"

#include "interface/estimators/autogen/estimator_defines.h"
#include "operational-space-control/unitree_go2/autogen/autogen_defines.h"


namespace {
    using MotorVector = Eigen::Vector<double, constants::estimator::nu>;
    using MotorVectorFloat = Eigen::Vector<float, constants::estimator::nu>;
    using Quaternion = Eigen::Vector<double, 4>;
    using QuaternionFloat = Eigen::Vector<float, 4>;
    using Vector3 = Eigen::Vector<double, 3>;
    using Vector3Float = Eigen::Vector<float, 3>;
    using SensorVector = Eigen::Vector<double, constants::estimator::sensor_size>;
    using SensorVectorFloat = Eigen::Vector<float, constants::estimator::sensor_size>;
    using ContactMask = Eigen::Vector<double, constants::model::contact_site_ids_size>;
}

struct EstimatorArgs {
    std::filesystem::path xml_path;
    int control_rate = 1000;
};

template <typename RobotDriver = UnitreeDriver>
class EstimatorInterface {
    public:
        EstimatorInterface(
            std::shared_ptr<RobotDriver> unitree_driver,
            const std::filesystem::path xml_path,
            const int control_rate_us) : 
            unitree_driver(unitree_driver),
            xml_path(xml_path),
            control_rate_us(control_rate_us)
        {
            char error[1000];
            mjModel* mj_model = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
            if(!mj_model)
                ABSL_CHECK(false) << "Failed to load model: " << error;

            double timestep = std::chrono::duration<double>(std::chrono::microseconds(control_rate_us)).count();
            mj_model->opt.timestep = timestep;
            
            estimator = mjpc::Kalman(mj_model);
        }
        ~EstimatorInterface() {};

        absl::Status clean_up() {
            estimator.CleanUp();
            return absl::OkStatus();
        }

        absl::Status initialize() {
           if(!unitree_driver->is_initialized())
                return absl::FailedPreconditionError("Unitree Driver not initialized");

            // Update Measurements:
            std::ignore = get_measurements();
            
            // Create intial state:

            // Figure out this type: (ndstate) from estimator.DimensionProcess() ndstate_ = 2 * nv + na;
            // estimator.SetState(state?);

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
            return absl::OkStatus();
        }
        
    // private:
    public:
        /* UnitreeDriver */
        std::shared_ptr<RobotDriver> unitree_driver;
        /* Estimator */
        mjpc::Kalman estimator;
        MotorVector ctrl = MotorVector::Zero();
        SensorVector sensor = SensorVector::Zero();
        /* Mujoco */
        std::filesystem::path xml_path;
        /* Thread */
        std::atomic<bool> running = true;
        std::thread thread;
        std::mutex mutex;
        bool estimator_thread_initialized = false;
        int control_rate_us;
        /* Contact -- Need to find nominal value */
        const short contact_threshold = 5;

        absl::Status update_state() {
            // Parse C Array from Estimator:
            return absl::OkStatus();
        }

        absl::Status get_measurements() {
            // Get Sensor Data:
            unitree::containers::LowState low_state = unitree_driver->get_low_state();
            unitree::containers::IMUState imu_state = unitree_driver->get_imu_state();
            unitree::containers::MotorState motor_state = unitree_driver->get_motor_state();
            
            // Calculate Contact Mask:
            ContactMask contact_mask = ContactMask::Zero();
            Eigen::Vector<short, 4> foot_force = Eigen::Map<Eigen::Vector<short, 4>>(low_state.foot_force.data());
            for(int i = 0; i < 4; i++) {
                contact_mask(i) = foot_force(i) > contact_threshold;
            }

            // Combine Control and Sensor Data:
            MotorVectorFloat q = Eigen::Map<MotorVectorFloat>(motor_state.q.data());
            MotorVectorFloat qd = Eigen::Map<MotorVectorFloat>(motor_state.qd.data());
            MotorVectorFloat torque_estimate = Eigen::Map<MotorVectorFloat>(motor_state.torque_estimate.data());
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
                    estimator.UpdateMeasurement(ctrl.data(), sensor.data());

                    // Update prediction:
                    estimator.UpdatePrediction();

                    // Get and Set State: (ndstate)
                    double* state = estimator.State();
                    update_state();
                }
            }
        }
};


// template <typename RobotDriver = UnitreeDriver>
// class EstimatorInterface {
// public:
//     // Template constructor that works with any compatible driver type
//     template <typename DriverType>
//     EstimatorInterface(
//         std::shared_ptr<DriverType> driver,
//         const std::filesystem::path xml_path,
//         const int control_rate_us) : 
//         unitree_driver(std::static_pointer_cast<RobotDriver>(driver)),
//         xml_path(xml_path),
//         control_rate_us(control_rate_us),
//         mj_model(nullptr),
//         mj_data(nullptr)
//     {
//         try {
//             char error[1000] = {0};
//             mj_model = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
//             if (!mj_model) {
//                 throw std::runtime_error(std::string("Failed to load model: ") + error);
//             }

//             double timestep = std::chrono::duration<double>(std::chrono::microseconds(control_rate_us)).count();
//             mj_model->opt.timestep = timestep;

//             mj_data = mj_makeData(mj_model);
//             if (!mj_data) {
//                 throw std::runtime_error("Failed to create MuJoCo data");
//             }

//             estimator = mjpc::Kalman(mj_model);
//         } catch (const std::exception& e) {
//             // Clean up resources if initialization fails
//             cleanup();
//             throw; // Rethrow the exception
//         }
//     }

//     // Specialized constructor for the default driver type
//     EstimatorInterface(
//         std::shared_ptr<RobotDriver> unitree_driver,
//         const std::filesystem::path xml_path,
//         const int control_rate_us) : 
//         unitree_driver(unitree_driver),
//         xml_path(xml_path),
//         control_rate_us(control_rate_us),
//         mj_model(nullptr),
//         mj_data(nullptr)
//     {
//         try {
//             char error[1000] = {0};
//             mj_model = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
//             if (!mj_model) {
//                 throw std::runtime_error(std::string("Failed to load model: ") + error);
//             }

//             double timestep = std::chrono::duration<double>(std::chrono::microseconds(control_rate_us)).count();
//             mj_model->opt.timestep = timestep;

//             mj_data = mj_makeData(mj_model);
//             if (!mj_data) {
//                 throw std::runtime_error("Failed to create MuJoCo data");
//             }

//             estimator = mjpc::Kalman(mj_model);
//         } catch (const std::exception& e) {
//             // Clean up resources if initialization fails
//             cleanup();
//             throw; // Rethrow the exception
//         }
//     }

//     ~EstimatorInterface() {
//         cleanup();
//     }

//     absl::Status initialize() {
//         if(!unitree_driver->is_initialized())
//              return absl::FailedPreconditionError("Unitree Driver not initialized");

//          std::ignore = get_measurements();
         
//          // Figure out this type: (ndstate) from estimator.DimensionProcess() ndstate_ = 2 * nv + na;
//          // estimator.SetState(state?);

//          return absl::OkStatus();
//      }

//      absl::Status initialize_estimator_thread() {
//          thread = std::thread(&EstimatorInterface::estimator_loop, this);
//          estimator_thread_initialized = true;
//          return absl::OkStatus();
//      }

//      absl::Status stop_estimator_thread() {
//          if(!estimator_thread_initialized)
//              return absl::FailedPreconditionError("Estimator thread not initialized");

//          running = false;
//          thread.join();
//          return absl::OkStatus();
//      }

//      absl::Status get_state(){
//          return absl::OkStatus();
//      }

//     // Rest of your class implementation...
//     /* UnitreeDriver */
//     std::shared_ptr<RobotDriver> unitree_driver;
//     /* Estimator */
//     mjpc::Kalman estimator;
//     MotorVector ctrl = MotorVector::Zero();
//     SensorVector sensor = SensorVector::Zero();
//     /* Mujoco */
//     std::filesystem::path xml_path;
//     mjModel* mj_model;
//     mjData* mj_data;
//     /* Thread */
//     std::atomic<bool> running = true;
//     std::thread thread;
//     std::mutex mutex;
//     bool estimator_thread_initialized = false;
//     int control_rate_us;
//     /* Contact -- Need to find nominal value */
//     const short contact_threshold = 5;

//     absl::Status update_state() {
//         // Parse C Array from Estimator:
//         return absl::OkStatus();
//     }

//     absl::Status get_measurements() {
//         // Get Sensor Data:
//         unitree::containers::LowState low_state = unitree_driver->get_low_state();
//         unitree::containers::IMUState imu_state = unitree_driver->get_imu_state();
//         unitree::containers::MotorState motor_state = unitree_driver->get_motor_state();
        
//         // Calculate Contact Mask:
//         ContactMask contact_mask = ContactMask::Zero();
//         Eigen::Vector<short, 4> foot_force = Eigen::Map<Eigen::Vector<short, 4>>(low_state.foot_force.data());
//         for(int i = 0; i < 4; i++) {
//             contact_mask(i) = foot_force(i) > contact_threshold;
//         }

//         // Combine Control and Sensor Data:
//         MotorVectorFloat q = Eigen::Map<MotorVectorFloat>(motor_state.q.data());
//         MotorVectorFloat qd = Eigen::Map<MotorVectorFloat>(motor_state.qd.data());
//         MotorVectorFloat torque_estimate = Eigen::Map<MotorVectorFloat>(motor_state.torque_estimate.data());
//         QuaternionFloat quaternion = Eigen::Map<QuaternionFloat>(imu_state.quaternion.data());
//         Vector3Float gyroscop = Eigen::Map<Vector3Float>(imu_state.gyroscope.data());
//         Vector3Float accelerometer = Eigen::Map<Vector3Float>(imu_state.accelerometer.data());
        
//         SensorVectorFloat sensor_vector;
//         sensor_vector << q, qd, torque_estimate, quaternion, gyroscop, accelerometer, contact_mask;

//         // Update estimator measurements:
//         ctrl = torque_estimate.cast<double>();
//         sensor = sensor_vector.cast<double>();

//         return absl::OkStatus();
//     }

//     void estimator_loop() {
//         using Clock = std::chrono::steady_clock;
//         auto start = Clock::now();
//         auto next_time = Clock::now();
//         while(running) {
//             // Calculate next time:
//             next_time += std::chrono::milliseconds(control_rate_us);
//             /* Lock Guard Scope */
//             {
//                 std::lock_guard<std::mutex> lock(mutex);
                
//                 // Get Measurements: (Updates ctrl and sensor)
//                 std::ignore = get_measurements();

//                 // Update Measurement:
//                 estimator.UpdateMeasurement(ctrl.data(), sensor.data());

//                 // Update prediction:
//                 estimator.UpdatePrediction();

//                 // Get and Set State: (ndstate)
//                 double* state = estimator.State();
//                 update_state();
//             }
//         }
//     }

// private:
//     void cleanup() {
//         if (mj_data) {
//             mj_deleteData(mj_data);
//             mj_data = nullptr;
//         }
//         if (mj_model) {
//             mj_deleteModel(mj_model);
//             mj_model = nullptr;
//         }
//     }

//     // Your existing members...
// };