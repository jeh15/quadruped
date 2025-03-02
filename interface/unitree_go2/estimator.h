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
#include "interface/unitree_go2/aliases.h"

struct EstimatorArgs {
    std::filesystem::path xml_path;
    int control_rate = 1000;
};

using namespace aliases::estimator;
using namespace aliases::interface;


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
            StateVector initial_state;
            StateVector default_state = Eigen::Map<StateVector>(estimator.State());

            initial_state << default_state.head(3), sensor.segment(36, 4), sensor.segment(0, 12),
                            default_state.segment(19, 3), sensor.segment(40, 3), sensor.segment(12, 12);
            
            // Set initial state:
            estimator.SetState(initial_state.data());

            // Update State:
            state = Eigen::Map<StateVector>(estimator.State());

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

        StateVector get_state(){
            std::lock_guard<std::mutex> lock(mutex);
            return state;
        }
        
    // private:
    public:
        /* UnitreeDriver */
        std::shared_ptr<RobotDriver> unitree_driver;
        /* Estimator */
        mjpc::Kalman estimator;
        MotorVector ctrl = MotorVector::Zero();
        SensorVector sensor = SensorVector::Zero();
        StateVector state = StateVector::Zero();
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
            sensor_vector << q, qd, torque_estimate, quaternion, gyroscop, accelerometer;

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
                    double* state_ptr = estimator.State();
                    StateVector state = Eigen::Map<StateVector>(state_ptr);
                }
                // Check for overrun and sleep until next time:
                auto now = Clock::now();
                if (now < next_time) {
                    std::this_thread::sleep_until(next_time);
                } 
                else {
                    // Log overrun:
                    auto overrun = std::chrono::duration_cast<std::chrono::microseconds>(now - next_time);
                    std::cout << "Estimator Loop Execution Time Exceeded Control Rate: " 
                        << overrun.count() << "us" << std::endl;
                    next_time = now;
                }
            }
        }
};
