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
#include "unitree-api/unitree_driver.h"
#include "unitree-api/containers.h"

#include "interface/estimators/autogen/estimator_defines.h"
#include "operational-space-control/unitree_go2/autogen/autogen_defines.h"
#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"
#include "interface/unitree_go2/logger.h"

using namespace interface::aliases::estimator;
using namespace interface::containers::estimator;
using namespace interface::containers::logger;

template <typename RobotDriver = UnitreeDriver>
class EstimatorInterface {
    public:
        EstimatorInterface(
            std::shared_ptr<RobotDriver> unitree_driver,
            EstimatorArgs estimator_args,
            LoggerArgs log_args
        ) : 
            unitree_driver(unitree_driver),
            xml_path(estimator_args.xml_path),
            control_rate_us(estimator_args.control_rate_us),
            logger(log_args.filepath, log_args.log_rate_us),
            enable_logging(log_args.enable_logging)
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
            absl::Status result;
            if(!unitree_driver->is_initialized())
                return absl::FailedPreconditionError("Unitree Driver not initialized");

            if(enable_logging)
                result.Update(logger.initialize());

            // Update Measurements:
            result.Update(get_measurements());
            
            // Create intial state:
            StateVector initial_state;
            StateVector default_state = Eigen::Map<StateVector>(estimator.State());

            // initial_state << default_state.head(3), sensor.segment(36, 4), sensor.segment(0, 12),
            //                 default_state.segment(19, 3), sensor.segment(40, 3), sensor.segment(12, 12);
            
            // Set initial state:
            // estimator.SetState(initial_state.data());

            // Update State:
            StateVector state_vector = Eigen::Map<StateVector>(estimator.State());
            result.Update(update_estimator_state(state_vector));
            
            // Assert Initialization:
            ABSL_CHECK(result.ok()) << result.message();

            initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_estimator_thread() {
            if(!initialized)
                return absl::FailedPreconditionError("Estimator not initialized");

            absl::Status result;
            thread = std::thread(&EstimatorInterface::estimator_loop, this);
            estimator_thread_initialized = true;
            if(enable_logging)
                result.Update(logger.initialize_log_thread());
            
            // Assert Initialization:
            ABSL_CHECK(result.ok()) << result.message();

            return absl::OkStatus();
        }

        absl::Status stop_estimator_thread() {
            absl::Status result;
            if(!estimator_thread_initialized)
                return absl::FailedPreconditionError("Estimator thread not initialized");

            running = false;
            thread.join();

            if(enable_logging)
                result.Update(logger.stop_log_thread());

            return result;
        }

        EstimatorState get_state(){
            std::lock_guard<std::mutex> lock(mutex);
            return state;
        }

        MotorVector get_control() {
            std::lock_guard<std::mutex> lock(mutex);
            return ctrl;
        }

        SensorVector get_sensor() {
            std::lock_guard<std::mutex> lock(mutex);
            return sensor;
        }
        
    // private:
    public:
        /* UnitreeDriver */
        std::shared_ptr<RobotDriver> unitree_driver;
        /* Estimator */
        mjpc::Kalman estimator;
        MotorVector ctrl = MotorVector::Zero();
        SensorVector sensor = SensorVector::Zero();
        EstimatorState state;
        bool initialized = false;
        /* Mujoco */
        std::filesystem::path xml_path;
        /* Thread */
        std::atomic<bool> running = true;
        std::thread thread;
        std::mutex mutex;
        bool estimator_thread_initialized = false;
        int control_rate_us;
        /* Contact -- Need to find nominal value */
        interface::aliases::controller::ContactMask contact_mask;
        const short contact_threshold = 5;
        /* Logging */
        EstimatorLogger logger;
        bool enable_logging;

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
            contact_mask = interface::aliases::controller::ContactMask::Zero();
            Eigen::Vector<short, 4> foot_force = Eigen::Map<Eigen::Vector<short, 4>>(low_state.foot_force.data());
            for(int i = 0; i < 4; i++) {
                contact_mask(i) = foot_force(i) > contact_threshold;
            }

            // Combine Control and Sensor Data:
            MotorVectorFloat q = Eigen::Map<MotorVectorFloat>(motor_state.q.data());
            MotorVectorFloat qd = Eigen::Map<MotorVectorFloat>(motor_state.qd.data());
            MotorVectorFloat torque_estimate = Eigen::Map<MotorVectorFloat>(motor_state.torque_estimate.data());
            QuaternionFloat quaternion = Eigen::Map<QuaternionFloat>(imu_state.quaternion.data());
            Vector3Float gyroscope = Eigen::Map<Vector3Float>(imu_state.gyroscope.data());
            Vector3Float accelerometer = Eigen::Map<Vector3Float>(imu_state.accelerometer.data());
            
            SensorVectorFloat sensor_vector;
            // With Torque Estimate:
            // sensor_vector << q, qd, torque_estimate, quaternion, gyroscope, accelerometer;
            // Without Torque Estimate:
            sensor_vector << q, qd, quaternion, gyroscope, accelerometer;


            // Update estimator measurements:
            ctrl = torque_estimate.cast<double>();
            sensor = sensor_vector.cast<double>();

            return absl::OkStatus();
        }

        absl::Status update_estimator_state(const StateVector& state_vector) {
            // Update State:
            state.body_position = state_vector.segment(0, 3);
            state.body_rotation = state_vector.segment(3, 4);
            state.motor_position = state_vector.segment(7, 12);
            state.linear_body_velocity = state_vector.segment(19, 3);
            state.angular_body_velocity = state_vector.segment(22, 3);
            state.motor_velocity = state_vector.segment(25, 12);
            return absl::OkStatus();
        }

        void estimator_loop() {
            using Clock = std::chrono::steady_clock;
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
                    StateVector state_vector = Eigen::Map<StateVector>(state_ptr);
                    std::ignore = update_estimator_state(state_vector);

                    if(enable_logging)
                        std::ignore = logger.update_state(state);
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
