#pragma once

#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <chrono>
#include <deque>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"

#include "Eigen/Dense"
#include "Eigen/Geometry"

#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"

using namespace interface::aliases;
using namespace interface::containers::estimator;


template <typename RobotDriver>
class IMUEstimator {
    public:
        IMUEstimator(std::shared_ptr<RobotDriver> unitree_driver, const int control_rate_us) : unitree_driver(unitree_driver), control_rate_us(control_rate_us) {};
        ~IMUEstimator() {};

        absl::Status initialize() {
            absl::Status result;
            if(!unitree_driver->is_initialized())
                return absl::FailedPreconditionError("Unitree Driver not initialized");

            // Initialize Queue:
            result.Update(initialize_queue());

            // Calculate Bias and initial Quaternion:
            result.Update(initialize_estimator_variables());

            // Initialize Measurements:
            result.Update(get_measurements());

            // Initialize State:
            result.Update(update_state());

            // Assert Initialization:
            ABSL_CHECK(result.ok()) << result.message();

            initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_thread() {
            if(!initialized)
                return absl::FailedPreconditionError("Estimator not initialized");

            thread = std::thread(&IMUEstimator::estimator_loop, this);
            thread_initialized = true;

            return absl::OkStatus();
        }

        absl::Status stop_thread() {
            if(!thread_initialized)
                return absl::FailedPreconditionError("Estimator thread not initialized");

            running = false;
            thread.join();

            return absl::OkStatus();
        }

        bool is_initialized() {
            return initialized;
        }

        bool is_thread_initialized() {
            return thread_initialized;
        }

        EstimatorState get_state() {
            std::lock_guard<std::mutex> lock(mutex);
            return estimator_state;
        }

        private:
            /* UnitreeDriver */
            std::shared_ptr<RobotDriver> unitree_driver;
            // Shared Variables:
            EstimatorState estimator_state;
            // Thread Variables:
            std::thread thread;
            std::mutex mutex;
            std::atomic<bool> running = true;
            int control_rate_us;
            bool initialized = false;
            bool thread_initialized = false;
            // IMU Estimation Constants:
            const float gyroscope_measurement_error = M_PI * (5.0f / 180.0f);
            const float measurement_beta = std::sqrt(3.0f / 4.0f) * gyroscope_measurement_error;
            // Unitree Measurements:
            common::MotorVector<float> q_estimate;
            common::MotorVector<float> qd_estimate;
            common::MotorVector<float> torque_estimate;
            // Estimation Variables:
            int start_up_time = 5;
            common::Vector3<float> gyroscope_bias = common::Vector3<float>::Zero();
            common::Vector3<float> accelerometer_bias = common::Vector3<float>::Zero();
            common::Vector3<float> g = common::Vector3<float>(0.0f, 0.0f, 9.81f);
            common::Vector3<float> gyroscope_estimate = common::Vector3<float>::Zero();
            common::Vector3<float> accelerometer_estimate = common::Vector3<float>::Zero();
            Eigen::Quaternion<float> quaternion_estimate = Eigen::Quaternion<float>::Identity();
            common::Vector3<float> position_estimate = common::Vector3<float>::Zero();
            common::Vector3<float> velocity_estimate = common::Vector3<float>::Zero();
            float delta_t = std::chrono::duration<float>(std::chrono::microseconds(control_rate_us)).count();
            // Integration Variables:
            const float h = delta_t / 3.0f;
            common::Vector3<float> acceleration_i = common::Vector3<float>::Zero();
            common::Vector3<float> acceleration_j = common::Vector3<float>::Zero();
            common::Vector3<float> acceleration_k = common::Vector3<float>::Zero();
            common::Vector3<float> velocity_i = common::Vector3<float>::Zero();
            common::Vector3<float> velocity_j = common::Vector3<float>::Zero();
            common::Vector3<float> velocity_k = common::Vector3<float>::Zero();
            common::Vector3<float> position = common::Vector3<float>::Zero();
            // Filter Variables:
            const float cutoff_frequency = 10.0f;
            static constexpr size_t lowpass_size = 10;
            static constexpr size_t highpass_size = 10;
            const float alpha = 0.9;
            const float time_constant = 1.0f / (2.0f * M_PI * cutoff_frequency);
            const float beta = time_constant / (time_constant + delta_t);
            std::deque<common::Vector3<float>> acceleration_queue;
            std::deque<common::Vector3<float>> velocity_queue;
            std::deque<common::Vector3<float>> position_queue;
            /* Contact -- Need better estimation */
            interface::aliases::controller::ContactMask<float> contact_mask;
            const short contact_threshold = 5;

            absl::Status initialize_queue() {
                common::Vector3<float> zero_vector = common::Vector3<float>::Zero();
                for(size_t i = 0; i < highpass_size; ++i) {
                    acceleration_queue.push_back(zero_vector);
                    velocity_queue.push_back(zero_vector);
                    position_queue.push_back(zero_vector);
                }

                return absl::OkStatus();
            }

            absl::Status initialize_estimator_variables() {
                std::vector<common::Vector3<float>> gyroscope_vector;
                std::vector<common::Vector3<float>> accelerometer_vector;
                std::vector<common::Vector4<float>> quaternion_vector;
                common::Vector4<float> quaternion_estimate_ = common::Vector4<float>::Zero();
    
                std::cout << "Calculating Gyroscope and Accelerometer Bias for " << start_up_time << " seconds" << std::endl;
    
                using Clock = std::chrono::steady_clock;
                auto start = Clock::now();
                while(Clock::now() - start < std::chrono::seconds(start_up_time)) {
                    unitree::containers::IMUState imu_state = unitree_driver->get_imu_state();
                    common::Vector3<float> gyroscope = Eigen::Map<common::Vector3<float>>(imu_state.gyroscope.data());
                    common::Vector3<float> accelerometer = Eigen::Map<common::Vector3<float>>(imu_state.accelerometer.data());
                    common::Vector4<float> quaternion = Eigen::Map<common::Vector4<float>>(imu_state.quaternion.data());
                    gyroscope_vector.push_back(gyroscope);
                    accelerometer_vector.push_back(accelerometer);
                    quaternion_vector.push_back(quaternion);
    
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                
                // Calculate average:
                for(auto& vector : gyroscope_vector) {
                    gyroscope_bias += vector;
                }
                if(!gyroscope_vector.empty()) {
                    gyroscope_bias /= static_cast<float>(gyroscope_vector.size());
                }
                else {
                    return absl::InternalError("Gyroscope Vector is Empty");
                }
                
                for(auto& vector : accelerometer_vector) {
                    accelerometer_bias += vector;
                }
                if(!accelerometer_vector.empty()) {
                    accelerometer_bias /= static_cast<float>(accelerometer_vector.size());
                }
                else {
                    return absl::InternalError("Accelerometer Vector is Empty");
                }

                for(auto& vector : quaternion_vector) {
                    quaternion_estimate_ += vector;
                }
                if(!quaternion_vector.empty()) {
                    quaternion_estimate_ /= static_cast<float>(quaternion_vector.size());
                    quaternion_estimate = Eigen::Quaternion<float>(quaternion_estimate_(0), quaternion_estimate_(1), quaternion_estimate_(2), quaternion_estimate_(3));
                    quaternion_estimate.normalize();
                }
                else {
                    return absl::InternalError("Quaternion Vector is Empty");
                }
                
                // Check Quaternion Estimate:
                Eigen::Quaternion<float> q = Eigen::Quaternion<float>(1, 0, 0, 0);
                Eigen::Quaternion<float> dq = quaternion_estimate * q.inverse();
                dq.normalize();
                if( dq.w() < 0.9 && dq.w() > -0.9) {
                    return absl::InternalError("Quaternion Estimate is not Valid");
                }

                std::cout << "Gyroscope and Accelerometer Initialization Complete" << std::endl;
                std::cout << "Gyroscope Bias: " << gyroscope_bias.transpose() << std::endl;
                std::cout << "Accelerometer Bias: " << accelerometer_bias.transpose() << std::endl;
                std::cout << "Quaternion Estimate: " << quaternion_estimate.w() << " " << quaternion_estimate.vec().transpose() << std::endl;

                return absl::OkStatus();
            }

            absl::Status get_measurements() {
                // Get Measurements:
                unitree::containers::LowState low_state = unitree_driver->get_low_state();
                unitree::containers::IMUState imu_state = unitree_driver->get_imu_state();
                unitree::containers::MotorState motor_state = unitree_driver->get_motor_state();
                common::Vector3<float> gyroscope_measurement = Eigen::Map<common::Vector3<float>>(imu_state.gyroscope.data());
                common::Vector3<float> accelerometer_measurement = Eigen::Map<common::Vector3<float>>(imu_state.accelerometer.data());

                // Calculate Contact Mask:
                contact_mask = interface::aliases::controller::ContactMask<float>::Zero();
                Eigen::Vector<short, 4> foot_force = Eigen::Map<Eigen::Vector<short, 4>>(low_state.foot_force.data());
                for(int i = 0; i < 4; i++) {
                    contact_mask(i) = foot_force(i) > contact_threshold;
                }
                
                // Parse Measurements:
                q_estimate = Eigen::Map<common::MotorVector<float>>(motor_state.q.data());
                qd_estimate = Eigen::Map<common::MotorVector<float>>(motor_state.qd.data());
                torque_estimate = Eigen::Map<common::MotorVector<float>>(motor_state.torque_estimate.data());
                
                // Correct Gyroscope and Accelerometer for Bias:
                gyroscope_estimate = gyroscope_measurement - gyroscope_bias;
                accelerometer_estimate = accelerometer_measurement - accelerometer_bias;

                return absl::OkStatus();
            }

            absl::Status quaternion_estimation_update() {
                // Unpack Gyroscope and Accelerometer Measurements:
                float w_x = gyroscope_estimate(0);
                float w_y = gyroscope_estimate(1);
                float w_z = gyroscope_estimate(2);
                float a_x = accelerometer_estimate(0);
                float a_y = accelerometer_estimate(1);
                float a_z = accelerometer_estimate(2);

                // Quaternion Estimation Update:
                float SEq_1 = quaternion_estimate.w();
                float SEq_2 = quaternion_estimate.x();
                float SEq_3 = quaternion_estimate.y();
                float SEq_4 = quaternion_estimate.z();

                // Local system variables
                float norm; // vector norm
                float SEqDot_omega_1, SEqDot_omega_2, SEqDot_omega_3, SEqDot_omega_4; // quaternion derrivative from gyroscopes elements
                float f_1, f_2, f_3; // objective function elements
                float J_11or24, J_12or23, J_13or22, J_14or21, J_32, J_33; // objective function Jacobian elements
                float SEqHatDot_1, SEqHatDot_2, SEqHatDot_3, SEqHatDot_4; // estimated direction of the gyroscope error
            
                // Axulirary variables to avoid reapeated calcualtions
                float halfSEq_1 = 0.5f * SEq_1;
                float halfSEq_2 = 0.5f * SEq_2;
                float halfSEq_3 = 0.5f * SEq_3;
                float halfSEq_4 = 0.5f * SEq_4;
                float twoSEq_1 = 2.0f * SEq_1;
                float twoSEq_2 = 2.0f * SEq_2;
                float twoSEq_3 = 2.0f * SEq_3;
            
                // Normalise the accelerometer measurement
                norm = std::sqrt(a_x * a_x + a_y * a_y + a_z * a_z);
                a_x /= norm;
                a_y /= norm;
                a_z /= norm;
            
                // Compute the objective function and Jacobian
                f_1 = twoSEq_2 * SEq_4 - twoSEq_1 * SEq_3 - a_x;
                f_2 = twoSEq_1 * SEq_2 + twoSEq_3 * SEq_4 - a_y;
                f_3 = 1.0f - twoSEq_2 * SEq_2 - twoSEq_3 * SEq_3 - a_z;
                J_11or24 = twoSEq_3; // J_11 negated in matrix multiplication
                J_12or23 = 2.0f * SEq_4;
                J_13or22 = twoSEq_1; // J_12 negated in matrix multiplication
                J_14or21 = twoSEq_2;
                J_32 = 2.0f * J_14or21; // negated in matrix multiplication
                J_33 = 2.0f * J_11or24; // negated in matrix multiplication
            
                // Compute the gradient (matrix multiplication)
                SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1;
                SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3;
                SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1;
                SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2;
            
                // Normalise the gradient
                norm = std::sqrt(SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2 + SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4);
                SEqHatDot_1 /= norm;
                SEqHatDot_2 /= norm;
                SEqHatDot_3 /= norm;
                SEqHatDot_4 /= norm;
            
                // Compute the quaternion derrivative measured by gyroscopes
                SEqDot_omega_1 = -halfSEq_2 * w_x - halfSEq_3 * w_y - halfSEq_4 * w_z;
                SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z - halfSEq_4 * w_y;
                SEqDot_omega_3 = halfSEq_1 * w_y - halfSEq_2 * w_z + halfSEq_4 * w_x;
                SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y - halfSEq_3 * w_x;
            
                // Compute then integrate the estimated quaternion derrivative
                SEq_1 += (SEqDot_omega_1 - (measurement_beta * SEqHatDot_1)) * delta_t;
                SEq_2 += (SEqDot_omega_2 - (measurement_beta * SEqHatDot_2)) * delta_t;
                SEq_3 += (SEqDot_omega_3 - (measurement_beta * SEqHatDot_3)) * delta_t;
                SEq_4 += (SEqDot_omega_4 - (measurement_beta * SEqHatDot_4)) * delta_t;
            
                // Normalise quaternion
                norm = std::sqrt(SEq_1 * SEq_1 + SEq_2 * SEq_2 + SEq_3 * SEq_3 + SEq_4 * SEq_4);
                SEq_1 /= norm;
                SEq_2 /= norm;
                SEq_3 /= norm;
                SEq_4 /= norm;

                // Update Quaternion Estimate:
                quaternion_estimate = Eigen::Quaternion<float>(SEq_1, SEq_2, SEq_3, SEq_4);

                return absl::OkStatus();
            }

            common::Vector3<float> lowpass_filter(std::deque<common::Vector3<float>>& queue, const common::Vector3<float>& value) {
                // Add value to queue:
                queue.push_back(value);

                // Filter Value:
                std::array<common::Vector3<float>, lowpass_size> filtered_values;
                filtered_values[0] = alpha * queue.front();
                for (size_t i = 1; i < lowpass_size; i++) {
                    filtered_values[i] = alpha * queue[i] + (1.0f - alpha) * filtered_values[i - 1];
                }

                // Remove oldest value:
                queue.pop_front();
                 
                return filtered_values.back();
            }

            common::Vector3<float> highpass_filter(std::deque<common::Vector3<float>>& queue, const common::Vector3<float>& value) {
                // Add value to queue:
                queue.push_back(value);

                // Filter Value:
                std::array<common::Vector3<float>, highpass_size> filtered_values;
                filtered_values[0] = queue.front();
                for (size_t i = 1; i < highpass_size; i++) {
                    filtered_values[i] = beta * (filtered_values[i - 1] + queue[i] - queue[i - 1]);
                }

                // Remove oldest value:
                queue.pop_front();
                 
                return filtered_values.back();
            }

            absl::Status motion_estimation_update() {
                // Precompute Values:
                Eigen::Matrix3<float> C = quaternion_estimate.toRotationMatrix();

                // Simpson's Rule:
                acceleration_i = C.transpose() * accelerometer_estimate;
                velocity_i = velocity_estimate + h * (acceleration_k + 4.0f * acceleration_j + acceleration_i);
                acceleration_j = acceleration_i;
                acceleration_k = acceleration_j;

                position = position_estimate + h * (velocity_k + 4.0f * velocity_j + velocity_i);
                velocity_j = velocity_i;
                velocity_k = velocity_j;

                // Filter Estimation: Causing Instability... Probably just a poor implementation
                common::Vector3<float> velocity = highpass_filter(velocity_queue, velocity_i);

                // Update State:
                position_estimate = position;
                velocity_estimate = velocity;

                return absl::OkStatus();
            }

            absl::Status update_state() {
                // Update Estimation Struct:
                estimator_state.body_position = position_estimate;
                estimator_state.body_rotation = quaternion_estimate;
                estimator_state.joint_position = q_estimate;
                estimator_state.linear_body_velocity = velocity_estimate;
                estimator_state.angular_body_velocity = gyroscope_estimate;
                estimator_state.joint_velocity = qd_estimate;
                estimator_state.linear_body_acceleration = accelerometer_estimate;
                estimator_state.torque_estimate = torque_estimate;
                estimator_state.contact_mask = contact_mask;

                return absl::OkStatus();
            }

            void estimator_loop() {
                using Clock = std::chrono::steady_clock;
                auto next_time = Clock::now();
                while(running) {
                    absl::Status result;
                    next_time += std::chrono::microseconds(control_rate_us);
                    /* Lock Guard Scope */
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        
                        // Get Measurements:
                        result.Update(get_measurements());
    
                        // Update Quaternion Estimate:
                        result.Update(quaternion_estimation_update());
    
                        // Update Position and Velocity Estimate:
                        result.Update(motion_estimation_update());
    
                        // Update Estimation Struct:
                        result.Update(update_state());
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
