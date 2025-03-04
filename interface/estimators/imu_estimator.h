#pragma once

#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>


#include "Eigen/Dense"
#include "Eigen/Geometry"

#include "interface/unitree_go2/aliases.h"


using namespace interface::aliases::common;


template <typename T>
struct IMUState {
    Quaternion<T> quaternion;
    Vector3<T> position;
    Vector3<T> linear_velocity;
    Vector3<T> gyroscope_bias;
    Vector3<T> accelerometer_bias;
};

class IMUEstimator {
    public:
        IMUEstimator(const int control_rate_us) : control_rate_us(control_rate_us) {};
        ~IMUEstimator() {};

        absl::Status initialize() {
            // Initialize IMU Estimator:
            calculate_bias();
            return absl::OkStatus();
        }

        // Push to autogen?
        void calculate_bias() {
            std::vector<Vector3<float>> gyroscope_vector;
            std::vector<Vector3<float>> accelerometer_vector;
            using Clock = std::chrono::steady_clock;
            auto start = Clock::now();
            while(Clock::now() - start < std::chrono::seconds(start_up_time)) {
                unitree::containers::IMUState imu_state = unitree_driver->get_imu_state();
                Vector3Float gyroscope = Eigen::Map<Vector3Float>(imu_state.gyroscope.data());
                Vector3Float accelerometer = Eigen::Map<Vector3Float>(imu_state.accelerometer.data());
                gyroscope_vector.push_back(gyroscope);
                accelerometer_vector.push_back(accelerometer);

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Calculate average:
            for(auto& vector : gyroscope_vector) {
                gyroscope_bias += vector;
            }
            if(!gyroscope_vector.empty())
                gyroscope_bias /= static_cast<float>(gyroscope_vector.size());
            
            for(auto& vector : accelerometer_vector) {
                accelerometer_bias += vector;
            }
            if(!accelerometer_vector.empty())
                accelerometer_bias /= static_cast<float>(accelerometer_vector.size());

        }


        private:
            // Constants:
            const float gyroscope_measurement_error = M_PI * (5.0f / 180.0f);
            const float beta = std::sqrt(3.0f / 4.0f) * gyroscope_measurement_error;
            int start_up_time = 5;
            Vector3<float> gyroscope_bias = Vector3<float>::Zero();
            Vector3<float> accelerometer_bias = Vector3<float>::Zero();
            Eigen::Quaternion<float> quaternion_estimate = Eigen::Quaternion<float>::Identity();
            Vector3<float> r = Vector3<float>::Zero();
            Vector3<float> v = Vector3<float>::Zero();
            int control_rate_us;
            float delta_t = std::chrono::duration<float>(std::chrono::microseconds(control_rate_us)).count();

            void quaternion_estimation_update(const Vector3<float>& gyroscope_measurement, const Vector3<float>& accelerometer_measurement) {
                // Unpack Gyroscope and Accelerometer Measurements:
                float w_x = gyroscope_measurement(0);
                float w_y = gyroscope_measurement(1);
                float w_z = gyroscope_measurement(2);
                float a_x = accelerometer_measurement(0);
                float a_y = accelerometer_measurement(1);
                float a_z = accelerometer_measurement(2);

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
                SEq_1 += (SEqDot_omega_1 - (beta * SEqHatDot_1)) * delta_t;
                SEq_2 += (SEqDot_omega_2 - (beta * SEqHatDot_2)) * delta_t;
                SEq_3 += (SEqDot_omega_3 - (beta * SEqHatDot_3)) * delta_t;
                SEq_4 += (SEqDot_omega_4 - (beta * SEqHatDot_4)) * delta_t;
            
                // Normalise quaternion
                norm = std::sqrt(SEq_1 * SEq_1 + SEq_2 * SEq_2 + SEq_3 * SEq_3 + SEq_4 * SEq_4);
                SEq_1 /= norm;
                SEq_2 /= norm;
                SEq_3 /= norm;
                SEq_4 /= norm;

                // Update Quaternion Estimate:
                quaternion_estimate = Eigen::Quaternion<float>(SEq_1, SEq_2, SEq_3, SEq_4);
            }

            absl::Status integrate_measurements() {
                // Precompute Values:
                Matrix3<float> C = quaternion_estimate.toRotationMatrix();
                Vector3<float> f = accelerometer_measurement - accelerometer_bias;
                Vector3<float> a = C.transpose() * f + g;

                // Integrate:
                Vector3<float> position_next = position_estimate + delta_t * velocity_estimate + 0.5 * delta_t * delta_t * a;
                Vector3<float> velocity_next = velocity_estimate + delta_t * a;
                
                // Update State:
                position_estimate = position_next;
                velocity_estimate = velocity_next;

                return absl::OkStatus();
            }


}



/* 
    Potential KF Formulation for IMU:
    Mesurements Gyroscope, Accelerometer, and Quaternion:

    x = [r, v, a, b_f] or x = [r, v, a]
    y = [a]

    r_k+1 = r_k + v_k * delta_t + 0.5 * delta_t^2 (C.T * (tilde_f_k - b_f_k) + g)
    r_k+1 = r_k + v_k * delta_t + 0.5 * delta_t^2 (C.T * a + g)

    With Bias:
    F = [
        1, deta_t, 0.5 * delta_t^2 * C.T * tild_f_k, -0.5 * delta_t^2 * C.T * b_f_k;
        0, 1, delta_t * C.T * tild_f_k, -delta_t * C.T * b_f_k;
        0, 0, 1, 0;
        0, 0, 0, 1;
    ]

    Without Bias:
    F = [
        1, deta_t, 0.5 * delta_t^2 * C.T * a;
        0, 1, delta_t * C.T * a;
        0, 0, 1;
    ]

    With Bias:
    H = [
        0, 0, 1, 1;
    ]

    Without Bias:
    H = [
        0, 0, 1;
    ]

*/