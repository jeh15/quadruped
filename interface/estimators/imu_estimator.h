#pragma once

#include <chrono>

#include "Eigen/Dense"


struct IMUState {
    Quaternion quaternion;
    Vector3 position;
    Vector3 linear_velocity;
    Vector3 gyroscope_bias;
    Vector3 accelerometer_bias;
}

class IMUEstimator {
    public:
        IMUEstimator() {};
        ~IMUEstimator() {};

        void calculate_bias() {
            std::vector<Vector3Float> gyroscope_vector;
            std::vector<Vector3Float> accelerometer_vector;
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
            int start_up_time = 5;
            Vector3Float gyroscope_bias = Vector3Float::Zero();
            Vector3Float accelerometer_bias = Vector3Float::Zero();


            void state_equation() {
                // State Equation:
                IMUState next_state;
                gyroscope_error = state.gyroscope_bias / 2 - measurements.gyroscope / 2;
                accelerometer_error = 
                next_state.quaternion(0) = state.quaternion(0) + 
            }

            template<typename T>
            void quaternion_estimation_update(const Vector3<T>& gyroscope_measurement, const Vector3<T>& accelerometer_measurement) {
                // Unpack Gyroscope and Accelerometer Measurements:
                T w_x = gyroscope_measurement(0);
                T w_y = gyroscope_measurement(1);
                T w_z = gyroscope_measurement(2);
                T a_x = accelerometer_measurement(0);
                T a_y = accelerometer_measurement(1);
                T a_z = accelerometer_measurement(2);

                // Local system variables
                T norm; // vector norm
                T SEqDot_omega_1, SEqDot_omega_2, SEqDot_omega_3, SEqDot_omega_4; // quaternion derrivative from gyroscopes elements
                T f_1, f_2, f_3; // objective function elements
                T J_11or24, J_12or23, J_13or22, J_14or21, J_32, J_33; // objective function Jacobian elements
                T SEqHatDot_1, SEqHatDot_2, SEqHatDot_3, SEqHatDot_4; // estimated direction of the gyroscope error
            
                // Axulirary variables to avoid reapeated calcualtions
                T halfSEq_1 = 0.5 * SEq_1;
                T halfSEq_2 = 0.5 * SEq_2;
                T halfSEq_3 = 0.5 * SEq_3;
                T halfSEq_4 = 0.5 * SEq_4;
                T twoSEq_1 = 2.0 * SEq_1;
                T twoSEq_2 = 2.0 * SEq_2;
                T twoSEq_3 = 2.0 * SEq_3;
            
                // Normalise the accelerometer measurement
                norm = sqrt(a_x * a_x + a_y * a_y + a_z * a_z);
                a_x /= norm;
                a_y /= norm;
                a_z /= norm;
            
                // Compute the objective function and Jacobian
                f_1 = twoSEq_2 * SEq_4 - twoSEq_1 * SEq_3 - a_x;
                f_2 = twoSEq_1 * SEq_2 + twoSEq_3 * SEq_4 - a_y;
                f_3 = 1.0 - twoSEq_2 * SEq_2 - twoSEq_3 * SEq_3 - a_z;
                J_11or24 = twoSEq_3; // J_11 negated in matrix multiplication
                J_12or23 = 2.0 * SEq_4;
                J_13or22 = twoSEq_1; // J_12 negated in matrix multiplication
                J_14or21 = twoSEq_2;
                J_32 = 2.0 * J_14or21; // negated in matrix multiplication
                J_33 = 2.0 * J_11or24; // negated in matrix multiplication
            
                // Compute the gradient (matrix multiplication)
                SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1;
                SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3;
                SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1;
                SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2;
            
                // Normalise the gradient
                norm = sqrt(SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2 + SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4);
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
                SEq_1 += (SEqDot_omega_1 - (beta * SEqHatDot_1)) * deltat;
                SEq_2 += (SEqDot_omega_2 - (beta * SEqHatDot_2)) * deltat;
                SEq_3 += (SEqDot_omega_3 - (beta * SEqHatDot_3)) * deltat;
                SEq_4 += (SEqDot_omega_4 - (beta * SEqHatDot_4)) * deltat;
            
                // Normalise quaternion
                norm = sqrt(SEq_1 * SEq_1 + SEq_2 * SEq_2 + SEq_3 * SEq_3 + SEq_4 * SEq_4);
                SEq_1 /= norm;
                SEq_2 /= norm;
                SEq_3 /= norm;
                SEq_4 /= norm;

                // Update Quaternion Estimate:
                quaternion_estimate = Quaternion(SEq_1, SEq_2, SEq_3, SEq_4);
            }
}