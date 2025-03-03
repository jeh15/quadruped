#pragma once

#include <chrono>

#include "Eigen/Dense"


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
}