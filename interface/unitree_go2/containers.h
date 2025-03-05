#pragma once

#include <filesystem>

#include "osqp++.h"

#include "interface/unitree_go2/aliases.h"


namespace interface::containers {

    namespace controller{
        struct OperationalSpaceControllerArgs {
            std::filesystem::path xml_path;
            int control_rate_us = 1000;
            osqp::OsqpSettings osqp_settings = osqp::OsqpSettings();
        };
    }
    namespace logger {

        struct LoggerArgs {
            std::filesystem::path filepath;
            int log_rate_us;
            bool enable_logging;
        };

    }

    namespace estimator {
        
        struct EstimatorArgs {
            std::filesystem::path xml_path;
            int control_rate_us = 1000;
        };

        struct EstimatorState {
            interface::aliases::common::Vector3<float> body_position;
            Eigen::Quaternion<float> body_rotation;
            interface::aliases::common::MotorVector<float> joint_position;
            interface::aliases::common::Vector3<float> linear_body_velocity;
            interface::aliases::common::Vector3<float> angular_body_velocity;
            interface::aliases::common::MotorVector<float> joint_velocity;
            interface::aliases::common::Vector3<float> linear_body_acceleration;
            interface::aliases::controller::ContactMask<float> contact_mask;
        };

    }

    namespace mock_unitree_driver {
        
        struct MockUnitreeDriverArgs {
            std::filesystem::path xml_path;
            int control_rate_us = 1000;
        };
    
    }

}
