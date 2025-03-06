#pragma once

#include <filesystem>

#include "osqp++.h"

#include "interface/unitree_go2/aliases.h"


namespace interface::containers {

    namespace controller {

        enum class ControlMode {
            Damping,
            Default,
            OperationalSpaceController
        };

        struct OperationalSpaceControllerArgs {
            std::filesystem::path xml_path;
            int control_rate_us = 1000;
            osqp::OsqpSettings osqp_settings = osqp::OsqpSettings();
        };

        struct SafetyControllerArgs {
            double stiffness = 0;
            double damping = 0;
        };
        
        struct SafetyControllerState {
            interface::aliases::common::MotorVector<double> position_setpoint = 
                interface::aliases::common::MotorVector<double>(
                    0.0, 0.9, -1.8,
                    0.0, 0.9, -1.8,
                    0.0, 0.9, -1.8,
                    0.0, 0.9, -1.8
                );
            interface::aliases::common::MotorVector<double> velocity_setpoint = 
                interface::aliases::common::MotorVector<double>::Zero();
            interface::aliases::common::MotorVector<double> torque_command = 
                interface::aliases::common::MotorVector<double>::Zero();
            double stiffness = 0.0;
            double damping = 0.0;
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
            int control_rate_us = 1000;
        };

        struct EstimatorState {
            interface::aliases::common::MotorVector<float> joint_position;
            interface::aliases::common::MotorVector<float> joint_velocity;
            interface::aliases::common::MotorVector<float> torque_estimate;
            Eigen::Quaternion<float> body_rotation;
            interface::aliases::controller::ContactMask<float> contact_mask;
            interface::aliases::common::Vector3<float> body_position;
            interface::aliases::common::Vector3<float> linear_body_velocity;
            interface::aliases::common::Vector3<float> angular_body_velocity;
            interface::aliases::common::Vector3<float> linear_body_acceleration;
        };

    }

    namespace mock_unitree_driver {
        
        struct MockUnitreeDriverArgs {
            std::filesystem::path xml_path;
            int control_rate_us = 1000;
        };
    
    }

}
