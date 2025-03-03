#pragma once
#include <filesystem>

#include "interface/unitree_go2/aliases.h"


namespace interface::containers {

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
            aliases::estimator::Vector3 body_position =aliases::estimator::Vector3::Zero(); 
            aliases::estimator::Quaternion body_rotation = aliases::estimator::Quaternion::Identity();
            aliases::estimator::MotorVector motor_position = aliases::estimator::MotorVector::Zero();
            aliases::estimator::Vector3 linear_body_velocity = aliases::estimator::Vector3::Zero();
            aliases::estimator::Vector3 angular_body_velocity = aliases::estimator::Vector3::Zero();
            aliases::estimator::MotorVector motor_velocity = aliases::estimator::MotorVector::Zero();
        };

    }

    namespace mock_unitree_driver {
        
        struct MockUnitreeDriverArgs {
            std::filesystem::path xml_path;
            int control_rate_us = 1000;
        };
    
    }

}
