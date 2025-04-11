#pragma once

#include <iostream>
#include <string>
#include <array>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <numbers>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"

#include "Eigen/Dense"

#include "unitree-api/unitree_driver.h"
#include "unitree-api/containers.h"

#include "interface/unitree_go2/safety_controller.h"
#include "interface/unitree_go2/logger.h"
#include "interface/estimators/imu_estimator.h"
#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"
#include "interface/unitree_go2/constants.h"

using namespace interface::aliases::common;
using namespace interface::containers::controller;
using namespace interface::containers::logger;
using namespace interface::containers::estimator;

template <typename RobotDriver = UnitreeDriver>
class PolicyInterface {
    public:
        PolicyInterface(
            std::shared_ptr<RobotDriver> unitree_driver,
            LoggerArgs log_args
        ) : 
            unitree_driver(unitree_driver),
            logger(log_args.filepath, log_args.log_rate_us),
            enable_logging(log_args.enable_logging) {}
        ~PolicyInterface() {}

        absl::Status initialize() {
            absl::Status result;
            // Initialize Unitree Driver:
            if(!unitree_driver->is_initialized())
                result.Update(unitree_driver->initialize());
            // Initialize Policy:
            if(!policy->is_initialized())
                result.Update(initialize_policy());

            // Initialize Logger:
            if(enable_logging)
                result.Update(logger.initialize());

            ABSL_CHECK(result.ok()) << result.message();
            
            return absl::OkStatus();
        }


};