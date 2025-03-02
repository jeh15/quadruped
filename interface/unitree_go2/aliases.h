#pragma once

#include "Eigen/Dense"

#include "operational-space-control/unitree_go2/autogen/autogen_defines.h"
#include "interface/estimators/autogen/estimator_defines.h"


namespace aliases {
    namespace estimator {
            using MotorVector = Eigen::Vector<double, constants::estimator::nu>;
            using MotorVectorFloat = Eigen::Vector<float, constants::estimator::nu>;
            using Quaternion = Eigen::Vector<double, 4>;
            using QuaternionFloat = Eigen::Vector<float, 4>;
            using Vector3 = Eigen::Vector<double, 3>;
            using Vector3Float = Eigen::Vector<float, 3>;
            using SensorVector = Eigen::Vector<double, constants::estimator::sensordata_size>;
            using SensorVectorFloat = Eigen::Vector<float, constants::estimator::sensordata_size>;
            using StateVector = Eigen::Vector<double, constants::estimator::state_size>;
    }
    namespace interface {
        using ContactMask = Eigen::Vector<double, constants::model::contact_site_ids_size>;
    }
}
