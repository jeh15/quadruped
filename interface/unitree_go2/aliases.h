#pragma once

#include "Eigen/Dense"

#include "operational-space-control/unitree_go2/constants.h"

using namespace operational_space_controller;


namespace interface::aliases {

    namespace common {
        template<typename T> 
        using Vector3 = Eigen::Vector<T, 3>;
        template<typename T>
        using Vector4 = Eigen::Vector<T, 4>;
        template<typename T>
        using MotorVector = Eigen::Vector<T, constants::model::nu_size>;
    }

    namespace estimator {
    }

    namespace controller {
        template<typename T>
        using ContactMask = Eigen::Vector<T, constants::model::contact_site_ids_size>;
    }

}
