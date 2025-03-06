#pragma once

#include "operational-space-control/unitree_go2/containers.h"
#include "unitree-api/containers.h"

#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"

namespace interface::constants {

    namespace controller {

        unitree::containers::MotorCommand damping_motor_command = {
            .q_setpoint = {
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8
            },
            .qd_setpoint = { 0 },
            .torque_feedforward = { 0 },
            .stiffness = { 0 },
            .damping = {
                20.0, 20.0, 20.0,
                20.0, 20.0, 20.0,
                20.0, 20.0, 20.0,
                20.0, 20.0, 20.0
            }
        };

    }

}