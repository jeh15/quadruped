#pragma once

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
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0
            }
        };

        unitree::containers::MotorCommand stand_motor_command = {
            .q_setpoint = {
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8
            },
            .qd_setpoint = { 0 },
            .torque_feedforward = { 0 },
            .stiffness = { 
                60.0, 60.0, 60.0,
                60.0, 60.0, 60.0,
                60.0, 60.0, 60.0,
                60.0, 60.0, 60.0
            },
            .damping = {
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5
            }
        };

    }

}