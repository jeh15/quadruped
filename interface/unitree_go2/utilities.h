#pragma once

#include <array>
#include <tuple>

#include "Eigen/Dense"

#include "unitree-api/containers.h"

#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"
#include "interface/unitree_go2/constants.h"

using namespace interface::aliases::common;
using namespace interface::containers::controller;


namespace interface::utilities  {

    std::tuple<unitree::containers::MotorCommand, ControlMode> get_up_routine(
        const common::MotorVector<float>& initial_position,
        const common::MotorVector<float>& desired_position,
        const float control_rate_us,
        const float duration_s = 5.0f
    ) {
        static float stand_percent = 0.0;
        if(stand_percent < 1.0) {
            std::array<float, unitree::containers::num_motors> q_setpoint;
            common::MotorVector<float> position_setpoint = 
                (1 - stand_percent) * initial_position + stand_percent * desired_position;
            Eigen::Map<common::MotorVector<float>>(q_setpoint.data()) = position_setpoint;

            // Create the motor command:
            unitree::containers::MotorCommand motor_command = {
                .q_setpoint = q_setpoint,
                .qd_setpoint = { 
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0
                },
                .torque_feedforward = { 
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0
                },
                .stiffness = {
                    35.0, 35.0, 35.0,
                    35.0, 35.0, 35.0,
                    35.0, 35.0, 35.0,
                    35.0, 35.0, 35.0
                },
                .damping = {
                    0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5
                },
            };

            float increment = duration_s / (control_rate_us * 1e-6);
            stand_percent += 1.0 / increment; 
            ControlMode mode = ControlMode::GetUp;
        }
        else {
            std::array<float, unitree::containers::num_motors> q_setpoint;
            Eigen::Map<common::MotorVector<float>>(q_setpoint.data()) = desired_position;

            unitree::containers::MotorCommand motor_command = {
                .q_setpoint = q_setpoint,
                .qd_setpoint = { 
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0
                },
                .torque_feedforward = { 
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0
                },
                .stiffness = {
                    35.0, 35.0, 35.0,
                    35.0, 35.0, 35.0,
                    35.0, 35.0, 35.0,
                    35.0, 35.0, 35.0
                },
                .damping = {
                    0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5
                },
            };

            ControlMode mode = ControlMode::Stand;
        }

        return std::make_tuple(motor_command, mode);
    }

}