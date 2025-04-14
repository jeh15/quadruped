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
        const MotorVector<float>& initial_position,
        const MotorVector<float>& desired_position,
        const float control_rate_us,
        const float duration_s = 3.0f
    ) {
        static float stand_percent = 0.0;

        unitree::containers::MotorCommand motor_command;
        ControlMode mode = ControlMode::Damping;

        if(stand_percent < 1.0) {
            std::array<float, unitree::containers::num_motors> q_setpoint;
            MotorVector<float> position_setpoint = 
                (1 - stand_percent) * initial_position + stand_percent * desired_position;
            Eigen::Map<MotorVector<float>>(q_setpoint.data()) = position_setpoint;

            // Create the motor command:
            motor_command.q_setpoint = q_setpoint,
            motor_command.qd_setpoint = { 
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
            };
            motor_command.torque_feedforward = { 
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
            };
            motor_command.stiffness = {
                60.0, 60.0, 60.0,
                60.0, 60.0, 60.0,
                60.0, 60.0, 60.0,
                60.0, 60.0, 60.0
            };
            motor_command.damping = {
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5
            };

            float increment = duration_s / (control_rate_us * 1e-6);
            stand_percent += 1.0 / increment; 
            mode = ControlMode::GetUp;
        }
        else {
            std::array<float, unitree::containers::num_motors> q_setpoint;
            Eigen::Map<MotorVector<float>>(q_setpoint.data()) = desired_position;

            motor_command.q_setpoint = q_setpoint,
            motor_command.qd_setpoint = { 
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
            };
            motor_command.torque_feedforward = { 
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
            };
            motor_command.stiffness = {
                60.0, 60.0, 60.0,
                60.0, 60.0, 60.0,
                60.0, 60.0, 60.0,
                60.0, 60.0, 60.0
            };
            motor_command.damping = {
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5
            };

            mode = ControlMode::Stand;
        }

        return std::make_tuple(motor_command, mode);
    }

}