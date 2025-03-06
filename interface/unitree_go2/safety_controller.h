#pragma once

#include <array>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"

#include "Eigen/Dense"

#include "operational-space-control/unitree_go2/constants.h"

#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"



using namespace interface::aliases::common;
using namespace interface::containers::controller;
namespace osc = operational_space_controller;


namespace {

    template <typename T> int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

}


class SafetyController {
    public:
        SafetyController(const double stiffness, const double damping) : 
            stiffness_default(stiffness), damping_default(damping) {}
        ~SafetyController() {}

        absl::Status torque_saturator(const MotorVector<double>& torque_feedforward, const osc::containers::State& state) {
            /*
                Saturates control input if past soft constraint 
                and overrides control if past the hard constraint.
            */
            Eigen::Vector<double, osc::constants::model::nu_size> position_command = 
                Eigen::Vector<double, osc::constants::model::nu_size>::Zero();
            Eigen::Vector<double, osc::constants::model::nu_size> velocity_command = 
                Eigen::Vector<double, osc::constants::model::nu_size>::Zero();
            double kp = 0.0;
            double kd = 0.0;
            for(int i = 0; i < osc::constants::model::nu_size; i++){
                double motor_position = state.motor_position[i];
                double motor_velocity = state.motor_velocity[i];
                if(motor_position > upper_soft[i]) {
                    if(motor_position >= upper_hard[i]) {
                        return absl::InternalError("Motor Position Exceeded Upper Bound");
                    }
                    kp = kp_lb + (abs(motor_position) - abs(upper_soft[i])) * (kp_ub - kp_lb) / ( abs(upper_hard[i]) - abs(upper_soft[i]));
                    position_command(i) = kp * (upper_soft[i] - motor_position);
                }
                else if(motor_position < lower_soft[i]) {
                    if(motor_position >= lower_hard[i]) {
                        return absl::InternalError("Motor Position Exceeded Lower Bound");
                    }
                    kp = kp_lb + (abs(motor_position) - abs(lower_soft[i])) * (kp_ub - kp_lb) / ( abs(lower_hard[i]) - abs(lower_soft[i]));
                    position_command(i) = kp * (lower_soft[i] - motor_position);
                }

                if(abs(motor_velocity) > velocity_soft[i]) {
                    if(abs(motor_velocity) >= velocity_hard[i]) {
                        return absl::InternalError("Motor Velocity Exceeded Limit");
                    }
                    kd = kd_lb + (motor_velocity - velocity_soft[i]) * (kd_ub - kd_lb) / (velocity_hard[i] - velocity_soft[i]);
                    double velocity_setpoint = sgn<double>(motor_velocity) * velocity_soft[i];
                    velocity_command(i) = kd * (velocity_setpoint - motor_velocity);
                }
            }

            // Add Safety Controller to Torque Command:
            torque_command = torque_feedforward + position_command + velocity_command;
            
            // Saturate Torque Command:
            torque_command = torque_command.cwiseMin(torque_ub).cwiseMax(torque_lb);

            return absl::OkStatus();
        }

        absl::Status setpoint_saturator(const MotorVector<double>& position_setpoints, const MotorVector<double>& velocity_setpoints) {
            // Saturates setpoints if past hard constraint.
            position_setpoint = position_setpoints.cwiseMin(upper_hard).cwiseMax(lower_hard);
            velocity_setpoint = velocity_setpoints.cwiseMin(velocity_hard).cwiseMax(-velocity_hard);

            stiffness = stiffness_value;
            damping = damping_value;

            return absl::OkStatus();
        }

        absl::Status setpoint_override() {
            // Override setpoints if past hard constraint.
            position_setpoint = default_position;
            velocity_setpoint = MotorVector<double>::Zero();
            torque_command = MotorVector<double>::Zero();
            
            stiffness = stiffness_override;
            damping = damping_override;

            stop_control_ = true;

            return absl::OkStatus();
        }

        SafetyControllerState get_state() {
            SafetyControllerState state;
            state.position_setpoint = position_setpoint;
            state.velocity_setpoint = velocity_setpoint;
            state.torque_command = torque_command;
            state.stiffness = stiffness;
            state.damping = damping;
            return state;
        }

        bool stop_control() {
            return stop_control_;
        }

        private:
            /* Motor Commands */
            MotorVector<double> position_setpoint = MotorVector<double>::Zero();
            MotorVector<double> velocity_setpoint = MotorVector<double>::Zero();
            MotorVector<double> torque_command = MotorVector<double>::Zero();
            double stiffness_default;
            double damping_default;
            /* Override */
            bool stop_control_ = false;
            MotorVector<double> default_position = MotorVector<double>(
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8
            );
            double stiffness_override = 20;
            double damping_override = 10;
            /* Safety Controller Variables */
            double kp_lb = 2.0;
            double kp_ub = 20.0;
            double kd_lb = 2.0;
            double kd_ub = 10.0;
            // Position Soft and Hard Limits:
            MotorVector<double> lower_soft = MotorVector<double>(
                -0.546, -0.5708, -1.7227,
                -0.546, -0.5708, -1.7227,
                -0.546, -0.0, -1.7227,
                -0.546, -0.0, -1.7227
            );
            MotorVector<double> lower_hard = MotorVector<double>(
                -0.8472, -1.3708, -2.5227,
                -0.8472, -1.3708, -2.5227,
                -0.8472, -0.3236, -2.5227,
                -0.8472, -0.3236, -2.5227
            );
            MotorVector<double> upper_soft = MotorVector<double>(
                0.546, 2.4907, 0.162,
                0.546, 2.4907, 0.162,
                0.546, 3.5379, 0.162,
                0.546, 3.5379, 0.162,
            );
            MotorVector<double> upper_hard = MotorVector<double>(
                0.8472, 3.2907, -0.63776,
                0.8472, 3.2907, -0.63776,
                0.8472, 4.3379, -0.63776,
                0.8472, 4.3379, -0.63776,
            );
            // Velocity Soft and Hard Limits:
            double v_lb = std::numbers::pi;
            double v_ub = 2 * std::numbers::pi;
            MotorVector<double> velocity_soft = MotorVector<double>(
                v_lb, v_lb, v_lb,
                v_lb, v_lb, v_lb,
                v_lb, v_lb, v_lb,
                v_lb, v_lb, v_lb,
            );
            MotorVector<double> velocity_hard = MotorVector<double>(
                v_ub, v_ub, v_ub,
                v_ub, v_ub, v_ub,
                v_ub, v_ub, v_ub,
                v_ub, v_ub, v_ub,
            );
            // Torque Saturation Limits:
            double torque_ub = 10.0;
            double torque_lb = -10.0;

};