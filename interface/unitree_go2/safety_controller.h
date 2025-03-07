#pragma once

#include <array>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"

#include "Eigen/Dense"

#include "operational-space-control/unitree_go2/containers.h"

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
                double abs_motor_velocity = std::abs(motor_velocity);
                if(motor_position >= upper_soft[i]) {
                    if(motor_position >= upper_hard[i]) {
                        return absl::InternalError("Motor Position Exceeded Upper Bound");
                    }
                    kp = kp_lb + (std::abs(motor_position - upper_soft[i])) * (kp_ub - kp_lb) / (std::abs(upper_hard[i] - upper_soft[i]));
                    position_command(i) = kp * (upper_soft[i] - motor_position);
                }
                else if(motor_position <= lower_soft[i]) {
                    if(motor_position <= lower_hard[i]) {
                        return absl::InternalError("Motor Position Exceeded Lower Bound");
                    }
                    kp = kp_lb + (std::abs(motor_position - lower_soft[i])) * (kp_ub - kp_lb) / (std::abs(lower_hard[i] - lower_soft[i]));
                    position_command(i) = kp * (lower_soft[i] - motor_position);
                }

                if(abs_motor_velocity >= velocity_soft[i]) {
                    if(abs_motor_velocity >= velocity_hard[i]) {
                        return absl::InternalError("Motor Velocity Exceeded Limit");
                    }
                    kd = kd_lb + (abs_motor_velocity - velocity_soft[i]) * (kd_ub - kd_lb) / (velocity_hard[i] - velocity_soft[i]);
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

            stiffness = stiffness_default;
            damping = damping_default;

            return absl::OkStatus();
        }

        absl::Status setpoint_override() {
            // Override setpoints if past hard constraint.
            position_setpoint = default_position;
            velocity_setpoint = MotorVector<double>::Zero();
            torque_command = MotorVector<double>::Zero();
            
            stiffness = stiffness_override;
            damping = damping_override;

            stop_flag = true;

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

        bool safety_stop() {
            return stop_flag;
        }

        private:
            /* Motor Commands */
            MotorVector<double> position_setpoint = MotorVector<double>::Zero();
            MotorVector<double> velocity_setpoint = MotorVector<double>::Zero();
            MotorVector<double> torque_command = MotorVector<double>::Zero();
            double stiffness_default;
            double damping_default;
            double stiffness = stiffness_default;
            double damping = damping_default;
            /* Override */
            bool stop_flag = false;
            MotorVector<double> default_position {
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8,
                0.0, 0.9, -1.8
            };
            double stiffness_override = 0.0;
            double damping_override = 20;
            /* Safety Controller Variables */
            double kp_lb = 2.0;
            double kp_ub = 20.0;
            double kd_lb = 2.0;
            double kd_ub = 10.0;
            /* 
            Position Soft and Hard Limits: 
                Note: These poses are already extreme and should probably result in damping much sooner...
                Soft limits [50%] from hardware limits relative to default position
                Hard limits [10%] from hardware limits relative to default position
            */
            // Mirrored Poses:
            // MotorVector<double> lower_soft {
            //     0.5236 , -0.3354 , -2.26135,
            //     -0.5236 , -0.3354 , -2.26135,
            //     0.5236 ,  0.1882 , -2.26135,
            //     -0.5236 ,  0.1882 , -2.26135
            // };
            // MotorVector<double> lower_hard {
            //     0.83776, -1.07664, -2.53816,
            //     -0.83776, -1.07664, -2.53816,
            //     0.83776, -0.23888, -2.53816,
            //     -0.83776, -0.23888, -2.53816
            // };
            // MotorVector<double> upper_soft {
            //     -0.5236, 2.19535, -1.31888,
            //     0.5236, 2.19535, -1.31888,
            //     -0.5236, 2.71895, -1.31888,
            //     0.5236, 2.71895, -1.31888
            // };
            // MotorVector<double> upper_hard {
            //     -0.83776, 2.97256, -1.030208,
            //     0.83776, 2.97256, -1.030208,
            //     -0.83776, 2.97256, -1.030208,
            //     0.83776, 2.97256, -1.030208
            // };
            // Non-Mirrored Poses:
            MotorVector<double> lower_soft {
                -0.5236, -0.3354, -2.26135,
                -0.5236, -0.3354, -2.26135,
                -0.5236,  0.1882, -2.26135,
                -0.5236,  0.1882, -2.26135
            };
            MotorVector<double> lower_hard {
                -0.83776, -1.07664, -2.53816,
                -0.83776, -1.07664, -2.53816,
                -0.83776, -0.23888, -2.53816,
                -0.83776, -0.23888, -2.53816
            };
            MotorVector<double> upper_soft {
                0.5236, 2.19535, -1.31888,
                0.5236, 2.19535, -1.31888,
                0.5236, 2.71895, -1.31888,
                0.5236, 2.71895, -1.31888
            };
            MotorVector<double> upper_hard {
                0.83776, 2.97256, -1.030208,
                0.83776, 2.97256, -1.030208,
                0.83776, 2.97256, -1.030208,
                0.83776, 2.97256, -1.030208
            };

            // Velocity Soft and Hard Limits:
            double v_lb = 1.0 * std::numbers::pi;
            double v_ub = 2.0 * std::numbers::pi;
            MotorVector<double> velocity_soft {
                v_lb, v_lb, v_lb,
                v_lb, v_lb, v_lb,
                v_lb, v_lb, v_lb,
                v_lb, v_lb, v_lb,
            };
            MotorVector<double> velocity_hard {
                v_ub, v_ub, v_ub,
                v_ub, v_ub, v_ub,
                v_ub, v_ub, v_ub,
                v_ub, v_ub, v_ub,
            };
            // Torque Saturation Limits:
            double torque_ub = 20.0;
            double torque_lb = -20.0;

};