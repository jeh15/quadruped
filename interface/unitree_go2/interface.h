#pragma once

#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

#include "absl/status/status.h"

#include "osqp++.h"

#include "operational_space_controller.h"
#include "unitree_go2_api.h"


class UnitreeGo2Interface {
    public:
        UnitreeGo2Interface() {}
        ~UnitreeGo2Interface() {}

        absl::Status initialize_operational_space_controller(const std::filesystem::path xml_path, const State initial_state, const int control_rate = 1000, const osqp::OsqpSettings osqp_settings = osqp::OsqpSettings()) {
            /*
                Initialize OperationalSpaceController:

                args:
                    initial_state: State - Initial Motor / Robot State
                    control_rate: int - Cotrol rate target of inner control loop in microseconds
                    osqp_settings: OsqpSettings - Settings for OSQP solver
                
                returns:
                    absl::Status - Status of initialization

            */
            // Call Costructor:
            operational_space_controller = OperationalSpaceController(initial_state, control_rate, osqp_settings);
            // Load Mujoco Model:
            operational_space_controller.initialize(xml_path);
            operational_space_controller_initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_motor_controller(const std::string& network_name, const int control_rate = 2000) {
            /*
                Initialize MotorController:

                args:
                    control_rate: int - Cotrol rate target of inner communication control loop in microseconds
                
                returns:
                    absl::Status - Status of initialization

            */
            motor_controller = MotorController(control_rate);
            motor_controller.initialize(network_name);
            motor_controller_initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_operational_space_controller_thread() {
            if (!operational_space_controller_initialized) {
                return absl::FailedPreconditionError("Operational Space Controller not initialized");
            }
            operational_space_controller.initialize_control_thread();
            return absl::OkStatus();
        }

        absl::Status initialize_motor_controller_thread() {
            if (!motor_controller_initialized) {
                return absl::FailedPreconditionError("Motor Controller not initialized");
            }
            motor_controller.initialize_control_thread();
            return absl::OkStatus();
        }

        absl::Status initialize_control_thread() {
            if (!operational_space_controller_initialized && !motor_controller_initialized)
                return absl::FailedPreconditionError("Operational Space Controller and/or Motor Controller not initialized");
            thread = std::thread(&UnitreeGo2Interface::control_loop, this);
            control_thread_initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_threads() {
            // Initialize all threads:
            absl::Status result = initialize_operational_space_controller_thread();
            if (!result.ok())
                return result;

            result = initialize_motor_controller_thread();
            if (!result.ok())
                return result;

            result = initialize_control_thread();
            if (!result.ok())
                return result;

            return absl::OkStatus();
        }

        absl::Status stop_control_thread() {
            if (!control_thread_initialized) {
                return absl::FailedPreconditionError("Control Thread not initialized");
            }
            running = false;
            thread.join();
            return absl::OkStatus();
        }

        absl::Status stop_child_threads() {
            if (operational_space_controller_initialized)
                operational_space_controller.stop_control_thread();

            if (motor_controller_initialized)
                motor_controller.stop_control_thread();

            if (!operational_space_controller_initialized || !motor_controller_initialized)
                return absl::FailedPreconditionError("Operational Space Controller and/or Motor Controller not initialized");

            return absl::OkStatus();
        }

        absl::Status stop_threads() {
            // Stop all threads:
            absl::Status result = stop_control_thread();
            if (!result.ok())
                return result;
            
            result = stop_child_threads();
            if (!result.ok())
                return result;

            return absl::OkStatus();
        }

        absl::Status clean_up() {
            if(!operational_space_controller_initialized) {
                return absl::FailedPreconditionError("Operational Space Controller not initialized. Nothing to clean up.");
            }
            operational_space_controller.close();            
            return absl::OkStatus();
        }

    private:
        OperationalSpaceController operational_space_controller;
        MotorController motor_controller;
        bool operational_space_controller_initialized = false;
        bool motor_controller_initialized = false;
        // Thread Variables:
        std::atomic<bool> running{true};
        std::thread thread;
        std::mutex mutex;
        bool control_thread_initialized = false;
        
        void control_loop() {
        }
};
