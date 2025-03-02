#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "absl/log/log.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/log/initialize.h"
#include "absl/strings/string_view.h"

#include "operational-space-control/unitree_go2/operational_space_controller.h"


class FileLogSink : public absl::LogSink {
    public:
        explicit FileLogSink(const std::string& filename) {
            file_.open(filename, std::ios::out | std::ios::app);
            if(!file_.is_open()) {
                LOG(ERROR) << "Failed to open log file: " << filename;
            }
        }

        ~FileLogSink() override {
            if(file_.is_open()) {
                file_.close();
            }
        }

        void Send(const absl::LogEntry& entry) override {
            if(!file_.is_open()) return;
            
            // Convert string_view to std::string if needed
            absl::string_view text_view = entry.text_message();
            std::string text(text_view);
            
            // Format manually instead of using FormatLogEntry
            file_ << entry.timestamp() << " [" << absl::LogSeverityName(entry.log_severity()) 
                    << "] " << text << std::endl;
            
            file_.flush();
        }

    private:
        std::ofstream file_;
};


struct StateLoggerArgs {
    std::filesystem::path filepath;
    int log_rate_us;
    bool enable_logging;
};


class StateLogger {
    public:
        StateLogger(const std::filesystem::path& filepath, const int log_rate_us) : 
            filepath(filepath), log_rate_us(log_rate_us) {}
        ~StateLogger() {}

        absl::Status initialize() {
            // Initialize Logger:
            file_sink = std::make_unique<FileLogSink>(filepath.c_str());
            absl::AddLogSink(file_sink.get());
            absl::InitializeLog(); 
            LOG(INFO) << "Logger Initialized";
            log_initialized = true;
            return absl::OkStatus();
        }

        absl::Status initialize_log_thread() {
            if(!log_initialized) {
                return absl::FailedPreconditionError("Logger not initialized");
            }

            // Start Logger Thread:
            thread = std::thread(&StateLogger::log_loop, this);
            log_thread_initialized = true;
            return absl::OkStatus();
        }

        absl::Status stop_log_thread() {
            if(!log_thread_initialized) {
                return absl::FailedPreconditionError("Log Thread not initialized");
            }

            running = false;
            thread.join();
            return absl::OkStatus();
        }

        absl::Status update_state(const State& new_state) {
            std::lock_guard<std::mutex> lock(mutex);
            state = new_state;
            return absl::OkStatus();
        }
    
    private:
        /* Shared Variables */
        State state;
        /* Log Variables */
        std::filesystem::path filepath;
        std::unique_ptr<FileLogSink> file_sink;
        bool log_initialized = false;
        /* Thread Variables */
        std::atomic<bool> running{true};
        std::thread thread;
        std::mutex mutex;
        bool log_thread_initialized = false;
        int log_rate_us;

        absl::Status log_state() {
            // Log State:
            LOG(INFO) << "Motor Position (rad): " << state.motor_position.transpose();
            LOG(INFO) << "Motor Velocity (rad/s): " << state.motor_velocity.transpose();
            LOG(INFO) << "Motor Acceleration (rad/s^2): " << state.motor_acceleration.transpose();
            LOG(INFO) << "Torque Estimate (Nm): " << state.torque_estimate.transpose();
            LOG(INFO) << "Body Rotation (Quaternion): " << state.body_rotation.transpose();
            LOG(INFO) << "Angular Body Velocity (rad/s): " << state.angular_body_velocity.transpose();
            LOG(INFO) << "Linear Body Velocity (m/s): " << state.linear_body_velocity.transpose();
            LOG(INFO) << "Linear Body Acceleration (m/s^2): " << state.linear_body_acceleration.transpose();
            LOG(INFO) << "Contact Mask: " << state.contact_mask.transpose();

            return absl::OkStatus();
        }

        void log_loop() {
            using Clock = std::chrono::steady_clock;
            auto next_time = Clock::now();
            while(running) {
                // Calculate next time:
                next_time += std::chrono::microseconds(log_rate_us);

                /* Lock Guard Scope */
                {
                    std::lock_guard<std::mutex> lock(mutex);

                    // Log State:
                    absl::Status result = log_state();
                    if(!result.ok()) {
                        LOG(ERROR) << "Failed to log state";
                    }

                }

                // Check for overrun and sleep until next time:
                auto now = Clock::now();
                if(now < next_time) {
                    std::this_thread::sleep_until(next_time);
                } else {
                    // Log overrun:
                    auto overrun = std::chrono::duration_cast<std::chrono::microseconds>(now - next_time);
                    std::cout << "Log Loop Execution Time Exceeded Log Rate: " 
                        << overrun.count() << "us" << std::endl;
                    next_time = now;
                }
            }
        }
};
