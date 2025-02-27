#include <filesystem>
#include <chrono>


class EstimatorInterface {
    public:
        EstimatorInterface(const std::filesystem::path xml_path) : mjpc::Kalman(xml_path) {}
        ~EstimatorInterface() {};

        absl::Status initialize_estimator_thread() {
            thread = std::thread(&EstimatorInterface::estimator_loop, this);
            estimator_thread_initialized = true;
            return absl::OkStatus();
        }

        absl::Status stop_estimator_thread() {
            if(!estimator_thread_initialized)
                return absl::FailedPreconditionError("Estimator thread not initialized");

            running = false;
            thread.join();
            return absl::OkStatus();
        }

        absl::Status update_sensor(){
            
        }
        
    private:
        /* Estimator */
        mjpc::Estimator* estimator;
        /* Thread */
        std::atomic<bool> running = true;
        std::thread thread;
        std::mutex mutex;
        bool estimator_thread_initialized = false;

        void estimator_loop() {
            using Clock = std::chrono::steady_clock;
            auto start = Clock::now();
            auto next_time = Clock::now();
            while(running) {
                // Calculate next time:
                next_time += std::chrono::milliseconds(control_rate_us);
                /* Lock Guard Scope */
                {
                    std::lock_guard<std::mutex> lock(mutex);

                    // Update estimator:
                    estimator->Update(ctrl.data(), sensor.data())

                    // Get and Set State:
                    double* state = estimator->State();
                    update_state(state);
                }
            }
        }
};
