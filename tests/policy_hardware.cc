#include <filesystem>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "Eigen/Dense"
#include "GLFW/glfw3.h"

#include "unitree-api/unitree_driver.h"
#include "interface/unitree_go2/policy_interface.h"

#include "unitree-api/containers.h"
#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"

using namespace interface::containers::controller;
using rules_cc::cc::runfiles::Runfiles;


// Helper function to convert joystick button index to string
std::string getButtonName(int buttonIndex) {
    // These names are approximate and may vary by controller
    static const std::vector<std::string> buttonNames = {
        "A", "B", "X", "Y", 
        "Left Bumper", "Right Bumper",
        "Back/Select", "Start", 
        "Guide/Home", 
        "Left Stick Press", "Right Stick Press",
        "D-pad Up", "D-pad Right", "D-pad Down", "D-pad Left"
    };
    
    if (buttonIndex >= 0 && buttonIndex < static_cast<int>(buttonNames.size())) {
        return buttonNames[buttonIndex];
    }
    
    return "Button " + std::to_string(buttonIndex);
}

// Helper function to convert joystick axis index to string
std::string getAxisName(int axisIndex) {
    static const std::vector<std::string> axisNames = {
        "DPad X", "DPad Y",
        "Left Trigger",
        "Right Axis X", "Right Axis Y",
        "Right Trigger",
    };
    
    if (axisIndex >= 0 && axisIndex < static_cast<int>(axisNames.size())) {
        return axisNames[axisIndex];
    }
    
    return "Axis " + std::to_string(axisIndex);
}

// Joystick button callback
void joystickCallback(int jid, int event) {
    if (event == GLFW_CONNECTED) {
        std::cout << "Controller connected: " << glfwGetJoystickName(jid) << std::endl;
    } else if (event == GLFW_DISCONNECTED) {
        std::cout << "Controller disconnected." << std::endl;
    }
}


int main(int argc, char** argv) {
    // Use Runfiles to get the path to the model:
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path onnx_model_path = 
        runfiles->Rlocation("unitree-interface/onnx_models/genial-breeze-150.onnx");

    absl::Status result;

    // Initialize Unitree Driver:
    std::string network_name = "eno2";
    int control_rate = 2000;
    std::shared_ptr<UnitreeDriver> unitree_driver = 
        std::make_shared<UnitreeDriver>(network_name, control_rate);
    result.Update(unitree_driver->initialize());
    ABSL_CHECK(result.ok()) << result.message();

    // Initialize Policy Interface:
    PolicyInterface policy_interface(
        onnx_model_path,
        unitree_driver
    );
    result.Update(policy_interface.initialize());
    ABSL_CHECK(result.ok()) << result.message();

    // Initialize GLFW for Joystick Control:
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "Hardware Control", NULL, NULL);
    glfwMakeContextCurrent(window);

    // Set Joystick Callback:
    glfwSetJoystickCallback(joystickCallback);

    // Find a connected joystick/gamepad
    int joystickId = -1;
    for (int i = GLFW_JOYSTICK_1; i <= GLFW_JOYSTICK_LAST; i++) {
        if (glfwJoystickPresent(i)) {
            joystickId = i;
            std::cout << "Found controller: " << glfwGetJoystickName(i) << std::endl;
            break;
        }
    }
    
    if (joystickId == -1) {
        std::cout << "No controller detected. Please connect a controller and restart." << std::endl;
    }

    // Store previous button states to detect changes
    std::vector<unsigned char> prevButtonStates;

    // Initialize Motor Command to be in Damping Mode:
    unitree::containers::MotorCommand motor_command;
    motor_command = interface::constants::controller::damping_motor_command;
    unitree_driver->update_command(motor_command);

    // Initialize Unitree Driver and Policy Driver threads:
    result.Update(policy_interface.initialize_thread());
    result.Update(unitree_driver->initialize_thread());
    ABSL_CHECK(result.ok()) << result.message();

    bool terminate = false;
    while(!terminate) {
        // Joystick:
        glfwPollEvents();

        float forward_command = 0.0;
        float lateral_command = 0.0;
        float yaw_command = 0.0;

        // Check if we have a controller connected
        if (joystickId != -1 && glfwJoystickPresent(joystickId)) {
            // Get button states
            int buttonCount;
            const unsigned char* buttons = glfwGetJoystickButtons(joystickId, &buttonCount);
            
            // Initialize previous button states if needed
            if (prevButtonStates.empty() && buttonCount > 0) {
                prevButtonStates.resize(buttonCount, GLFW_RELEASE);
            }
            
            // Check for button presses
            for (int i = 0; i < buttonCount; i++) {
                if (buttons[i] == GLFW_PRESS && prevButtonStates[i] == GLFW_RELEASE) {
                    std::string name = getButtonName(i);
                    if(name == "Start") {
                        std::ignore = policy_interface.set_control_mode(ControlMode::GetUp);
                    }
                    else if(name == "A") {
                        std::ignore = policy_interface.set_control_mode(ControlMode::Policy);
                    }
                    else if(name == "B") {
                        std::ignore = policy_interface.set_control_mode(ControlMode::Damping);
                    }
                    else if(name == "Back/Select") {
                        std::ignore = policy_interface.set_control_mode(ControlMode::Damping);
                        terminate = true;
                    }
                }
                prevButtonStates[i] = buttons[i];
            }

            // Get axis values
            int axisCount;
            const float* axes = glfwGetJoystickAxes(joystickId, &axisCount);
            
            // Only print significant movements to reduce output spam
            const float deadzone = 0.25f;
            for (int i = 0; i < axisCount; i++) {
                std::string name = getAxisName(i);
                if (std::abs(axes[i]) > deadzone) {
                    if(name == "DPad X") {
                        lateral_command = -1 * axes[i];
                    }
                    else if(name == "DPad Y"){
                        forward_command = -1 * axes[i];
                    }
                    else if(name == "Right Axis X"){
                        yaw_command = -1 * axes[i];
                    }
                }
            }
        } 
        
        // Set Command:
        Vector3<float> command = Vector3<float>(
            forward_command,
            lateral_command,
            yaw_command
        );
        std::ignore = policy_interface.set_command(command);
    }

    // Clean up GLFW:
    glfwTerminate();

    // Stop Threads and Clean up:
    result.Update(policy_interface.stop_thread());
    result.Update(unitree_driver->stop_thread());
    ABSL_CHECK(result.ok()) << result.message();

    return 0;
}

