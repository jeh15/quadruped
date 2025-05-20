#include <filesystem>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "GLFW/glfw3.h"

#include "interface/unitree_go2/mock_unitree_driver.h"
#include "interface/unitree_go2/policy_interface.h"

#include "unitree-api/containers.h"
#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"

using namespace interface::containers::controller;
using rules_cc::cc::runfiles::Runfiles;


// Visualization:
mjvCamera cam;
mjvPerturb pert;
mjvOption opt;
mjvScene scn;
mjrContext con;


// Helper function to convert joystick button index to string
std::string getButtonName(int buttonIndex) {
    // These names are approximate and may vary by controller
    static const std::vector<std::string> buttonNames = {
        "B", "A", "X", "Y", 
        "Extra Button", 
        "Left Bumper", "Right Bumper", "Left Trigger", "Right Trigger", 
        "Select", "Start", "Home",
        "Left Stick", "Right Stick",
        "DPad Up", "DPad Right", "DPad Down", "DPad Left"
    };

    if (buttonIndex >= 0 && buttonIndex < static_cast<int>(buttonNames.size())) {
        return buttonNames[buttonIndex];
    }
    
    return "Button " + std::to_string(buttonIndex);
}

// Helper function to convert joystick axis index to string
std::string getAxisName(int axisIndex) {
    static const std::vector<std::string> axisNames = {
        "Left Axis X", "Left Axis Y",
        "Right Axis X", "Right Axis Y"
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
        runfiles->Rlocation("unitree-interface/onnx_models/policy.onnx");


    std::filesystem::path mock_model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/scene_mjx_collision.xml");

    absl::Status result;

    // Initialize Mock Unitree Driver:
    std::shared_ptr<MockUnitreeDriver> unitree_driver = 
        std::make_shared<MockUnitreeDriver>(mock_model_path, 2000, 1);
    result.Update(unitree_driver->initialize());
    ABSL_CHECK(result.ok()) << result.message();

    // Initialize Policy Interface:
    PolicyInterface policy_interface(
        onnx_model_path,
        unitree_driver
    );
    result.Update(policy_interface.initialize());
    ABSL_CHECK(result.ok()) << result.message();

    // Expose mj_model and mj_data for visualization:
    auto mj_model = unitree_driver->mj_model;
    auto mj_data = unitree_driver->mj_data;

    // Visualization:
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

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

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultPerturb(&pert);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
    mjv_makeScene(mj_model, &scn, 1000);
    mjr_makeContext(mj_model, &con, mjFONTSCALE_100);

    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // Initialize Unitree Driver and Policy Driver threads:
    result.Update(policy_interface.initialize_thread());
    result.Update(unitree_driver->initialize_thread());
    ABSL_CHECK(result.ok()) << result.message();

    double visualization_timer = unitree_driver->mj_data->time;
    double visualization_start_time = visualization_timer;
    double visualization_interval = 0.01;

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
                    std::cout << name << std::endl;
                    if(name == "Start") {
                        std::ignore = policy_interface.set_control_mode(ControlMode::GetUp);
                    }
                    else if(name == "B") {
                        std::ignore = policy_interface.set_control_mode(ControlMode::Policy);
                    }
                    else if(name == "A") {
                        std::ignore = policy_interface.set_control_mode(ControlMode::Damping);
                    }
                    else if(name == "Select") {
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
                    if(name == "Left Axis X") {
                        lateral_command = -1 * axes[i];
                    }
                    else if(name == "Left Axis Y"){
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
        policy_interface.set_command(command);

        // Mujoco Simulation:
        visualization_timer = unitree_driver->mj_data->time - visualization_start_time;

        mj_data = unitree_driver->mj_data;
        if(visualization_timer > visualization_interval) {
            visualization_start_time = unitree_driver->mj_data->time;

            mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            // swap OpenGL buffers (blocking call due to v-sync)
            glfwSwapBuffers(window);

            // process pending GUI events, call GLFW callbacks
            glfwPollEvents();
        }
    }

    // Clean up visualization:
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Stop Threads and Clean up:
    result.Update(policy_interface.stop_thread());
    result.Update(unitree_driver->stop_thread());
    result.Update(unitree_driver->clean_up());
    ABSL_CHECK(result.ok()) << result.message();

    return 0;
}

