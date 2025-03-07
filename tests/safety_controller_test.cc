#include <filesystem>
#include <cmath>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "GLFW/glfw3.h"

#include "interface/unitree_go2/safety_controller.h"

#include "operational-space-control/unitree_go2/constants.h"
#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"
#include "interface/unitree_go2/constants.h"

using rules_cc::cc::runfiles::Runfiles;
using namespace interface::aliases::common;


// Visualization:
mjvCamera cam;
mjvPerturb pert;
mjvOption opt;
mjvScene scn;
mjrContext con;

osc::containers::State mj_data_to_state(const mjData& mj_data) {
    osc::containers::State state;

    Vector4<double> body_rotation {mj_data.qpos[3], mj_data.qpos[4], mj_data.qpos[5], mj_data.qpos[6]};
    Vector3<double> linear_body_velocity {mj_data.qvel[0], mj_data.qvel[1], mj_data.qvel[2]};
    Vector3<double> angular_body_velocity {mj_data.qvel[3], mj_data.qvel[4], mj_data.qvel[5]};

    state.body_rotation = body_rotation;
    state.linear_body_velocity = linear_body_velocity;
    state.angular_body_velocity = angular_body_velocity;
    state.motor_position = Eigen::Map<MotorVector<double>>(mj_data.qpos + 7);
    state.motor_velocity = Eigen::Map<MotorVector<double>>(mj_data.qvel + 6);
    state.torque_estimate = Eigen::Map<MotorVector<double>>(mj_data.qfrc_actuator);
    // Unused Body Acceleration and Contact Mask:
    state.linear_body_acceleration = Vector3<double>::Zero();
    state.contact_mask = Vector4<double>::Zero();
    return state;
}

int main(int argc, char** argv) {
    // Use Runfiles to get the path to the model:
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/scene_go2.xml");

    // Safety Controller Args:
    interface::containers::controller::SafetyControllerArgs safety_args = {
        .stiffness = 0.0,
        .damping = 5.0,
    };

    // Initialize Mujoco Model:
    mjModel* mj_model = mj_loadXML(model_path.c_str(), NULL, NULL, 0);
    char mj_error[1000];
    mj_model = mj_loadXML(model_path.c_str(), nullptr, mj_error, 1000);
    if( !mj_model ) {
        printf("%s\n", mj_error);
    }

    mjData* mj_data = mj_makeData(mj_model);

    mj_data->qpos = mj_model->key_qpos;
    mj_data->qvel = mj_model->key_qvel;
    mj_data->ctrl = mj_model->key_ctrl;

    mj_forward(mj_model, mj_data);

    // Initialize Interface:
    SafetyController safety_controller(
        safety_args.stiffness, safety_args.damping
    );

    // Visualization:
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

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


    // Test 1: Safety Controller in Default Position
    MotorVector<double> torque = MotorVector<double>::Zero();
    osc::containers::State state = mj_data_to_state(*mj_data);

    // Visualize:
    mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    glfwSwapBuffers(window);
    glfwPollEvents();

    absl::Status result;
    result.Update(safety_controller.torque_saturator(torque, state));
    std::cout << "Status Message: " << result.message() << std::endl;

    // Test 2: Safety Controller in soft limit position:
    // Lower Soft with deviation:
    // Eigen::Vector<double, 19> new_pose {
    //     0.0, 0.0, 0.285, 1, 0, 0, 0,
    //     0.0, 0.9 , -1.8,
    //     -0.5236 - 0.1 , -0.3354 - 0.1, -2.26135 - 0.1,
    //     0.5236 + 0.1 ,  0.1882 - 0.1 , -2.26135 - 0.1,
    //     -0.5236 - 0.1,  0.1882 - 0.1, -2.26135 - 0.1
    // };

    // Lower Hard:
    Eigen::Vector<double, 19> new_pose {
        0.0, 0.0, 0.285, 1, 0, 0, 0,
        0.0, 0.9 , -1.8,
        -0.83776, -1.07664, -2.53816,
        0.83776, -0.23888, -2.53816,
        -0.83776, -0.23888, -2.53816
    };

    // Upper Soft:
    // Eigen::Vector<double, 19> new_pose {
    //     0.0, 0.0, 0.285, 1, 0, 0, 0,
    //     0.0, 0.9 , -1.8,
    //     0.5236, 2.19535, -1.31888,
    //     -0.5236, 2.71895, -1.31888,
    //     0.5236, 2.71895, -1.31888
    // };

    // Upper Hard:
    // Eigen::Vector<double, 19> new_pose {
    //     0.0, 0.0, 0.285, 1, 0, 0, 0,
    //     0.0, 0.9 , -1.8,
    //     0.83776, 2.97256, -1.030208,
    //     -0.83776, 2.97256, -1.030208,
    //     0.83776, 2.97256, -1.030208
    // };


    mj_data->qpos = new_pose.data();
    mj_forward(mj_model, mj_data);
    state = mj_data_to_state(*mj_data);

    // Visualize:
    cam.azimuth = 180 + 60;
    mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    glfwSwapBuffers(window);
    glfwPollEvents();

    result.Update(safety_controller.torque_saturator(torque, state));
    std::cout << "Status Message: " << result.message() << std::endl;

    // Clean up visualization:
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Clean up Mujoco model and data:
    mj_deleteData(mj_data);
    mj_deleteModel(mj_model);

    return 0;
}