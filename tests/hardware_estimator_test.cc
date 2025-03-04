#include <filesystem>
#include <iostream>
#include <string>

#include "absl/status/status.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "Eigen/Dense"
#include "Eigen/Geometry"

#include "unitree-api/unitree_driver.h"
#include "interface/estimators/imu_estimator.h"

#include "interface/unitree_go2/aliases.h"
#include "interface/unitree_go2/containers.h"

// Visualize Estimate
#include "mujoco/mujoco.h"
#include "GLFW/glfw3.h"


using rules_cc::cc::runfiles::Runfiles;


mjvCamera cam;
mjvPerturb pert;
mjvOption opt;
mjvScene scn;
mjrContext con;


int main(int argc, char** argv) {
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );
    // Initialize Driver and Estimator:
    std::string network_name = "eno2";
    int control_rate = 2000;
    absl::Status result;
    std::shared_ptr<UnitreeDriver> unitree_driver = std::make_shared<UnitreeDriver>(network_name, control_rate);
    result.Update(unitree_driver->initialize());

    std::cout << "Unitree Driver Initialized" << std::endl;

    int estimator_control_rate = 1000;
    IMUEstimator<UnitreeDriver> estimator_interface(unitree_driver, estimator_control_rate);

    // Initialize Estimator:
    result.Update(estimator_interface.initialize());

    std::cout << "Estimator Initialized" << std::endl;

    // Print New State:
    auto state = estimator_interface.get_state();
    std::cout << "Estimator State: " << std::endl;
    std::cout << "Body Position: " << state.body_position.transpose() << std::endl;
    std::cout << "Body Rotation: " << state.body_rotation.w() << " " << state.body_rotation.vec().transpose() << std::endl;
    std::cout << "Joint Position: " << state.joint_position.transpose() << std::endl;
    std::cout << "Linear Body Velocity: " << state.linear_body_velocity.transpose() << std::endl;
    std::cout << "Angular Body Velocity: " << state.angular_body_velocity.transpose() << std::endl;
    std::cout << "Joint Velocity: " << state.joint_velocity.transpose() << std::endl;
    std::cout << "Linear Body Acceleration: " << state.linear_body_acceleration.transpose() << std::endl;
    std::cout << "Contact Mask: " << state.contact_mask.transpose() << std::endl;

    // Visualize Estimation:
    std::filesystem::path model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/go2_estimation.xml");

    mjModel* mj_model = mj_loadXML(model_path.c_str(), nullptr, nullptr, 1000);
    mjData* mj_data = mj_makeData(mj_model);

    Eigen::Vector<double, 19> initial_qpos;
    Eigen::Vector<double, 3> body_position = state.body_position.cast<double>();
    Eigen::Vector<double, 4> body_rotation = state.body_rotation.cast<double>().coeffs();
    Eigen::Vector<double, 12> joint_position = state.joint_position.cast<double>();
    initial_qpos << body_position, body_rotation, joint_position;

    mj_data->qpos = initial_qpos.data();

    mj_forward(mj_model, mj_data);

    // Initialize Visualization:
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

    // Initialize Estimator Thread:
    result.Update(estimator_interface.initialize_estimator_thread());

    std::cout << "Estimator Thread Initialized" << std::endl;

    int visualization_iter = 0;
    while(true) {

        if(visualization_iter > 100) {
            // Print New State:
            auto state = estimator_interface.get_state();
            std::cout << "Estimator State: " << std::endl;
            std::cout << "Body Position: " << state.body_position.transpose() << std::endl;
            std::cout << "Body Rotation: " << state.body_rotation.w() << " " << state.body_rotation.vec().transpose() << std::endl;
            std::cout << "Joint Position: " << state.joint_position.transpose() << std::endl;
            std::cout << "Linear Body Velocity: " << state.linear_body_velocity.transpose() << std::endl;
            std::cout << "Angular Body Velocity: " << state.angular_body_velocity.transpose() << std::endl;
            std::cout << "Joint Velocity: " << state.joint_velocity.transpose() << std::endl;
            std::cout << "Linear Body Acceleration: " << state.linear_body_acceleration.transpose() << std::endl;
            std::cout << "Contact Mask: " << state.contact_mask.transpose() << std::endl;

            unitree::containers::IMUState imu_state = unitree_driver->get_imu_state();
            common::Vector4<float> unitree_quaternion = Eigen::Map<common::Vector4<float>>(imu_state.quaternion.data());
            std::cout << "Unitree Quaternion: " << unitree_quaternion.transpose() << std::endl;
            
            Eigen::Vector<double, 19> qpos;
            Eigen::Vector<double, 3> body_position = state.body_position.cast<double>();
            Eigen::Vector<double, 4> body_rotation = state.body_rotation.cast<double>().coeffs();
            Eigen::Vector<double, 12> joint_position = state.joint_position.cast<double>();
            qpos << body_position, body_rotation, joint_position;

            mj_data->qpos = qpos.data();
            mj_forward(mj_model, mj_data);

            mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            // swap OpenGL buffers (blocking call due to v-sync)
            glfwSwapBuffers(window);

            // process pending GUI events, call GLFW callbacks
            glfwPollEvents();

            visualization_iter = 0;
        }
        visualization_iter++;
    }

    // Clean up:
    mj_deleteModel(mj_model);
    mj_deleteData(mj_data);

    return 0;
};
