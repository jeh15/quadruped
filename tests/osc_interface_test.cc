#include <filesystem>

#include "interface/unitree_go2/osc_interface.h"

int main(int argc, char** argv) {
    // OSC Args:
    OperationalSpaceControllerArgs osc_args = {
        .xml_path = std::filesystem::path("dummy.path"),
        .control_rate = 1000,
        .osqp_settings = osqp::OsqpSettings(),
    };

    // MC Args:
    MotorControllerArgs mc_args = {
        .network_name = "dummy_network",
        .control_rate = 2000,
    };

    // Initialize Interface Driver:
    UnitreeGo2Interface unitree_driver(osc_args, mc_args);

    return 0;
}