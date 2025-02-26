cc_library(
    name = "osc_interface",
    srcs = ["osc_interface.h"],
    deps = [
        "@unitree-bazel//:unitree_go2_api",
        "@operational-space-controller//:operational_space_controller",
        "@mujoco-bazel//:mujoco",
        "@eigen//:eigen",
        "@osqp-cpp//:osqp++",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status:status",
    ],
    visibility = ["//visibility:public"],
)