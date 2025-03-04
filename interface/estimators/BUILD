load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "imu_estimator",
    srcs = ["imu_estimator.h"],
    deps = [
        "//interface/unitree_go2:aliases",
        "@eigen//:eigen",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status:status",
    ],
    visibility = ["//visibility:public"],
)
