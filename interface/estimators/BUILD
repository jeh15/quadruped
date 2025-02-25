cc_library(
    name = "estimator",
    srcs = ["estimator.h"],
    deps = [
        "//interface/estimators/utilities:utilities",
        "@mujoco-bazel//:mujoco",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kalman",
    srcs = ["kalman.cc"],
    hdrs = ["kalman.h"],
    deps = [
        ":estimator",
        "//interface/estimators/utilities:utilities",
        "//interface/estimators/utilities:array_safety",
        "@mujoco-bazel//:mujoco",
    ],
    visibility = ["//visibility:public"],
)
