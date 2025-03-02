load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "array_safety",
    srcs = ["array_safety.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "utilities",
    srcs = ["utilities.cc"],
    hdrs = ["utilities.h"],
    deps = [
        ":array_safety",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings:strings",
        "@mujoco-bazel//:mujoco",
    ],
    visibility = ["//visibility:public"],
)
