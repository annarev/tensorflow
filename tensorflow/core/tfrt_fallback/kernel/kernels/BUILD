package_group(
    name = "internal",
    packages = [
        "//tensorflow/core/tfrt_fallback/...",
    ],
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "aggregate_ops",
    srcs = ["aggregate_ops.cc"],
    deps = [
        "//tensorflow/core/kernels:aggregate_ops_headers",
        "//tensorflow/core/tfrt_fallback/kernel:tfrt_op_kernel",
    ],
    alwayslink = 1,
)

cc_library(
    name = "conv_ops_3d",
    srcs = ["conv_ops_3d.cc"],
    deps = [
        "//tensorflow/core/kernels:conv_ops_3d_headers",
        "//tensorflow/core/tfrt_fallback/kernel:tfrt_op_kernel",
    ],
    alwayslink = 1,
)
