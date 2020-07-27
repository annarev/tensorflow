load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/core/tfrt_fallback/...",
    ],
)

package(
    default_visibility = [":internal"],
    licenses = ["notice"],  # Apache 2.0
)

# SUPPORTED_FORWARDING_KERNELS are compatible with
# TFRTForwardingOpKernel{,Construction,Context}. These kernels can be directly
# called from TFRT, bypassing the existing TF runtime.
# @unused
SUPPORTED_FORWARDING_KERNELS = [
    "//tensorflow/core/tfrt_fallback/kernel/kernels:aggregate_ops",
    "//tensorflow/core/tfrt_fallback/kernel/kernels:conv_ops_3d",
]
