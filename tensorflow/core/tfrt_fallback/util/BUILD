load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
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

cc_library(
    name = "attr_util",
    srcs = [
        "attr_util.cc",
    ],
    hdrs = [
        "attr_type.def",
        "attr_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:support",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite_no_runtime",
        ],
        "//conditions:default": [
            "//tensorflow/core/framework:tensor",
            "//tensorflow/core/framework:tensor_proto_cc",
            "//tensorflow/core/framework:types_proto_cc",
            "//tensorflow/core/platform:platform",
        ],
    }),
)

cc_library(
    name = "type_util",
    hdrs = [
        "dtype.def",
        "type_util.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite_no_runtime",
        ],
        "//conditions:default": [
            "//tensorflow/core/framework:types_proto_cc",
        ],
    }),
)

tf_cuda_library(
    name = "tensor_util",
    srcs = ["tensor_util.cc"],
    hdrs = [
        "tensor_util.h",
    ],
    cuda_deps = [
        "@tf_runtime//backends/gpu:gpu_config",
    ],
    deps = [
        ":type_util",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/c:tf_tensor_internal",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite_no_runtime",
        ],
        "//conditions:default": [
            "//tensorflow/core/framework:tensor",
            "//tensorflow/core/platform:status",
        ],
    }),
)
