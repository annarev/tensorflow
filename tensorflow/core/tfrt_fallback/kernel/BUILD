# Fallback from TFRT to TF kernels and related utilities.
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tfrt_op_kernel",
    srcs = ["tfrt_op_kernel.cc"],
    hdrs = ["tfrt_op_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":attr_util",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "//tensorflow/core/tfrt_fallback/util:attr_util",
        "@tf_runtime//backends/common:eigencompat",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite_no_runtime",
        ],
        "//conditions:default": [
            "//tensorflow/core/framework:tensor",
            "//tensorflow/core/framework:tensor_shape",
            "//tensorflow/core/platform:errors",
            "//tensorflow/core/platform:status",
            "//tensorflow/core/platform:str_util",
            "//tensorflow/core/platform:stringpiece",
        ],
    }),
)

tf_cc_test(
    name = "tfrt_op_kernel_test",
    srcs = ["tfrt_op_kernel_test.cc"],
    deps = [
        ":tfrt_op_kernel",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/lib/core:error_codes_proto_cc",
        "//tensorflow/core/util:padding",
        "@tf_runtime//:core_runtime",
    ],
)

cc_library(
    name = "attr_util",
    srcs = ["attr_util.cc"],
    hdrs = ["attr_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:Support",
        "//tensorflow/core/tfrt_fallback/util:attr_util",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite_no_runtime",
        ],
        "//conditions:default": [
            "//tensorflow/core/framework:types_proto_cc",
            "//tensorflow/core/platform:errors",
            "//tensorflow/core/platform:status",
            "//tensorflow/core/platform:str_util",
            "//tensorflow/core/platform:stringpiece",
            "//tensorflow/core/util:padding",
        ],
    }),
)

tf_cc_test(
    name = "attr_util_test",
    srcs = ["attr_util_test.cc"],
    deps = [
        ":attr_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:types",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "kernel_fallback_kernels_alwayslink",
    srcs = [
        "kernel_fallback_kernels.cc",
        "kernel_fallback_static_registration.cc",
    ],
    includes = [
        "third_party/tf_runtime/include",
    ],
    deps = [
        ":attr_util",
        ":kernel_fallback_execute",
        "@llvm-project//llvm:Support",
        "//tensorflow/core/tfrt_fallback/util:attr_util",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite_no_runtime",
        ],
        "//conditions:default": [
            "//tensorflow/core/framework:tensor",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "kernel_fallback_tensor",
    srcs = ["kernel_fallback_tensor.cc"],
    hdrs = ["kernel_fallback_tensor.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/c:tf_tensor_internal",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:tensor_alwayslink",
        "//tensorflow/core/tfrt_fallback/util:type_util",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite_no_runtime",
        ],
        "//conditions:default": [
            "//tensorflow/core/framework:tensor",
            "//tensorflow/core/framework:tensor_shape",
            "//tensorflow/core/framework:types_proto_cc",
            "//tensorflow/core/platform:status",
        ],
    }),
)

cc_library(
    name = "kernel_fallback_op_handler",
    srcs = ["kernel_fallback_op_handler.cc"],
    hdrs = ["kernel_fallback_op_handler.h"],
    deps = [
        ":kernel_fallback_execute",
        ":kernel_fallback_tensor",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/tfrt_fallback/util:tensor_util",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor_alwayslink",
    ],
)

cc_library(
    name = "kernel_fallback_execute",
    srcs = ["kernel_fallback_execute.cc"],
    hdrs = ["kernel_fallback_execute.h"],
    deps = [
        ":kernel_fallback_tensor",
        ":tfrt_op_kernel",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:tensor_alwayslink",
        "@tf_runtime//backends/common:eigencompat",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite_no_runtime",
        ],
        "//conditions:default": [
            "//tensorflow/core/framework:tensor",
        ],
    }),
)
