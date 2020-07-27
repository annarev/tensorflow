/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//===- conv_ops_3d.cc ---------------------------------------------------===//
//
// Register TensorFlow's Conv3DOp so it can be called directly from TFRT.
//
//===----------------------------------------------------------------------===//

#include "tensorflow/core/kernels/conv_ops_3d.h"

#include "tensorflow/core/tfrt_fallback/kernel/tfrt_op_kernel.h"

namespace tensorflow {

// TODO(annarev): reuse op registration from nn_ops.cc. This
// requires supporting attribute outputs and compound attribute types.
REGISTER_KERNEL_FALLBACK_OP("Conv3D").Output("out: int32");

REGISTER_KERNEL_FALLBACK_KERNEL(
    "Conv3D", Conv3DOp<CPUDevice, int32, TFRTOpKernel, TFRTOpKernelConstruction,
                       TFRTOpKernelContext>);

}  // namespace tensorflow
