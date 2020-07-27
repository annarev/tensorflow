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
#ifndef TENSORFLOW_CORE_TFRT_FALLBACK_UTIL_TENSOR_UTIL_H_
#define TENSORFLOW_CORE_TFRT_FALLBACK_UTIL_TENSOR_UTIL_H_

#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "third_party/tf_runtime/backends/gpu/include/tfrt/gpu/memory/gpu_buffer.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {
namespace tfd {

struct TFTensorDeleter {
  void operator()(TF_Tensor* p) const { TF_DeleteTensor(p); }
};
using OwnedTFTensor = std::unique_ptr<TF_Tensor, TFTensorDeleter>;

// Moves one ref on HostBuffer to tensorflow::Tensor.
tfrt::Expected<tensorflow::Tensor> MoveHostBufferToTFTensor(
    tfrt::RCReference<tfrt::HostBuffer> host_buffer, tfrt::DType dtype,
    const tfrt::TensorShape& shape);

// Creates a tensorflow::Tensor based on StringHostTensor.
tensorflow::Tensor CopySHTToTFTensor(const tfrt::StringHostTensor& sht,
                                     tfrt::HostContext* host);

// Converts tfrt shape to tensorflow shape.
inline tensorflow::TensorShape GetTFShape(const tfrt::TensorShape& shape) {
  llvm::SmallVector<ssize_t, 4> dimensions;
  shape.GetDimensions(&dimensions);
  llvm::SmallVector<int64, 4> dims(dimensions.begin(), dimensions.end());
  return tensorflow::TensorShape(dims);
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Moves one ref on GpuBuffer to tensorflow::Tensor.
tfrt::Expected<tensorflow::Tensor> MoveGpuBufferToTFTensor(
    tfrt::RCReference<tfrt::gpu::GpuBuffer> gpu_buffer, tfrt::DType dtype,
    tfrt::TensorShape shape);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tfd
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_TFRT_FALLBACK_UTIL_TENSOR_UTIL_H_
