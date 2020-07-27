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
#include "tensorflow/core/tfrt_fallback/kernel/kernel_fallback_tensor.h"

#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <sys/types.h>

#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt_fallback/util/type_util.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace tensorflow {
using tfrt::DenseHostTensor;
using tfrt::HostContext;

/*static*/
KernelFallbackTensor KernelFallbackTensor::Create(
    const tensorflow::Tensor& tensor) {
  tfrt::DType dtype = tfd::GetTFRTDtype(tensor.dtype());
  llvm::SmallVector<ssize_t, 4> shape_array(tensor.shape().dims());
  for (auto dim : tensor.shape()) {
    shape_array.push_back(static_cast<ssize_t>(dim.size));
  }
  tfrt::TensorShape shape(shape_array);
  return KernelFallbackTensor(shape, dtype, tensor);
}

KernelFallbackTensor::KernelFallbackTensor(const tfrt::TensorShape& shape,
                                           tfrt::DType dtype,
                                           const ::tensorflow::Tensor& tensor)
    : tfrt::Tensor(
          tfrt::Tensor::Subclass::TFKernelFallback,
          tfrt::TensorMetadata(
              dtype.IsValid() ? dtype : tfrt::GetDType<int8_t>(), shape)),
      tensor_(std::move(tensor)) {
  assert(dtype.IsValid() && "Invalid dtype");
}

tfrt::AsyncValueRef<tfrt::HostTensor> KernelFallbackTensor::ConvertToHostTensor(
    tfrt::HostContext* host, uint32_t allowed_formats) const {
  assert(metadata().dtype.kind() != tfrt::DType::Unsupported &&
         "Unsupported dtype");

  void* data = tensor_.data();
  size_t size = tensor_.AllocatedBytes();
  tfrt::RCReference<tfrt::HostBuffer> host_buffer =
      tfrt::HostBuffer::CreateFromExternal(
          data, size, [tensor = std::move(tensor_)](void*, size_t) {});
  // Assume HostBuffer::CreateFromExternal never fails.
  return host->MakeAvailableAsyncValueRef<DenseHostTensor>(
      metadata(), std::move(host_buffer));
}

void KernelFallbackTensor::Print(tfrt::raw_ostream& os) const {
  os << tensor_.DebugString();
}
}  // namespace tensorflow
