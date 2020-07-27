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
//===- kernel_fallback_tensor.h ---------------------------------*- C++ -*-===//
//
// This file declares TF kernel fallback tensor.
//===----------------------------------------------------------------------===//
#ifndef TENSORFLOW_CORE_TFRT_FALLBACK_KERNEL_KERNEL_FALLBACK_TENSOR_H_
#define TENSORFLOW_CORE_TFRT_FALLBACK_KERNEL_KERNEL_FALLBACK_TENSOR_H_

#include <stdint.h>

#include "tfrt/tensor/tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tfrt {
class HostContext;
}

namespace tensorflow {

class KernelFallbackTensor final : public tfrt::Tensor {
 public:
  explicit KernelFallbackTensor(const ::tensorflow::Tensor& tensor);
  KernelFallbackTensor(const tfrt::TensorShape& shape, tfrt::DType dtype,
                       const ::tensorflow::Tensor& tensor);

  tfrt::AsyncValueRef<tfrt::HostTensor> ConvertToHostTensor(
      tfrt::HostContext* host, uint32_t allowed_formats) const override;

  void Print(tfrt::raw_ostream& os) const override;

  const ::tensorflow::Tensor* GetTensor() const { return &tensor_; }

  static bool classof(const tfrt::Tensor* t) {
    return t->subclass() == tfrt::Tensor::Subclass::TFKernelFallback;
  }

  static KernelFallbackTensor Create(const tensorflow::Tensor& tensor);

 private:
  ::tensorflow::Tensor tensor_;
  bool is_valid_type_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_FALLBACK_KERNEL_KERNEL_FALLBACK_TENSOR_H_
