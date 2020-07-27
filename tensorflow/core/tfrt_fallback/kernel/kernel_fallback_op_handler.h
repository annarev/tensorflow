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
//===- kernel_fallback_op_handler.h -----------------------------*- C++ -*-===//
//
// This file declares KernelFallbackOpHandler, responsible for running TFRT ops
// on Tensorflow.
//
//===----------------------------------------------------------------------===//

#ifndef TENSORFLOW_CORE_TFRT_FALLBACK_KERNEL_KERNEL_FALLBACK_OP_HANDLER_H_
#define TENSORFLOW_CORE_TFRT_FALLBACK_KERNEL_KERNEL_FALLBACK_OP_HANDLER_H_

#include <assert.h>
#include <stdlib.h>

#include <memory>

#include "llvm/Support/Error.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tfrt {
class HostTensor;
}
namespace tfrt {
class Tensor;
}

namespace tensorflow {
namespace tfd {

class KernelFallbackOpHandler : public tfrt::OpHandler {
 public:
  static llvm::Expected<std::unique_ptr<KernelFallbackOpHandler>> Create(
      tfrt::CoreRuntime* runtime, OpHandler* fallback);

  ~KernelFallbackOpHandler() override;

  llvm::Expected<tfrt::CoreRuntimeOp> MakeOp(
      tfrt::string_view op_name) override;

  tfrt::AsyncValueRef<tfrt::HostTensor> CopyDeviceTensorToHost(
      const tfrt::ExecutionContext& exec_ctx,
      const tfrt::Tensor& tensor) override;

  tfrt::AsyncValueRef<tfrt::Tensor> CopyHostTensorToDevice(
      const tfrt::DenseHostTensor& tensor) override {
    assert(false &&
           "KernelFallbackOpHandler::CopyHostTensorToDevice should not be "
           "called");
    abort();
  }

 private:
  explicit KernelFallbackOpHandler(tfrt::CoreRuntime* runtime,
                                   tfrt::RCReference<tfrt::Device> device);

  llvm::Error Initialize();
  tfrt::RCReference<tfrt::Device> device_;
};

}  // namespace tfd
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_TFRT_FALLBACK_KERNEL_KERNEL_FALLBACK_OP_HANDLER_H_
