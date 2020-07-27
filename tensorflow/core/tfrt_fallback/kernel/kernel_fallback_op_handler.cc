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
#include "tensorflow/core/tfrt_fallback/kernel/kernel_fallback_op_handler.h"

#include <assert.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <utility>

#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/tfrt_fallback/kernel/kernel_fallback_execute.h"
#include "tensorflow/core/tfrt_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/tfrt_fallback/util/tensor_util.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/dispatch_utils.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler_factory.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_invocation.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_frame.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace {
using tfrt::AsyncValue;
using tfrt::AsyncValueRef;
using tfrt::Chain;
using tfrt::CoreRuntime;
using tfrt::CoreRuntimeOp;
using tfrt::DenseHostTensor;
using tfrt::ExecutionContext;
using tfrt::Expected;
using tfrt::OpAttrsRef;
using tfrt::OpHandler;
using tfrt::OpHandlerRegistration;
using tfrt::OpInvocation;
using tfrt::OpMetadataFn;
using tfrt::raw_ostream;
using tfrt::RCReference;
using tfrt::SmallVector;
using tfrt::string_view;
using tfrt::TensorMetadata;
}  // namespace

namespace tensorflow {
namespace tfd {

namespace {

using TfDispatchFn =
    bool (*)(const ExecutionContext& exec_ctx, string_view op_name,
             tfrt::ArrayRef<AsyncValue*> arguments,
             tfrt::MutableArrayRef<RCReference<AsyncValue>> results,
             const OpAttrsRef& attrs, KernelFallbackOutputType output_type);

struct TfOpEntry {
  string_view op_name;
  OpMetadataFn metadata_fn = nullptr;
  // All ops use the same dispatch function.
  TfDispatchFn dispatch_fn = &KernelFallbackExecute;
};

struct TfOpHandlerTraits {
  // AsyncValue that contains tensorflow::Tensor
  using InputTensorTy = tfrt::AsyncValue;
  using OpEntryTy = TfOpEntry;
  using OpHandlerInfoTy = KernelFallbackOpHandler*;

  // Converts tfrt::Tensor to tensorflow::Tensor.
  static bool MaybeConvertTensor(const TfOpEntry& op_entry,
                                 KernelFallbackOpHandler* tf_op_handler,
                                 const tfrt::Tensor& arg_tensor,
                                 const ExecutionContext& exec_ctx,
                                 RCReference<AsyncValue>* converted) {
    // Convert input to tensorflow::Tensor.
    auto* host = exec_ctx.host();
    if (llvm::isa<KernelFallbackTensor>(&arg_tensor)) {
      const tensorflow::Tensor* tftensor =
          llvm::dyn_cast<const KernelFallbackTensor>(&arg_tensor)->GetTensor();
      *converted = host->MakeAvailableAsyncValueRef<tensorflow::Tensor>(
          std::move(*tftensor));
      return true;
    }
    if (auto* dht = llvm::dyn_cast<DenseHostTensor>(&arg_tensor)) {
      // Copy and transfer one HostBuffer reference to tensorflow::Tensor.
      Expected<tensorflow::Tensor> tensor = MoveHostBufferToTFTensor(
          dht->buffer().CopyRef(), dht->dtype(), dht->shape());
      if (tensor) {
        *converted = host->MakeAvailableAsyncValueRef<tensorflow::Tensor>(
            std::move(tensor.get()));
      } else {
        *converted = EmitErrorAsync(exec_ctx, tensor.takeError());
      }
      return true;
    }
    // TODO(annarev): add other conversions, for e.g. from
    // RuntimeFallbackTensor.

    *converted = host->MakeErrorAsyncValueRef(
        "argument tensor conversion to tensorflow::Tensor is not supported");
    return true;
  }

  static void Dispatch(const TfOpEntry& op_entry,
                       KernelFallbackOpHandler* tf_op_handler,
                       llvm::ArrayRef<tfrt::AsyncValue*> inputs,
                       const OpAttrsRef& attrs,
                       llvm::ArrayRef<TensorMetadata> result_mds,
                       llvm::MutableArrayRef<RCReference<AsyncValue>> results,
                       AsyncValueRef<Chain>* chain,
                       const ExecutionContext& exec_ctx) {
    tfrt::HostContext* host = exec_ctx.host();
    // TODO(annarev): Add a call to runtime fallback if kernel fallback
    // does not support the kernel.
    // Dispatch kernel call
    // TODO(annarev): pass chain as well.
    for (auto& result : results) {
      result = host->MakeUnconstructedAsyncValueRef<KernelFallbackTensor>();
    }
    bool status =
        op_entry.dispatch_fn(exec_ctx, op_entry.op_name, inputs, results, attrs,
                             KernelFallbackOutputType::KERNEL_FALLBACK_TENSOR);
    // Handle possible error.
    if (!status) {
      std::string error_message = "Kernel fallback dispatch failed.";
      // Set all results to error.
      for (auto& result : results) {
        result->SetError(tfrt::EmitError(exec_ctx, error_message));
      }
      *chain = EmitErrorAsync(exec_ctx, error_message);
    } else {
      *chain = host->GetReadyChain();
    }
  }
};
}  // namespace

Expected<CoreRuntimeOp> KernelFallbackOpHandler::MakeOp(string_view op_name) {
  // NOTE(fishx): Copying string here will cost extra overhead in graph
  // execution. Because in current implementation, we needs to prepare the op
  // before each executions.
  // TODO(fishx): Avoid this heap allocation by getting op registration
  // information from current TF.
  return CoreRuntimeOp(
      [op_name = op_name.str(), this](const OpInvocation& invocation) {
        // If the op does not have outputs, then it is expected to output an
        // out chain.
        // TODO(b/152886204): Have a better way to support stateful op.
        bool update_chain = invocation.results.empty();
        TfOpEntry fallback_op_entry;
        fallback_op_entry.op_name = op_name;
        return tfrt::ExecuteOnOpHandler<TfOpHandlerTraits>(
            update_chain, invocation, this->device_.CopyRef(),
            fallback_op_entry, this);
      },
      /*is_fallback=*/true);
}

llvm::Expected<std::unique_ptr<KernelFallbackOpHandler>>
KernelFallbackOpHandler::Create(CoreRuntime* runtime, OpHandler* fallback) {
  // TODO(b/158775215): Support output GPU tensor as well since this op handler
  // is used for both CPU and GPU.
  std::unique_ptr<KernelFallbackOpHandler> op_handler(
      new KernelFallbackOpHandler(
          runtime, runtime->GetHostContext()->GetHostDeviceRef()));
  if (auto error = op_handler->Initialize()) {
    return std::move(error);
  }
  return op_handler;
}

KernelFallbackOpHandler::KernelFallbackOpHandler(
    CoreRuntime* runtime, tfrt::RCReference<tfrt::Device> device)
    : OpHandler("tfkernels", runtime, nullptr), device_(std::move(device)) {}

KernelFallbackOpHandler::~KernelFallbackOpHandler() {}

llvm::Error KernelFallbackOpHandler::Initialize() {
  return llvm::Error::success();
}

tfrt::AsyncValueRef<tfrt::HostTensor>
KernelFallbackOpHandler::CopyDeviceTensorToHost(
    const tfrt::ExecutionContext& exec_ctx, const tfrt::Tensor& tensor) {
  assert(llvm::isa<KernelFallbackTensor>(tensor) &&
         "KernelFallbackOpHandler::CopyDeviceTensorToHost expects a "
         "KernelFallbackTensor");
  auto& fallback_tensor = llvm::cast<KernelFallbackTensor>(tensor);
  auto host = GetRuntime()->GetHostContext();
  uint32_t allowed_formats =
      1 << static_cast<uint32_t>(tfrt::Tensor::Subclass::DenseHost);
  return fallback_tensor.ConvertToHostTensor(host, allowed_formats);
}

static OpHandlerRegistration op_handler_registration(
    "tfkernel", KernelFallbackOpHandler::Create);

}  // namespace tfd
}  // namespace tensorflow
