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
#include "tensorflow/core/tfrt_fallback/util/tensor_util.h"

#include "tensorflow/core/tfrt_fallback/util/type_util.h"
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

using tfrt::DType;
using tfrt::Expected;
using tfrt::HostBuffer;
using tfrt::HostContext;
using tfrt::RCReference;
using tfrt::StringHostTensor;
using tfrt::TensorShape;

// Moves one ref on HostBuffer to tensorflow::Tensor.
Expected<tensorflow::Tensor> MoveHostBufferToTFTensor(
    RCReference<HostBuffer> host_buffer, DType dtype,
    const tfrt::TensorShape& shape) {
  llvm::SmallVector<ssize_t, 4> dims;
  shape.GetDimensions(&dims);

  auto deallocator = [](void* data, size_t len, void* arg) {
    auto* host_buffer = reinterpret_cast<HostBuffer*>(arg);
    host_buffer->DropRef();
  };
  // Transfer one HostBuffer ref to TFTensor.
  OwnedTFTensor tf_tensor{
      TF_NewTensor(static_cast<TF_DataType>(GetTFDataType(dtype)), dims.data(),
                   dims.size(), host_buffer->data(), host_buffer->size(),
                   deallocator, host_buffer.release())};
  Tensor tensor;
  Status status = tensorflow::TF_TensorToTensor(tf_tensor.get(), &tensor);
  if (!status.ok())
    return tfrt::MakeStringError(
        tfrt::StrCat("error converting TF_Tensor to tensorflow::Tensor:",
                     status.error_message()));
  return std::move(tensor);
}

tensorflow::Tensor CopySHTToTFTensor(const StringHostTensor& sht,
                                     HostContext* host) {
  llvm::SmallVector<ssize_t, 4> dims;
  sht.shape().GetDimensions(&dims);

  tensorflow::Tensor tensor(tensorflow::DT_STRING,
                            tensorflow::TensorShape(llvm::SmallVector<int64, 4>(
                                dims.begin(), dims.end())));

  auto len = tensor.NumElements();
  auto from = sht.strings();
  auto to = tensor.flat<tensorflow::tstring>();

  // TODO(tfrt-devs): Consider a more efficient way to pass string
  // tensors between TFRT and TF.
  for (int i = 0; i < len; ++i) {
    to(i).assign(from[i].data(), from[i].size());
  }
  return tensor;
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace {
struct TFManagedBufferDeleter {
  void operator()(TF_ManagedBuffer* p) const { p->Unref(); }
};
using OwnedTFManagedBuffer =
    std::unique_ptr<TF_ManagedBuffer, TFManagedBufferDeleter>;
}  // namespace

// Moves one ref on GpuBuffer to tensorflow::Tensor.
Expected<tensorflow::Tensor> MoveGpuBufferToTFTensor(
    RCReference<tfrt::gpu::GpuBuffer> gpu_buffer, DType dtype,
    TensorShape shape) {
  auto deallocator = [](void* data, size_t len, void* arg) {
    auto* gpu_buffer = reinterpret_cast<tfrt::gpu::GpuBuffer*>(arg);
    gpu_buffer->DropRef();
  };

  // `owns_memory` is used by tensorflow::Tensor::RefCountIsOne.
  // One ref on `gpu_buffer` is transferred here to TF_ManagedBuffer.
  OwnedTFManagedBuffer tf_managed_buffer{
      new TF_ManagedBuffer(gpu_buffer->pointer().raw(), gpu_buffer->size(),
                           deallocator, gpu_buffer.release(),
                           /*owns_memory=*/false)};
  tensorflow::Tensor tensor(GetTFDataType(dtype), GetTFShape(shape),
                            tf_managed_buffer.get());
  return std::move(tensor);
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tfd
}  // namespace tensorflow
