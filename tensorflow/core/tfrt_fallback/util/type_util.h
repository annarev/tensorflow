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
#ifndef TENSORFLOW_CORE_TFRT_FALLBACK_UTIL_TYPE_UTIL_H_
#define TENSORFLOW_CORE_TFRT_FALLBACK_UTIL_TYPE_UTIL_H_

#include "llvm/Support/ErrorHandling.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

// Map tfrt::Dtype to TF_DataType.
inline DataType GetTFDataType(tfrt::DType dtype) {
  switch (dtype.kind()) {
    case tfrt::DType::Invalid:
      llvm_unreachable("invalid dtype");
    case tfrt::DType::Unsupported:
      llvm_unreachable("unsupported dtype");
    case tfrt::DType::I1:
      llvm_unreachable("unsupported dtype");
#define DTYPE(TFRT_ENUM, DT_ENUM) \
  case tfrt::DType::TFRT_ENUM:    \
    return DataType::DT_ENUM;
#include "tensorflow/core/tfrt_fallback/util/dtype.def"  // NOLINT
  }
}

inline tfrt::DType GetTFRTDtype(DataType dtype) {
  switch (dtype) {
    default:
      return tfrt::DType(tfrt::DType::Unsupported);
    case DataType::DT_INVALID:
      return tfrt::DType();
#define DTYPE(TFRT_ENUM, DT_ENUM) \
  case DataType::DT_ENUM:         \
    return tfrt::DType(tfrt::DType::TFRT_ENUM);
#include "tensorflow/core/tfrt_fallback/util/dtype.def"  // NOLINT
  }
}

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_FALLBACK_UTIL_TYPE_UTIL_H_
