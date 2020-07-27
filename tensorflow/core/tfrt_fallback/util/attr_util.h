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
#ifndef TENSORFLOW_CORE_TFRT_FALLBACK_UTIL_ATTR_UTIL_H_
#define TENSORFLOW_CORE_TFRT_FALLBACK_UTIL_ATTR_UTIL_H_

#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tfrt/core_runtime/op_attr_type.h"  // from @tf_runtime
#include "tfrt/support/bef_encoding.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

inline absl::string_view ToAbslStringView(tfrt::string_view sv) {
  return absl::string_view(sv.data(), sv.size());
}

DataType ParseTFDataType(absl::string_view dtype);

DataType ConvertToTFDataType(tfrt::OpAttrType op_attr_type);

tfrt::OpAttrType ConvertFromTFDataType(DataType data_type);

DataType ConvertBEFAttrTypeToTFDataType(tfrt::BEFDataType attr_type);

bool ParseTensorAttrValue(absl::string_view attr_value,
                          tensorflow::Tensor* tensor);

std::vector<int64_t> ParseTensorShapeAttrValue(absl::string_view attr_value);

unsigned char ParseBoolAttrValue(absl::string_view attr_value);

int ParseIntAttrValue(absl::string_view attr_value);

inline std::vector<absl::string_view> AttrValueSplit(absl::string_view str) {
  return absl::StrSplit(str, absl::MaxSplits('$', 1));
}

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_FALLBACK_UTIL_ATTR_UTIL_H_
