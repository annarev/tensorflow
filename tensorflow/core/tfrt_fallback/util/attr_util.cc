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
#include "tensorflow/core/tfrt_fallback/util/attr_util.h"

#include <cstdlib>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace tfd {

DataType ParseTFDataType(absl::string_view dtype) {
  if (dtype == "DT_INT8") {
    return DataType::DT_INT8;
  } else if (dtype == "DT_INT32") {
    return DataType::DT_INT32;
  } else if (dtype == "DT_INT64") {
    return DataType::DT_INT64;
  } else if (dtype == "DT_HALF") {
    return DataType::DT_HALF;
  } else if (dtype == "DT_FLOAT") {
    return DataType::DT_FLOAT;
  } else if (dtype == "DT_DOUBLE") {
    return DataType::DT_DOUBLE;
  } else {
    assert(false && "Unsupported dtype");
    abort();
  }
}

DataType ConvertToTFDataType(tfrt::OpAttrType op_attr_type) {
  switch (op_attr_type) {
#define OP_ATTR_TYPE(TFRT_ENUM, DT_ENUM) \
  case tfrt::OpAttrType::TFRT_ENUM:      \
    return DataType::DT_ENUM;
#include "tensorflow/core/tfrt_fallback/util/attr_type.def"  // NOLINT
    default:
      llvm_unreachable("unsupported dtype in TFRT delegate kernel.");
  }
}

tfrt::OpAttrType ConvertFromTFDataType(DataType data_type) {
  switch (data_type) {
#define OP_ATTR_TYPE(TFRT_ENUM, DT_ENUM) \
  case DataType::DT_ENUM:                \
    return tfrt::OpAttrType::TFRT_ENUM;
#include "tensorflow/core/tfrt_fallback/util/attr_type.def"  // NOLINT
    default:
      llvm_unreachable("unsupported dtype in TFRT delegate kernel.");
  }
}

// TODO(chky): Unify BEFDataType with OpAttrType.
DataType ConvertBEFAttrTypeToTFDataType(tfrt::BEFDataType attr_type) {
  switch (attr_type) {
    case tfrt::BEFDataType::kI32:
      return DataType::DT_INT32;
    case tfrt::BEFDataType::kI64:
      return DataType::DT_INT64;
    case tfrt::BEFDataType::kF16:
      return DataType::DT_HALF;
    case tfrt::BEFDataType::kF32:
      return DataType::DT_FLOAT;
    case tfrt::BEFDataType::kF64:
      return DataType::DT_DOUBLE;
    case tfrt::BEFDataType::kString:
      return DataType::DT_STRING;
    default:
      llvm_unreachable("unsupported dtype in TFRT delegate kernel.");
  }
}

unsigned char ParseBoolAttrValue(absl::string_view attr_value) {
  if (attr_value == "false") {
    return 0;
  } else if (attr_value == "true") {
    return 1;
  } else {
    assert(false && "Bool attribute value invalid");
    abort();
  }
}

int ParseIntAttrValue(absl::string_view attr_value) {
  int value;
  bool success = absl::SimpleAtoi(attr_value, &value);
  assert(success && "SimpleAtoi integer parsing failed");
  (void)success;
  return value;
}

bool ParseTensorAttrValue(absl::string_view attr_value,
                          tensorflow::Tensor* tensor) {
  // if (std::is_base_of<tensorflow::protobuf::Message,
  //                     tensorflow::TensorProto>()) {
  //   tensorflow::TensorProto tensor_proto;
  //   // We use reinterpret_cast here to make sure ParseFromStringPiece call
  //   // below compiles if TensorProto is not a subclass of Message.
  //   // At run time, we should never get to this point if TensorProto
  //   // is not a subclass of message due to if-condition above.
  //   auto* message = reinterpret_cast<protobuf::Message*>(&tensor_proto);
  //   return tensorflow::protobuf::TextFormat::ParseFromStringPiece(attr_value,
  //                                                                 message) &&
  //          tensor->FromProto(tensor_proto);
  // } else {
  //   // TextFormat does not work with portable proto implementations.
  //   assert(false && "Tensor attributes are not supported on mobile.");
  //   return false;
  // }
  return false;
}

std::vector<int64_t> ParseTensorShapeAttrValue(absl::string_view attr_value) {
  assert(attr_value.size() >= 2 &&
         "tensor shape attribute must be a list of the form [1,2...]");
  assert(attr_value[0] == '[' && "tensor shape attribute must start with a [");
  assert(attr_value[attr_value.size() - 1] == ']' &&
         "tensor shape attribute must end with a ]");
  absl::string_view attr_value_trunc =
      attr_value.substr(1, attr_value.size() - 2);
  // `container` is an absl::strings_internal::Splitter, which is a
  // lazy-splitting iterable. So we cannot get its size to reserve `dims`.
  auto container = absl::StrSplit(attr_value_trunc, ',');
  std::vector<int64_t> dims;
  for (auto it = container.begin(); it != container.end(); ++it) {
    dims.push_back(ParseIntAttrValue(*it));
  }
  return dims;
}

}  // namespace tfd
}  // namespace tensorflow
