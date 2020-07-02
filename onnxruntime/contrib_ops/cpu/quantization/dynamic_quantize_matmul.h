// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class DynamicQuantizeMatMul final : public OpKernel {
 public:
  DynamicQuantizeMatMul(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed) override {
    is_packed = false;
    if (input_idx == 1) {
      m_b_shape = tensor.Shape();

      auto alloc = Info().GetAllocator(0, OrtMemTypeDefault);
      auto* packed_b_data = alloc->Alloc(m_b_shape.Size() * sizeof(T));
      m_b_data = BufferUniquePtr(packed_b_data, BufferDeleter(alloc));

      is_packed = true;
    }
    return Status::OK();
  }

 private:
  TensorShape m_b_shape;
  BufferUniquePtr m_b_data;
};
}  // namespace contrib
}  // namespace onnxruntime
