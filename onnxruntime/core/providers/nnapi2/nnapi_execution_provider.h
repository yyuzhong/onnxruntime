// Copyright 2019 JD.com Inc. JD AI

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/onnx_protobuf.h"
#include "nnapi_lib/NeuralNetworksTypes.h"

namespace onnxruntime {
class Nnapi2ExecutionProvider : public IExecutionProvider {
 public:
  NnapiExecutionProvider();
  virtual ~NnapiExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;
  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  std::unordered_map<std::string, std::unique_ptr<ANeuralNetworksCompilation>> compilations_;
};
}  // namespace onnxruntime
