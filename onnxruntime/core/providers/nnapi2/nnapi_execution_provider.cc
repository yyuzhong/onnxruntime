// Copyright 2019 JD.com Inc. JD AI

#include "nnapi_execution_provider.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"

namespace onnxruntime {

constexpr const char* NNAPI = "Nnapi";

Nnapi2ExecutionProvider::NnapiExecutionProvider()
    : IExecutionProvider{onnxruntime::kNnapiExecutionProvider} {
  DeviceAllocatorRegistrationInfo device_info{OrtMemTypeDefault,
                                              [](int) { return onnxruntime::make_unique<CPUAllocator>(
                                                            onnxruntime::make_unique<OrtMemoryInfo>(NNAPI,
                                                                                               OrtAllocatorType::OrtDeviceAllocator)); },
                                              std::numeric_limits<size_t>::max()};
  InsertAllocator(CreateAllocator(device_info));

  DeviceAllocatorRegistrationInfo cpu_memory_info({OrtMemTypeCPUOutput,
                                                      [](int) { return onnxruntime::make_unique<CPUAllocator>(onnxruntime::make_unique<OrtMemoryInfo>(NNAPI, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput)); },
                                                      std::numeric_limits<size_t>::max()});

  InsertAllocator(CreateAllocator(cpu_memory_info));
}

NnapiExecutionProvider::~Nnapi2ExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
Nnapi2ExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                      const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // Find inputs, initializers and outputs for each supported subgraph
  std::vector<std::unique_ptr<ComputeCapability>> result;
  return result;
}

common::Status Nnapi2ExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& /* fused_nodes */,
                                               std::vector<NodeComputeInfo>& /* node_compute_funcs */) {
  return Status::OK();
}
}  // namespace onnxruntime
