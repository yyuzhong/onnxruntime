// Copyright 2019 JD.com Inc. JD AI

#include "core/providers/nnapi/nnapi_provider_factory.h"
#include "nnapi_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct Nnapi2ProviderFactory : IExecutionProviderFactory {
  Nnapi2ProviderFactory() {}
  ~Nnapi2ProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> NnapiProviderFactory::CreateProvider() {
  return onnxruntime::make_unique<Nnapi2ExecutionProvider>();
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi2() {
  return std::make_shared<onnxruntime::Nnapi2ProviderFactory>();
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nnapi2, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Nnapi2());
  return nullptr;
}


