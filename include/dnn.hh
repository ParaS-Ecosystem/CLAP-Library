#pragma once

#include "clap/dnn_factory.hpp"
#include "clap/dnn_types.hpp"
#include "clap/idnn_backend.hpp"

namespace clap {

inline std::unique_ptr<IDnnBackend> createDnnBackend()
{
#if defined(gpu)
    return DnnFactory::create(DnnBackendType::GPU);
#elif defined(cpu)
    return DnnFactory::create(DnnBackendType::CPU);
#else
    return DnnFactory::create(DnnBackendType::CPU);
#endif
}

} // namespace clap
