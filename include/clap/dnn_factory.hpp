#pragma once

#include "idnn_backend.hpp"
#include <memory>

namespace clap {

class DnnFactory {
public:
    static std::unique_ptr<IDnnBackend> create(DnnBackendType requested = DnnBackendType::CPU);
};

} // namespace clap
