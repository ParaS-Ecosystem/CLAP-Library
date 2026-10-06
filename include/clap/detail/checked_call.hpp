// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <utility>

namespace clap::detail {

template <typename Function, typename Check>
class CheckedCall {
public:
    CheckedCall(Function &function, Check check, const char *operation)
        : function_(function), check_(check), operation_(operation) {}

    template <typename... Args>
    void operator()(Args&&... args) const {
        check_(function_(std::forward<Args>(args)...), operation_);
    }

private:
    Function &function_;
    Check check_;
    const char *operation_;
};

template <typename Function, typename Check>
auto checked_call(Function &function, Check check, const char *operation) {
    return CheckedCall<Function, Check>(function, check, operation);
}

}  // namespace clap::detail
