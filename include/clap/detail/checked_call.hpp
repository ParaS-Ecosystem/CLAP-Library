// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// This file is part of the CLAP library, a component of the ParaS Ecosystem.
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License (LGPL) version 3
// as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library. If not, see <https://www.gnu.org/licenses/>.
// -----------------------------------------------------------------------------

#pragma once

#include <utility>

namespace clap::detail {

template <typename Function, typename Check> class CheckedCall {
public:
  CheckedCall(Function &function, Check check, const char *operation)
      : function_(function), check_(check), operation_(operation) {}

  template <typename... Args> void operator()(Args &&...args) const {
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

} // namespace clap::detail
