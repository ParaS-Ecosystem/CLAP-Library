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

#include "clap/dnn_types.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace clap {
namespace dnn_contract {

[[noreturn]] inline void unsupported(const std::string& backend, const std::string& what)
{
    throw DnnUnsupportedError("CLAP_DNN " + backend + ": " + what);
}

inline void requireBackend(const ExecutionContext& ctx,
                           ExecutionBackend expected,
                           const char* backend_name,
                           const char* operation)
{
    if (ctx.backend != expected) {
        throw std::runtime_error(std::string("CLAP_DNN ") + backend_name + " " + operation +
                                 ": ExecutionContext targets " + executionBackendName(ctx.backend) +
                                 " but this backend executes on " + executionBackendName(expected));
    }
}

inline bool traceEnabled()
{
    static const bool enabled = std::getenv("CLAP_DNN_VERBOSE") != nullptr;
    return enabled;
}

inline void trace(const char* backend, const char* operation, const char* vendor_call)
{
    if (traceEnabled())
        std::cout << "[CLAP_DNN][" << backend << "] " << operation << " -> " << vendor_call << "\n";
}

inline void requireSameShape(const TensorDesc& a, const TensorDesc& b, const char* where)
{
    if (a.dims != b.dims)
        throw std::runtime_error(std::string(where) + ": tensor shapes do not match");
}

inline void requireSameType(const TensorDesc& a, const TensorDesc& b, const char* where)
{
    if (a.type != b.type)
        throw std::runtime_error(std::string(where) + ": tensor datatypes do not match (" +
                                 dataTypeName(a.type) + " vs " + dataTypeName(b.type) + ")");
}

inline void squeezeUnitDims(const TensorDesc& d,
                            std::vector<std::int64_t>& dims,
                            std::vector<std::int64_t>& strides)
{
    const auto s = d.effectiveStrides();
    dims.clear();
    strides.clear();
    for (std::size_t i = 0; i < d.dims.size(); ++i) {
        if (d.dims[i] == 1)
            continue;
        dims.push_back(d.dims[i]);
        strides.push_back(s[i]);
    }
}

inline bool collapseRange(const TensorDesc& d,
                          std::size_t first,
                          std::size_t last,
                          std::int64_t& extent,
                          std::int64_t& stride)
{
    const auto s = d.effectiveStrides();
    extent = 1;
    stride = 1;
    bool have_inner = false;
    std::int64_t expected = 0;

    for (std::size_t i = last; i-- > first;) {
        if (d.dims[i] == 1)
            continue;
        if (!have_inner) {
            stride = s[i];
            expected = s[i] * d.dims[i];
            have_inner = true;
        } else {
            if (s[i] != expected)
                return false;
            expected = s[i] * d.dims[i];
        }
        extent *= d.dims[i];
    }
    if (!have_inner)
        stride = 1;
    return true;
}

struct AxisView {
    std::int64_t outer = 1;
    std::int64_t axis = 1;
    std::int64_t inner = 1;
    std::int64_t outer_stride = 1;
    std::int64_t axis_stride = 1;
    std::int64_t inner_stride = 1;
};

inline bool makeAxisView(const TensorDesc& d, int axis, AxisView& view)
{
    const auto s = d.effectiveStrides();
    const std::size_t a = static_cast<std::size_t>(axis);

    if (!collapseRange(d, 0, a, view.outer, view.outer_stride))
        return false;
    if (!collapseRange(d, a + 1, d.dims.size(), view.inner, view.inner_stride))
        return false;

    view.axis = d.dims[a];
    view.axis_stride = s[a];

    if (view.inner == 1)
        view.inner_stride = 1;
    if (view.outer == 1)
        view.outer_stride = view.axis * view.axis_stride;
    return true;
}

struct RowsView {
    std::int64_t rows = 1;
    std::int64_t inner = 1;
    std::int64_t row_stride = 1;
};

inline bool makeRowsView(const TensorDesc& d, RowsView& view)
{
    if (d.dims.empty())
        return false;
    const auto s = d.effectiveStrides();
    view.inner = d.dims.back();
    if (view.inner != 1 && s.back() != 1)
        return false;
    if (!collapseRange(d, 0, d.dims.size() - 1, view.rows, view.row_stride))
        return false;
    if (view.rows == 1)
        view.row_stride = view.inner;
    return view.rows == 1 || view.row_stride >= view.inner;
}

inline int normalizeLastAxis(const TensorDesc& d, int axis, const char* where)
{
    const int a = d.normalizeAxis(axis, where);
    if (a != static_cast<int>(d.dims.size()) - 1)
        throw DnnUnsupportedError(std::string(where) + ": only normalization over the last dimension is supported");
    return a;
}

struct AttentionShape {
    std::int64_t batch = 0;
    std::int64_t q_heads = 0;
    std::int64_t kv_heads = 0;
    std::int64_t q_len = 0;
    std::int64_t kv_len = 0;
    std::int64_t head_dim = 0;
    std::int64_t value_dim = 0;
    double scale = 1.0;
    bool has_mask = false;

    std::int64_t groupSize() const { return q_heads / kv_heads; }
};

inline AttentionShape validateAttention(const AttentionDesc& a,
                                        const TensorDesc& q,
                                        const TensorDesc& k,
                                        const TensorDesc& v,
                                        const TensorDesc* mask,
                                        const TensorDesc& o)
{
    const char* where = "scaledDotProductAttentionForward";

    if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4 || o.rank() != 4)
        throw std::runtime_error(std::string(where) + ": Q, K, V and O must be rank-4 [B, H, S, D] tensors");

    AttentionShape shape;
    shape.batch = q.dims[0];
    shape.q_heads = q.dims[1];
    shape.q_len = q.dims[2];
    shape.head_dim = q.dims[3];
    shape.kv_heads = k.dims[1];
    shape.kv_len = k.dims[2];
    shape.value_dim = v.dims[3];

    if (k.dims[0] != shape.batch || v.dims[0] != shape.batch || o.dims[0] != shape.batch)
        throw std::runtime_error(std::string(where) + ": batch dimensions do not match");
    if (k.dims[3] != shape.head_dim)
        throw std::runtime_error(std::string(where) + ": Q and K head dimensions do not match");
    if (v.dims[1] != shape.kv_heads || v.dims[2] != shape.kv_len)
        throw std::runtime_error(std::string(where) + ": K and V must share [B, Hkv, Skv]");
    if (o.dims[1] != shape.q_heads || o.dims[2] != shape.q_len || o.dims[3] != shape.value_dim)
        throw std::runtime_error(std::string(where) + ": O must be [B, Hq, Sq, Dv]");
    if (shape.kv_heads <= 0 || shape.q_heads % shape.kv_heads != 0)
        throw std::runtime_error(std::string(where) + ": query heads must be a multiple of key/value heads");
    if (a.num_query_heads != 0 && a.num_query_heads != shape.q_heads)
        throw std::runtime_error(std::string(where) + ": AttentionDesc::num_query_heads does not match Q");
    if (a.num_kv_heads != 0 && a.num_kv_heads != shape.kv_heads)
        throw std::runtime_error(std::string(where) + ": AttentionDesc::num_kv_heads does not match K/V");
    if (q.type != k.type || q.type != v.type || q.type != o.type)
        throw std::runtime_error(std::string(where) + ": Q, K, V and O must share one datatype");
    if (a.dropout < 0.0f || a.dropout >= 1.0f)
        throw std::runtime_error(std::string(where) + ": dropout probability must be in [0, 1)");

    if (mask) {
        if (mask->rank() != 4)
            throw std::runtime_error(std::string(where) + ": mask must be rank-4 and broadcastable to [B, Hq, Sq, Skv]");
        const std::int64_t full[4] = {shape.batch, shape.q_heads, shape.q_len, shape.kv_len};
        for (int i = 0; i < 4; ++i) {
            if (mask->dims[i] != 1 && mask->dims[i] != full[i])
                throw std::runtime_error(std::string(where) + ": mask is not broadcastable to [B, Hq, Sq, Skv]");
        }
        shape.has_mask = true;
    }

    shape.scale = a.scale != 0.0
        ? a.scale
        : 1.0 / std::sqrt(static_cast<double>(shape.head_dim));
    return shape;
}

}
}
