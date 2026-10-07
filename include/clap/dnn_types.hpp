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

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clap {

enum class DnnBackendType {
    CPU,
    GPU,
    CUDA,
    ROCM
};

enum class DataType {
    Float32,
    Float16,
    BFloat16
};

enum class TensorLayout {
    NCHW,
    Strided
};

enum class ActivationMode {
    ReLU,
    Sigmoid,
    Tanh,
    SiLU
};

enum class PoolingMode {
    Max,
    Average
};

enum class LrnMode {
    CrossChannel
};

struct LrnDesc {
    std::int64_t local_size = 5;
    double alpha = 1.0e-4;
    double beta = 0.75;
    double k = 1.0;
};

inline std::size_t dataTypeSize(DataType type)
{
    switch (type) {
        case DataType::Float32:  return 4;
        case DataType::Float16:  return 2;
        case DataType::BFloat16: return 2;
    }
    return 4;
}

inline const char* dataTypeName(DataType type)
{
    switch (type) {
        case DataType::Float32:  return "Float32";
        case DataType::Float16:  return "Float16";
        case DataType::BFloat16: return "BFloat16";
    }
    return "unknown";
}

struct TensorDesc {
    DataType type = DataType::Float32;
    TensorLayout layout = TensorLayout::NCHW;
    std::vector<std::int64_t> dims;

    std::vector<std::int64_t> strides;

    TensorDesc() = default;
    TensorDesc(std::initializer_list<std::int64_t> d) : dims(d) {}
    explicit TensorDesc(std::vector<std::int64_t> d) : dims(std::move(d)) {}

    TensorDesc(DataType t,
               std::vector<std::int64_t> d,
               std::vector<std::int64_t> s = {})
        : type(t),
          layout(TensorLayout::Strided),
          dims(std::move(d)),
          strides(std::move(s))
    {
    }

    std::size_t rank() const noexcept { return dims.size(); }

    std::size_t elements() const {
        std::size_t n = 1;
        for (auto d : dims) {
            if (d <= 0) throw std::runtime_error("Tensor dimensions must be positive");
            n *= static_cast<std::size_t>(d);
        }
        return n;
    }

    std::size_t elementSize() const noexcept { return dataTypeSize(type); }

    std::vector<std::int64_t> packedStrides() const {
        std::vector<std::int64_t> s(dims.size(), 1);
        std::int64_t running = 1;
        for (std::size_t i = dims.size(); i-- > 0;) {
            s[i] = running;
            running *= dims[i];
        }
        return s;
    }

    std::vector<std::int64_t> effectiveStrides() const {
        if (strides.empty())
            return packedStrides();
        if (strides.size() != dims.size())
            throw std::runtime_error("TensorDesc: strides must have one entry per dimension");
        return strides;
    }

    bool isPacked() const {
        if (strides.empty())
            return true;
        const auto packed = packedStrides();
        if (strides.size() != dims.size())
            return false;
        for (std::size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] != 1 && strides[i] != packed[i])
                return false;
        }
        return true;
    }

    std::size_t spanElements() const {
        const auto s = effectiveStrides();
        std::size_t span = 1;
        for (std::size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] <= 0) throw std::runtime_error("Tensor dimensions must be positive");
            if (s[i] < 0) throw std::runtime_error("TensorDesc: negative strides are not supported");
            span += static_cast<std::size_t>(dims[i] - 1) * static_cast<std::size_t>(s[i]);
        }
        return span;
    }

    std::size_t bytes() const { return spanElements() * elementSize(); }

    int normalizeAxis(int axis, const char* where) const {
        const int r = static_cast<int>(dims.size());
        const int a = axis < 0 ? axis + r : axis;
        if (r == 0 || a < 0 || a >= r)
            throw std::runtime_error(std::string(where) + ": axis " + std::to_string(axis) +
                                     " is out of range for rank " + std::to_string(r));
        return a;
    }

    void require4d(const char* where) const {
        if (dims.size() != 4)
            throw std::runtime_error(std::string(where) + ": CLAP_DNN4 currently supports 4D NCHW tensors only");
    }
};

struct ConvolutionDesc {
    std::int64_t pad_h = 0;
    std::int64_t pad_w = 0;
    std::int64_t stride_h = 1;
    std::int64_t stride_w = 1;
    std::int64_t dilation_h = 1;
    std::int64_t dilation_w = 1;
};

struct ActivationDesc {
    ActivationMode mode = ActivationMode::ReLU;
    double alpha = 0.0;
};

struct PoolingDesc {
    PoolingMode mode = PoolingMode::Max;
    std::int64_t window_h = 2;
    std::int64_t window_w = 2;
    std::int64_t pad_h = 0;
    std::int64_t pad_w = 0;
    std::int64_t stride_h = 2;
    std::int64_t stride_w = 2;
};

struct BatchNormDesc {
    double epsilon = 1.0e-5;
};

struct LayerNormDesc {
    double epsilon = 1.0e-5;
};

struct DropoutDesc {
    float probability = 0.5f;
    std::uint64_t seed = 12345;
};

enum class ExecutionBackend {
    CPU,
    CUDA,
    ROCM
};

inline const char* executionBackendName(ExecutionBackend backend)
{
    switch (backend) {
        case ExecutionBackend::CPU:  return "CPU";
        case ExecutionBackend::CUDA: return "CUDA";
        case ExecutionBackend::ROCM: return "ROCM";
    }
    return "unknown";
}

struct ExecutionContext {
    ExecutionBackend backend = ExecutionBackend::CPU;
    void* stream = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;

    ExecutionContext() = default;

    ExecutionContext(ExecutionBackend b, void* s = nullptr)
        : backend(b), stream(s)
    {
    }

    static ExecutionContext cpu(void* dnnl_stream = nullptr) { return {ExecutionBackend::CPU, dnnl_stream}; }
    static ExecutionContext cuda(void* cuda_stream) { return {ExecutionBackend::CUDA, cuda_stream}; }
    static ExecutionContext rocm(void* hip_stream) { return {ExecutionBackend::ROCM, hip_stream}; }
};

struct SoftmaxDesc {
    int axis = -1;
    bool log_softmax = false;
};

struct RmsNormDesc {
    double epsilon = 1.0e-6;
    int axis = -1;
};

struct AttentionDesc {
    double scale = 0.0;
    bool causal = false;
    float dropout = 0.0f;
    std::uint64_t seed = 0;
    int num_query_heads = 0;
    int num_kv_heads = 0;
};

enum class DnnOperation {
    ConvolutionForward,
    ConvolutionBackwardData,
    ConvolutionBackwardWeights,
    ConvolutionBackwardBias,
    ConvolutionTransposeForward,
    FusedConvolutionBiasActivation,
    ActivationForward,
    ActivationBackward,
    PoolingForward,
    PoolingBackward,
    SoftmaxForward,
    SoftmaxBackward,
    BatchNormForwardTraining,
    BatchNormForwardInference,
    BatchNormBackward,
    LayerNormForward,
    LayerNormBackward,
    RmsNormForward,
    RmsNormBackward,
    DropoutForward,
    DropoutBackward,
    LrnForward,
    LrnBackward,
    TensorAdd,
    TensorMultiply,
    TensorMin,
    TensorMax,
    TensorReduceSum,
    TensorReduceProduct,
    ScaledDotProductAttentionForward
};

struct DnnCapabilityQuery {
    DnnOperation operation = DnnOperation::ActivationForward;
    std::vector<TensorDesc> tensors;
    ActivationDesc activation;
    SoftmaxDesc softmax;
    RmsNormDesc rms_norm;
    AttentionDesc attention;
    bool has_mask = false;

    static DnnCapabilityQuery forActivation(DnnOperation op, const ActivationDesc& a, const TensorDesc& x)
    {
        DnnCapabilityQuery q;
        q.operation = op;
        q.activation = a;
        q.tensors = {x};
        return q;
    }

    static DnnCapabilityQuery forSoftmax(DnnOperation op, const SoftmaxDesc& s, const TensorDesc& x)
    {
        DnnCapabilityQuery q;
        q.operation = op;
        q.softmax = s;
        q.tensors = {x};
        return q;
    }

    static DnnCapabilityQuery forRmsNorm(DnnOperation op, const RmsNormDesc& n, const TensorDesc& x, const TensorDesc& weight)
    {
        DnnCapabilityQuery q;
        q.operation = op;
        q.rms_norm = n;
        q.tensors = {x, weight, x};
        return q;
    }

    static DnnCapabilityQuery forAttention(const AttentionDesc& a,
                                           const TensorDesc& q_desc,
                                           const TensorDesc& k_desc,
                                           const TensorDesc& v_desc,
                                           const TensorDesc& o_desc,
                                           const TensorDesc* mask_desc = nullptr)
    {
        DnnCapabilityQuery q;
        q.operation = DnnOperation::ScaledDotProductAttentionForward;
        q.attention = a;
        q.tensors = {q_desc, k_desc, v_desc, o_desc};
        if (mask_desc) {
            q.tensors.push_back(*mask_desc);
            q.has_mask = true;
        }
        return q;
    }

    static DnnCapabilityQuery forOperation(DnnOperation op, const TensorDesc& x)
    {
        DnnCapabilityQuery q;
        q.operation = op;
        q.tensors = {x};
        return q;
    }

    const TensorDesc& tensor(std::size_t i) const
    {
        if (tensors.empty())
            throw std::runtime_error("DnnCapabilityQuery requires at least one tensor descriptor");
        return i < tensors.size() ? tensors[i] : tensors[0];
    }
};

struct DnnSupport {
    bool supported = false;
    std::string reason;

    explicit operator bool() const noexcept { return supported; }

    static DnnSupport yes(std::string note = {}) { return {true, std::move(note)}; }
    static DnnSupport no(std::string why) { return {false, std::move(why)}; }
};

class DnnUnsupportedError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline const char* dnnOperationName(DnnOperation operation) noexcept
{
    switch (operation) {
        case DnnOperation::ConvolutionForward:               return "ConvolutionForward";
        case DnnOperation::ConvolutionBackwardData:          return "ConvolutionBackwardData";
        case DnnOperation::ConvolutionBackwardWeights:       return "ConvolutionBackwardWeights";
        case DnnOperation::ConvolutionBackwardBias:          return "ConvolutionBackwardBias";
        case DnnOperation::ConvolutionTransposeForward:      return "ConvolutionTransposeForward";
        case DnnOperation::FusedConvolutionBiasActivation:   return "FusedConvolutionBiasActivation";
        case DnnOperation::ActivationForward:                return "ActivationForward";
        case DnnOperation::ActivationBackward:               return "ActivationBackward";
        case DnnOperation::PoolingForward:                   return "PoolingForward";
        case DnnOperation::PoolingBackward:                  return "PoolingBackward";
        case DnnOperation::SoftmaxForward:                   return "SoftmaxForward";
        case DnnOperation::SoftmaxBackward:                  return "SoftmaxBackward";
        case DnnOperation::BatchNormForwardTraining:         return "BatchNormForwardTraining";
        case DnnOperation::BatchNormForwardInference:        return "BatchNormForwardInference";
        case DnnOperation::BatchNormBackward:                return "BatchNormBackward";
        case DnnOperation::LayerNormForward:                 return "LayerNormForward";
        case DnnOperation::LayerNormBackward:                return "LayerNormBackward";
        case DnnOperation::RmsNormForward:                   return "RmsNormForward";
        case DnnOperation::RmsNormBackward:                  return "RmsNormBackward";
        case DnnOperation::DropoutForward:                   return "DropoutForward";
        case DnnOperation::DropoutBackward:                  return "DropoutBackward";
        case DnnOperation::LrnForward:                       return "LrnForward";
        case DnnOperation::LrnBackward:                      return "LrnBackward";
        case DnnOperation::TensorAdd:                        return "TensorAdd";
        case DnnOperation::TensorMultiply:                   return "TensorMultiply";
        case DnnOperation::TensorMin:                        return "TensorMin";
        case DnnOperation::TensorMax:                        return "TensorMax";
        case DnnOperation::TensorReduceSum:                  return "TensorReduceSum";
        case DnnOperation::TensorReduceProduct:              return "TensorReduceProduct";
        case DnnOperation::ScaledDotProductAttentionForward: return "ScaledDotProductAttentionForward";
    }
    return "unknown";
}

}
