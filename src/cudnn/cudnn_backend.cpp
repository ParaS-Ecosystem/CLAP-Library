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

#include "clap/cudnn_backend.hpp"
#include "clap/dnn_dyn_backends.hpp"

#include "clap/dnn_contract_utils.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace clap {

namespace {

void checkCuda(abi::cudaError_t status, const char* where)
{
    if (status != 0)
        throw std::runtime_error(std::string("CLAP_DNN CUDA error in ") + where +
                                 " (status=" + std::to_string(status) + ")");
}

void checkCudnn(abi::cudnnStatus_t status, const char* where)
{
    if (status == abi::CUDNN_STATUS_NOT_SUPPORTED)
        throw DnnUnsupportedError(std::string("CLAP_DNN cuDNN: no native support in ") + where +
                                  " (CUDNN_STATUS_NOT_SUPPORTED)");
    if (status != abi::CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string("CLAP_DNN cuDNN error in ") + where +
                                 " (status=" + std::to_string(status) + ")");
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t bytes)
    {
        allocate(bytes);
    }

    ~DeviceBuffer()
    {
        reset();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t bytes)
    {
        reset();
        bytes_ = bytes;
        if (bytes_ != 0)
            checkCuda(dnn_dyn::p_cudaMalloc(&ptr_, bytes_), "cudaMalloc");
    }

    void reset() noexcept
    {
        if (ptr_ && dnn_dyn::p_cudaFree)
            dnn_dyn::p_cudaFree(ptr_);
        ptr_ = nullptr;
        bytes_ = 0;
    }

    void copyFromHost(const void* src, std::size_t bytes)
    {
        if (bytes > bytes_)
            throw std::runtime_error("DeviceBuffer::copyFromHost size exceeds allocation");
        checkCuda(dnn_dyn::p_cudaMemcpy(ptr_, src, bytes, abi::cudaMemcpyHostToDevice),
                  "cudaMemcpy H2D");
    }

    void copyToHost(void* dst, std::size_t bytes) const
    {
        if (bytes > bytes_)
            throw std::runtime_error("DeviceBuffer::copyToHost size exceeds allocation");
        checkCuda(dnn_dyn::p_cudaMemcpy(dst, ptr_, bytes, abi::cudaMemcpyDeviceToHost),
                  "cudaMemcpy D2H");
    }

    void zero()
    {
        if (bytes_ != 0)
            checkCuda(dnn_dyn::p_cudaMemset(ptr_, 0, bytes_), "cudaMemset");
    }

    void* get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return bytes_; }

private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

class StreamScratch {
public:
    explicit StreamScratch(const ExecutionContext& ctx)
        : stream_(ctx.stream),
          base_(static_cast<char*>(ctx.workspace)),
          capacity_(ctx.workspace ? ctx.workspace_bytes : 0)
    {
    }

    ~StreamScratch()
    {
        for (void* p : owned_) {
            if (dnn_dyn::p_cudaFreeAsync)
                dnn_dyn::p_cudaFreeAsync(p, stream_);
        }
    }

    StreamScratch(const StreamScratch&) = delete;
    StreamScratch& operator=(const StreamScratch&) = delete;

    void* get(std::size_t bytes)
    {
        if (bytes == 0)
            return nullptr;

        const std::size_t aligned = (offset_ + 255) & ~static_cast<std::size_t>(255);
        if (base_ && aligned + bytes <= capacity_) {
            offset_ = aligned + bytes;
            return base_ + aligned;
        }

        void* p = nullptr;
        checkCuda(dnn_dyn::p_cudaMallocAsync(&p, bytes, stream_), "cudaMallocAsync");
        owned_.push_back(p);
        return p;
    }

private:
    abi::cudaStream_t stream_ = nullptr;
    char* base_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t offset_ = 0;
    std::vector<void*> owned_;
};

abi::cudnnDataType_t toCudnnType(DataType type)
{
    switch (type) {
        case DataType::Float32:  return abi::CUDNN_DATA_FLOAT;
        case DataType::Float16:  return abi::CUDNN_DATA_HALF;
        case DataType::BFloat16: return abi::CUDNN_DATA_BFLOAT16;
    }
    return abi::CUDNN_DATA_FLOAT;
}

int toInt(std::int64_t value, const char* where)
{
    if (value < 0 || value > INT_MAX)
        throw DnnUnsupportedError(std::string("CLAP_DNN cuDNN ") + where +
                                  ": tensor extent or stride does not fit the cuDNN int descriptor");
    return static_cast<int>(value);
}

class TensorDescriptor {
public:
    explicit TensorDescriptor(const TensorDesc& d)
    {
        checkCudnn(dnn_dyn::p_cudnnCreateTensorDescriptor(&desc_),
                   "cudnnCreateTensorDescriptor");

        try {
            set(d);
        }
        catch (...) {
            dnn_dyn::p_cudnnDestroyTensorDescriptor(desc_);
            throw;
        }
    }

    TensorDescriptor(int n, int c, int h, int w)
        : TensorDescriptor(DataType::Float32, n, c, h, w)
    {
    }

    TensorDescriptor(DataType type, int n, int c, int h, int w)
    {
        checkCudnn(dnn_dyn::p_cudnnCreateTensorDescriptor(&desc_),
                   "cudnnCreateTensorDescriptor");

        checkCudnn(
            dnn_dyn::p_cudnnSetTensor4dDescriptor(
                desc_, abi::CUDNN_TENSOR_NCHW, toCudnnType(type),
                n, c, h, w),
            "cudnnSetTensor4dDescriptor");
    }

    TensorDescriptor(DataType type,
                     std::int64_t n, std::int64_t c, std::int64_t h,
                     std::int64_t n_stride, std::int64_t c_stride, std::int64_t h_stride)
    {
        checkCudnn(dnn_dyn::p_cudnnCreateTensorDescriptor(&desc_),
                   "cudnnCreateTensorDescriptor");

        try {
            checkCudnn(
                dnn_dyn::p_cudnnSetTensor4dDescriptorEx(
                    desc_,
                    toCudnnType(type),
                    toInt(n, "TensorDescriptor"),
                    toInt(c, "TensorDescriptor"),
                    toInt(h, "TensorDescriptor"),
                    1,
                    toInt(n_stride, "TensorDescriptor"),
                    toInt(c_stride, "TensorDescriptor"),
                    toInt(h_stride, "TensorDescriptor"),
                    1),
                "cudnnSetTensor4dDescriptorEx");
        }
        catch (...) {
            dnn_dyn::p_cudnnDestroyTensorDescriptor(desc_);
            throw;
        }
    }

    ~TensorDescriptor()
    {
        if (desc_ && dnn_dyn::p_cudnnDestroyTensorDescriptor)
            dnn_dyn::p_cudnnDestroyTensorDescriptor(desc_);
    }

    TensorDescriptor(const TensorDescriptor&) = delete;
    TensorDescriptor& operator=(const TensorDescriptor&) = delete;

    operator abi::cudnnTensorDescriptor_t() const noexcept { return desc_; }

private:
    void set(const TensorDesc& d)
    {
        if (d.dims.empty() || d.dims.size() > 8)
            throw DnnUnsupportedError("CLAP_DNN cuDNN: tensors must have rank 1..8");

        const auto s = d.effectiveStrides();

        if (d.dims.size() == 4) {
            checkCudnn(
                dnn_dyn::p_cudnnSetTensor4dDescriptorEx(
                    desc_,
                    toCudnnType(d.type),
                    toInt(d.dims[0], "TensorDescriptor"),
                    toInt(d.dims[1], "TensorDescriptor"),
                    toInt(d.dims[2], "TensorDescriptor"),
                    toInt(d.dims[3], "TensorDescriptor"),
                    toInt(s[0], "TensorDescriptor"),
                    toInt(s[1], "TensorDescriptor"),
                    toInt(s[2], "TensorDescriptor"),
                    toInt(s[3], "TensorDescriptor")),
                "cudnnSetTensor4dDescriptorEx");
            return;
        }

        std::vector<int> dims;
        std::vector<int> strides;
        const std::size_t pad = d.dims.size() < 4 ? 4 - d.dims.size() : 0;
        const std::int64_t outer_stride = d.dims[0] * s[0];

        for (std::size_t i = 0; i < pad; ++i) {
            dims.push_back(1);
            strides.push_back(toInt(outer_stride, "TensorDescriptor"));
        }
        for (std::size_t i = 0; i < d.dims.size(); ++i) {
            dims.push_back(toInt(d.dims[i], "TensorDescriptor"));
            strides.push_back(toInt(s[i], "TensorDescriptor"));
        }

        checkCudnn(
            dnn_dyn::p_cudnnSetTensorNdDescriptor(
                desc_,
                toCudnnType(d.type),
                static_cast<int>(dims.size()),
                dims.data(),
                strides.data()),
            "cudnnSetTensorNdDescriptor");
    }

    abi::cudnnTensorDescriptor_t desc_ = nullptr;
};

class FilterDescriptor {
public:
    explicit FilterDescriptor(const TensorDesc& d)
    {
        d.require4d("FilterDescriptor");
        if (!d.isPacked())
            throw DnnUnsupportedError("CLAP_DNN cuDNN: convolution filters must be packed KCRS tensors");

        checkCudnn(dnn_dyn::p_cudnnCreateFilterDescriptor(&desc_),
                   "cudnnCreateFilterDescriptor");

        checkCudnn(
            dnn_dyn::p_cudnnSetFilter4dDescriptor(
                desc_,
                toCudnnType(d.type),
                abi::CUDNN_TENSOR_NCHW,
                static_cast<int>(d.dims[0]),
                static_cast<int>(d.dims[1]),
                static_cast<int>(d.dims[2]),
                static_cast<int>(d.dims[3])),
            "cudnnSetFilter4dDescriptor");
    }

    ~FilterDescriptor()
    {
        if (desc_ && dnn_dyn::p_cudnnDestroyFilterDescriptor)
            dnn_dyn::p_cudnnDestroyFilterDescriptor(desc_);
    }

    operator abi::cudnnFilterDescriptor_t() const noexcept { return desc_; }

private:
    abi::cudnnFilterDescriptor_t desc_ = nullptr;
};

class ConvolutionDescriptor {
public:
    explicit ConvolutionDescriptor(const ConvolutionDesc& c)
    {
        checkCudnn(dnn_dyn::p_cudnnCreateConvolutionDescriptor(&desc_),
                   "cudnnCreateConvolutionDescriptor");

        checkCudnn(
            dnn_dyn::p_cudnnSetConvolution2dDescriptor(
                desc_,
                static_cast<int>(c.pad_h),
                static_cast<int>(c.pad_w),
                static_cast<int>(c.stride_h),
                static_cast<int>(c.stride_w),
                static_cast<int>(c.dilation_h),
                static_cast<int>(c.dilation_w),
                abi::CUDNN_CROSS_CORRELATION,
                abi::CUDNN_DATA_FLOAT),
            "cudnnSetConvolution2dDescriptor");
    }

    ~ConvolutionDescriptor()
    {
        if (desc_ && dnn_dyn::p_cudnnDestroyConvolutionDescriptor)
            dnn_dyn::p_cudnnDestroyConvolutionDescriptor(desc_);
    }

    operator abi::cudnnConvolutionDescriptor_t() const noexcept { return desc_; }

private:
    abi::cudnnConvolutionDescriptor_t desc_ = nullptr;
};

abi::cudnnActivationMode_t toCudnnActivation(ActivationMode mode)
{
    switch (mode) {
        case ActivationMode::ReLU:    return abi::CUDNN_ACTIVATION_RELU;
        case ActivationMode::Sigmoid: return abi::CUDNN_ACTIVATION_SIGMOID;
        case ActivationMode::Tanh:    return abi::CUDNN_ACTIVATION_TANH;
        case ActivationMode::SiLU:    return abi::CUDNN_ACTIVATION_SWISH;
    }
    return abi::CUDNN_ACTIVATION_RELU;
}

class ActivationDescriptor {
public:
    explicit ActivationDescriptor(const ActivationDesc& a)
    {
        checkCudnn(dnn_dyn::p_cudnnCreateActivationDescriptor(&desc_),
                   "cudnnCreateActivationDescriptor");

        checkCudnn(
            dnn_dyn::p_cudnnSetActivationDescriptor(
                desc_, toCudnnActivation(a.mode), abi::CUDNN_PROPAGATE_NAN, a.alpha),
            "cudnnSetActivationDescriptor");

        if (a.mode == ActivationMode::SiLU) {
            checkCudnn(
                dnn_dyn::p_cudnnSetActivationDescriptorSwishBeta(desc_, 1.0),
                "cudnnSetActivationDescriptorSwishBeta");
        }
    }

    ~ActivationDescriptor()
    {
        if (desc_ && dnn_dyn::p_cudnnDestroyActivationDescriptor)
            dnn_dyn::p_cudnnDestroyActivationDescriptor(desc_);
    }

    operator abi::cudnnActivationDescriptor_t() const noexcept { return desc_; }

private:
    abi::cudnnActivationDescriptor_t desc_ = nullptr;
};

abi::cudnnPoolingMode_t toCudnnPooling(PoolingMode mode)
{
    return mode == PoolingMode::Max
        ? abi::CUDNN_POOLING_MAX
        : abi::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
}

class PoolingDescriptor {
public:
    explicit PoolingDescriptor(const PoolingDesc& p)
    {
        checkCudnn(dnn_dyn::p_cudnnCreatePoolingDescriptor(&desc_),
                   "cudnnCreatePoolingDescriptor");

        checkCudnn(
            dnn_dyn::p_cudnnSetPooling2dDescriptor(
                desc_,
                toCudnnPooling(p.mode),
                abi::CUDNN_PROPAGATE_NAN,
                static_cast<int>(p.window_h),
                static_cast<int>(p.window_w),
                static_cast<int>(p.pad_h),
                static_cast<int>(p.pad_w),
                static_cast<int>(p.stride_h),
                static_cast<int>(p.stride_w)),
            "cudnnSetPooling2dDescriptor");
    }

    ~PoolingDescriptor()
    {
        if (desc_ && dnn_dyn::p_cudnnDestroyPoolingDescriptor)
            dnn_dyn::p_cudnnDestroyPoolingDescriptor(desc_);
    }

    operator abi::cudnnPoolingDescriptor_t() const noexcept { return desc_; }

private:
    abi::cudnnPoolingDescriptor_t desc_ = nullptr;
};

void requireCuda()
{
    if (!dnn_dyn::loadCudaAndCudnn())
        throw std::runtime_error(std::string("cuDNN/NVIDIA runtime load failed: ") +
                                 dnn_dyn::lastError());
}

std::size_t channelCount(const TensorDesc& d)
{
    d.require4d("channelCount");
    return static_cast<std::size_t>(d.dims[1]);
}

ExecutionContext legacyContext()
{
    return ExecutionContext::cuda(nullptr);
}

void synchronizeLegacy()
{
    checkCuda(dnn_dyn::p_cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");
}

void requireCudaContext(const ExecutionContext& ctx, const char* operation)
{
    dnn_contract::requireBackend(ctx, ExecutionBackend::CUDA, "cuDNN", operation);
}

class ScopedHandle {
public:
    ScopedHandle(std::recursive_mutex& mutex,
                 std::map<int, abi::cudnnHandle_t>& handles,
                 const ExecutionContext& ctx)
        : lock_(mutex)
    {
        int device = 0;
        checkCuda(dnn_dyn::p_cudaGetDevice(&device), "cudaGetDevice");

        auto it = handles.find(device);
        if (it == handles.end()) {
            abi::cudnnHandle_t h = nullptr;
            checkCudnn(dnn_dyn::p_cudnnCreate(&h), "cudnnCreate");
            it = handles.emplace(device, h).first;
        }

        handle_ = it->second;
        checkCudnn(dnn_dyn::p_cudnnSetStream(handle_, ctx.stream), "cudnnSetStream");
    }

    operator abi::cudnnHandle_t() const noexcept { return handle_; }

private:
    std::lock_guard<std::recursive_mutex> lock_;
    abi::cudnnHandle_t handle_ = nullptr;
};

struct LayerNormJit {
    abi::CUmodule module = nullptr;
    abi::CUfunction forward = nullptr;
    abi::CUfunction backward = nullptr;
    bool ready = false;

    ~LayerNormJit()
    {
        if (module && dnn_dyn::p_cuModuleUnload)
            dnn_dyn::p_cuModuleUnload(module);
    }
};

LayerNormJit& layerNormJit()
{
    static LayerNormJit jit;

    if (jit.ready)
        return jit;

    if (!dnn_dyn::loadCudaJit())
        throw std::runtime_error(std::string("CUDA JIT load failed: ") + dnn_dyn::lastError());

    static const char* source = R"CUDA(
extern "C" __global__
void clap_layernorm_forward(const float* x,
                            const float* scale,
                            const float* bias,
                            float* y,
                            float* means,
                            float* rstds,
                            int outer,
                            int inner,
                            float eps)
{
    const int row = (int)blockIdx.x;
    if (row >= outer || threadIdx.x != 0) return;

    const float* xr = x + (long long)row * inner;
    float* yr = y + (long long)row * inner;

    float mean = 0.0f;
    for (int j = 0; j < inner; ++j) mean += xr[j];
    mean /= (float)inner;

    float var = 0.0f;
    for (int j = 0; j < inner; ++j) {
        float v = xr[j] - mean;
        var += v * v;
    }
    var /= (float)inner;

    float rstd = rsqrtf(var + eps);
    if (means) means[row] = mean;
    if (rstds) rstds[row] = rstd;

    for (int j = 0; j < inner; ++j) {
        float xhat = (xr[j] - mean) * rstd;
        yr[j] = xhat * scale[j] + bias[j];
    }
}

extern "C" __global__
void clap_layernorm_backward(const float* x,
                             const float* dy,
                             const float* scale,
                             const float* means,
                             const float* rstds,
                             float* dx,
                             float* dscale,
                             float* dbias,
                             int outer,
                             int inner)
{
    const int row = (int)blockIdx.x;
    if (row >= outer || threadIdx.x != 0) return;

    const float* xr = x + (long long)row * inner;
    const float* dyr = dy + (long long)row * inner;
    float* dxr = dx + (long long)row * inner;

    const float mean = means[row];
    const float rstd = rstds[row];

    float sum_dyg = 0.0f;
    float sum_dyg_xhat = 0.0f;

    for (int j = 0; j < inner; ++j) {
        float xhat = (xr[j] - mean) * rstd;
        float dyg = dyr[j] * scale[j];
        sum_dyg += dyg;
        sum_dyg_xhat += dyg * xhat;
        atomicAdd(dscale + j, dyr[j] * xhat);
        atomicAdd(dbias + j, dyr[j]);
    }

    float inv_n = 1.0f / (float)inner;
    for (int j = 0; j < inner; ++j) {
        float xhat = (xr[j] - mean) * rstd;
        float dyg = dyr[j] * scale[j];
        dxr[j] = rstd * inv_n *
                 ((float)inner * dyg - sum_dyg - xhat * sum_dyg_xhat);
    }
}
)CUDA";

    abi::nvrtcProgram prog = nullptr;
    if (dnn_dyn::p_nvrtcCreateProgram(&prog, source, "clap_layernorm.cu", 0, nullptr, nullptr) != abi::NVRTC_SUCCESS)
        throw std::runtime_error("nvrtcCreateProgram failed for LayerNorm");

    const char* opts[] = {"--std=c++14"};
    const auto compile_status = dnn_dyn::p_nvrtcCompileProgram(prog, 1, opts);

    if (compile_status != abi::NVRTC_SUCCESS) {
        std::size_t log_size = 0;
        dnn_dyn::p_nvrtcGetProgramLogSize(prog, &log_size);
        std::vector<char> log(log_size + 1, '\0');
        if (log_size)
            dnn_dyn::p_nvrtcGetProgramLog(prog, log.data());
        dnn_dyn::p_nvrtcDestroyProgram(&prog);
        throw std::runtime_error(std::string("NVRTC LayerNorm compilation failed: ") + log.data());
    }

    std::size_t ptx_size = 0;
    if (dnn_dyn::p_nvrtcGetPTXSize(prog, &ptx_size) != abi::NVRTC_SUCCESS) {
        dnn_dyn::p_nvrtcDestroyProgram(&prog);
        throw std::runtime_error("nvrtcGetPTXSize failed for LayerNorm");
    }

    std::vector<char> ptx(ptx_size);
    if (dnn_dyn::p_nvrtcGetPTX(prog, ptx.data()) != abi::NVRTC_SUCCESS) {
        dnn_dyn::p_nvrtcDestroyProgram(&prog);
        throw std::runtime_error("nvrtcGetPTX failed for LayerNorm");
    }

    dnn_dyn::p_nvrtcDestroyProgram(&prog);

    if (dnn_dyn::p_cuModuleLoadData(&jit.module, ptx.data()) != abi::CUDA_SUCCESS)
        throw std::runtime_error("cuModuleLoadData failed for LayerNorm");

    if (dnn_dyn::p_cuModuleGetFunction(&jit.forward, jit.module, "clap_layernorm_forward") != abi::CUDA_SUCCESS)
        throw std::runtime_error("cuModuleGetFunction forward failed for LayerNorm");

    if (dnn_dyn::p_cuModuleGetFunction(&jit.backward, jit.module, "clap_layernorm_backward") != abi::CUDA_SUCCESS)
        throw std::runtime_error("cuModuleGetFunction backward failed for LayerNorm");

    jit.ready = true;
    return jit;
}

std::size_t legacyBytes(const TensorDesc& d)
{
    if (d.type != DataType::Float32)
        throw std::runtime_error("CLAP_DNN cuDNN: the float* host API requires Float32 descriptors; "
                                 "use the ExecutionContext overloads for Float16/BFloat16");
    return d.bytes();
}

}

namespace cudnn_detail {

class BackendDescriptor {
public:
    explicit BackendDescriptor(abi::cudnnBackendDescriptorType_t type)
    {
        checkCudnn(dnn_dyn::p_cudnnBackendCreateDescriptor(type, &desc_),
                   "cudnnBackendCreateDescriptor");
    }

    ~BackendDescriptor()
    {
        if (desc_ && dnn_dyn::p_cudnnBackendDestroyDescriptor)
            dnn_dyn::p_cudnnBackendDestroyDescriptor(desc_);
    }

    BackendDescriptor(const BackendDescriptor&) = delete;
    BackendDescriptor& operator=(const BackendDescriptor&) = delete;

    void set(abi::cudnnBackendAttributeName_t name,
             abi::cudnnBackendAttributeType_t type,
             std::int64_t count,
             const void* values,
             const char* what)
    {
        checkCudnn(
            dnn_dyn::p_cudnnBackendSetAttribute(desc_, name, type, count, values),
            what);
    }

    void setDescriptor(abi::cudnnBackendAttributeName_t name,
                       abi::cudnnBackendDescriptor_t value,
                       const char* what)
    {
        set(name, abi::CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &value, what);
    }

    void finalize(const char* what)
    {
        checkCudnn(dnn_dyn::p_cudnnBackendFinalize(desc_), what);
    }

    abi::cudnnBackendDescriptor_t get() const noexcept { return desc_; }

private:
    abi::cudnnBackendDescriptor_t desc_ = nullptr;
};

struct GraphPlan {
    std::vector<std::unique_ptr<BackendDescriptor>> descriptors;
    std::unique_ptr<BackendDescriptor> plan;
    std::int64_t workspace_bytes = 0;
};

std::vector<std::int64_t> packedStrides(const std::vector<std::int64_t>& dims)
{
    std::vector<std::int64_t> s(dims.size(), 1);
    std::int64_t running = 1;
    for (std::size_t i = dims.size(); i-- > 0;) {
        s[i] = running;
        running *= dims[i];
    }
    return s;
}

class GraphBuilder {
public:
    abi::cudnnBackendDescriptor_t tensor(std::int64_t uid,
                                         abi::cudnnDataType_t type,
                                         const std::vector<std::int64_t>& dims,
                                         const std::vector<std::int64_t>& strides,
                                         bool by_value = false)
    {
        return makeTensor(uid, type, dims, strides, false, by_value);
    }

    abi::cudnnBackendDescriptor_t virtualTensor(abi::cudnnDataType_t type,
                                                const std::vector<std::int64_t>& dims)
    {
        return makeTensor(next_virtual_uid_++, type, dims, packedStrides(dims), true, false);
    }

    void pointwise(int mode,
                   abi::cudnnBackendDescriptor_t x,
                   abi::cudnnBackendDescriptor_t b,
                   abi::cudnnBackendDescriptor_t t,
                   abi::cudnnBackendDescriptor_t y,
                   abi::cudnnDataType_t math_type = abi::CUDNN_DATA_FLOAT,
                   std::int64_t axis = -1)
    {
        BackendDescriptor& pw = make(abi::CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        pw.set(abi::CUDNN_ATTR_POINTWISE_MODE, abi::CUDNN_TYPE_POINTWISE_MODE, 1, &mode, "CUDNN_ATTR_POINTWISE_MODE");
        pw.set(abi::CUDNN_ATTR_POINTWISE_MATH_PREC, abi::CUDNN_TYPE_DATA_TYPE, 1, &math_type, "CUDNN_ATTR_POINTWISE_MATH_PREC");
        if (axis >= 0)
            pw.set(abi::CUDNN_ATTR_POINTWISE_AXIS, abi::CUDNN_TYPE_INT64, 1, &axis, "CUDNN_ATTR_POINTWISE_AXIS");
        pw.finalize("pointwise descriptor finalize");

        BackendDescriptor& op = make(abi::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pw.get(), "CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_POINTWISE_XDESC, x, "CUDNN_ATTR_OPERATION_POINTWISE_XDESC");
        if (b)
            op.setDescriptor(abi::CUDNN_ATTR_OPERATION_POINTWISE_BDESC, b, "CUDNN_ATTR_OPERATION_POINTWISE_BDESC");
        if (t)
            op.setDescriptor(abi::CUDNN_ATTR_OPERATION_POINTWISE_TDESC, t, "CUDNN_ATTR_OPERATION_POINTWISE_TDESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_POINTWISE_YDESC, y, "CUDNN_ATTR_OPERATION_POINTWISE_YDESC");
        op.finalize("pointwise operation finalize");
        ops_.push_back(op.get());
    }

    void reduction(int reduce_op,
                   abi::cudnnBackendDescriptor_t x,
                   abi::cudnnBackendDescriptor_t y)
    {
        const abi::cudnnDataType_t comp = abi::CUDNN_DATA_FLOAT;

        BackendDescriptor& red = make(abi::CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        red.set(abi::CUDNN_ATTR_REDUCTION_OPERATOR, abi::CUDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &reduce_op, "CUDNN_ATTR_REDUCTION_OPERATOR");
        red.set(abi::CUDNN_ATTR_REDUCTION_COMP_TYPE, abi::CUDNN_TYPE_DATA_TYPE, 1, &comp, "CUDNN_ATTR_REDUCTION_COMP_TYPE");
        red.finalize("reduction descriptor finalize");

        BackendDescriptor& op = make(abi::CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_REDUCTION_XDESC, x, "CUDNN_ATTR_OPERATION_REDUCTION_XDESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_REDUCTION_YDESC, y, "CUDNN_ATTR_OPERATION_REDUCTION_YDESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_REDUCTION_DESC, red.get(), "CUDNN_ATTR_OPERATION_REDUCTION_DESC");
        op.finalize("reduction operation finalize");
        ops_.push_back(op.get());
    }

    void matmul(abi::cudnnBackendDescriptor_t a,
                abi::cudnnBackendDescriptor_t b,
                abi::cudnnBackendDescriptor_t c)
    {
        const abi::cudnnDataType_t comp = abi::CUDNN_DATA_FLOAT;

        BackendDescriptor& mm = make(abi::CUDNN_BACKEND_MATMUL_DESCRIPTOR);
        mm.set(abi::CUDNN_ATTR_MATMUL_COMP_TYPE, abi::CUDNN_TYPE_DATA_TYPE, 1, &comp, "CUDNN_ATTR_MATMUL_COMP_TYPE");
        mm.finalize("matmul descriptor finalize");

        BackendDescriptor& op = make(abi::CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_MATMUL_ADESC, a, "CUDNN_ATTR_OPERATION_MATMUL_ADESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_MATMUL_BDESC, b, "CUDNN_ATTR_OPERATION_MATMUL_BDESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_MATMUL_CDESC, c, "CUDNN_ATTR_OPERATION_MATMUL_CDESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_MATMUL_DESC, mm.get(), "CUDNN_ATTR_OPERATION_MATMUL_DESC");
        op.finalize("matmul operation finalize");
        ops_.push_back(op.get());
    }

    void normForward(int mode,
                     int phase,
                     abi::cudnnBackendDescriptor_t x,
                     abi::cudnnBackendDescriptor_t scale,
                     abi::cudnnBackendDescriptor_t epsilon,
                     abi::cudnnBackendDescriptor_t y,
                     abi::cudnnBackendDescriptor_t inv_variance)
    {
        BackendDescriptor& op = make(abi::CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR);
        op.set(abi::CUDNN_ATTR_OPERATION_NORM_FWD_MODE, abi::CUDNN_TYPE_NORM_MODE, 1, &mode, "CUDNN_ATTR_OPERATION_NORM_FWD_MODE");
        op.set(abi::CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, abi::CUDNN_TYPE_NORM_FWD_PHASE, 1, &phase, "CUDNN_ATTR_OPERATION_NORM_FWD_PHASE");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, x, "CUDNN_ATTR_OPERATION_NORM_FWD_XDESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, scale, "CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, epsilon, "CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, y, "CUDNN_ATTR_OPERATION_NORM_FWD_YDESC");
        if (inv_variance)
            op.setDescriptor(abi::CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, inv_variance, "CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC");
        op.finalize("norm forward operation finalize");
        ops_.push_back(op.get());
    }

    void normBackward(int mode,
                      abi::cudnnBackendDescriptor_t x,
                      abi::cudnnBackendDescriptor_t dy,
                      abi::cudnnBackendDescriptor_t scale,
                      abi::cudnnBackendDescriptor_t inv_variance,
                      abi::cudnnBackendDescriptor_t dx,
                      abi::cudnnBackendDescriptor_t dscale)
    {
        BackendDescriptor& op = make(abi::CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR);
        op.set(abi::CUDNN_ATTR_OPERATION_NORM_BWD_MODE, abi::CUDNN_TYPE_NORM_MODE, 1, &mode, "CUDNN_ATTR_OPERATION_NORM_BWD_MODE");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, x, "CUDNN_ATTR_OPERATION_NORM_BWD_XDESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, dy, "CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, scale, "CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, inv_variance, "CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, dx, "CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC");
        op.setDescriptor(abi::CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, dscale, "CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC");
        op.finalize("norm backward operation finalize");
        ops_.push_back(op.get());
    }

    std::unique_ptr<GraphPlan> build(abi::cudnnHandle_t handle, const char* what)
    {
        std::unique_ptr<GraphPlan> result(new GraphPlan());

        BackendDescriptor& graph = make(abi::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        graph.set(abi::CUDNN_ATTR_OPERATIONGRAPH_HANDLE, abi::CUDNN_TYPE_HANDLE, 1, &handle, "CUDNN_ATTR_OPERATIONGRAPH_HANDLE");
        graph.set(
            abi::CUDNN_ATTR_OPERATIONGRAPH_OPS,
            abi::CUDNN_TYPE_BACKEND_DESCRIPTOR,
            static_cast<std::int64_t>(ops_.size()),
            ops_.data(),
            "CUDNN_ATTR_OPERATIONGRAPH_OPS");
        graph.finalize("operation graph finalize");

        const int modes[] = {abi::CUDNN_HEUR_MODE_A, abi::CUDNN_HEUR_MODE_FALLBACK};

        for (int mode : modes) {
            std::unique_ptr<BackendDescriptor> heur(new BackendDescriptor(abi::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR));
            const abi::cudnnBackendDescriptor_t graph_desc = graph.get();
            heur->setDescriptor(abi::CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, graph_desc, "CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH");
            heur->set(abi::CUDNN_ATTR_ENGINEHEUR_MODE, abi::CUDNN_TYPE_HEUR_MODE, 1, &mode, "CUDNN_ATTR_ENGINEHEUR_MODE");
            if (dnn_dyn::p_cudnnBackendFinalize(heur->get()) != abi::CUDNN_STATUS_SUCCESS)
                continue;

            std::int64_t count = 0;
            if (dnn_dyn::p_cudnnBackendGetAttribute(
                    heur->get(),
                    abi::CUDNN_ATTR_ENGINEHEUR_RESULTS,
                    abi::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    0,
                    &count,
                    nullptr) != abi::CUDNN_STATUS_SUCCESS || count <= 0)
                continue;

            std::vector<std::unique_ptr<BackendDescriptor>> configs;
            std::vector<abi::cudnnBackendDescriptor_t> raw;
            for (std::int64_t i = 0; i < count; ++i) {
                configs.emplace_back(new BackendDescriptor(abi::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
                raw.push_back(configs.back()->get());
            }

            std::int64_t returned = 0;
            checkCudnn(
                dnn_dyn::p_cudnnBackendGetAttribute(
                    heur->get(),
                    abi::CUDNN_ATTR_ENGINEHEUR_RESULTS,
                    abi::CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    count,
                    &returned,
                    raw.data()),
                "CUDNN_ATTR_ENGINEHEUR_RESULTS");

            for (std::int64_t i = 0; i < returned; ++i) {
                std::unique_ptr<BackendDescriptor> plan(new BackendDescriptor(abi::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR));
                plan->set(abi::CUDNN_ATTR_EXECUTION_PLAN_HANDLE, abi::CUDNN_TYPE_HANDLE, 1, &handle, "CUDNN_ATTR_EXECUTION_PLAN_HANDLE");
                plan->setDescriptor(abi::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, raw[static_cast<std::size_t>(i)], "CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG");

                if (dnn_dyn::p_cudnnBackendFinalize(plan->get()) != abi::CUDNN_STATUS_SUCCESS)
                    continue;

                std::int64_t workspace = 0;
                std::int64_t elements = 0;
                checkCudnn(
                    dnn_dyn::p_cudnnBackendGetAttribute(
                        plan->get(),
                        abi::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                        abi::CUDNN_TYPE_INT64,
                        1,
                        &elements,
                        &workspace),
                    "CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE");

                result->workspace_bytes = workspace;
                result->plan = std::move(plan);
                result->descriptors = std::move(owned_);
                result->descriptors.push_back(std::move(heur));
                for (auto& cfg : configs)
                    result->descriptors.push_back(std::move(cfg));
                return result;
            }
        }

        throw DnnUnsupportedError(std::string("CLAP_DNN cuDNN: no cuDNN engine supports the ") + what +
                                  " operation graph for this shape/datatype on this GPU");
    }

private:
    BackendDescriptor& make(abi::cudnnBackendDescriptorType_t type)
    {
        owned_.emplace_back(new BackendDescriptor(type));
        return *owned_.back();
    }

    abi::cudnnBackendDescriptor_t makeTensor(std::int64_t uid,
                                             abi::cudnnDataType_t type,
                                             const std::vector<std::int64_t>& dims,
                                             const std::vector<std::int64_t>& strides,
                                             bool is_virtual,
                                             bool by_value)
    {
        const std::int64_t alignment = 16;
        const bool virtual_flag = is_virtual;
        const bool by_value_flag = by_value;

        BackendDescriptor& t = make(abi::CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        t.set(abi::CUDNN_ATTR_TENSOR_DATA_TYPE, abi::CUDNN_TYPE_DATA_TYPE, 1, &type, "CUDNN_ATTR_TENSOR_DATA_TYPE");
        t.set(abi::CUDNN_ATTR_TENSOR_DIMENSIONS, abi::CUDNN_TYPE_INT64, static_cast<std::int64_t>(dims.size()), dims.data(), "CUDNN_ATTR_TENSOR_DIMENSIONS");
        t.set(abi::CUDNN_ATTR_TENSOR_STRIDES, abi::CUDNN_TYPE_INT64, static_cast<std::int64_t>(strides.size()), strides.data(), "CUDNN_ATTR_TENSOR_STRIDES");
        t.set(abi::CUDNN_ATTR_TENSOR_UNIQUE_ID, abi::CUDNN_TYPE_INT64, 1, &uid, "CUDNN_ATTR_TENSOR_UNIQUE_ID");
        t.set(abi::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, abi::CUDNN_TYPE_INT64, 1, &alignment, "CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT");
        if (virtual_flag)
            t.set(abi::CUDNN_ATTR_TENSOR_IS_VIRTUAL, abi::CUDNN_TYPE_BOOLEAN, 1, &virtual_flag, "CUDNN_ATTR_TENSOR_IS_VIRTUAL");
        if (by_value_flag)
            t.set(abi::CUDNN_ATTR_TENSOR_IS_BY_VALUE, abi::CUDNN_TYPE_BOOLEAN, 1, &by_value_flag, "CUDNN_ATTR_TENSOR_IS_BY_VALUE");
        t.finalize("tensor descriptor finalize");
        return t.get();
    }

    std::vector<std::unique_ptr<BackendDescriptor>> owned_;
    std::vector<abi::cudnnBackendDescriptor_t> ops_;
    std::int64_t next_virtual_uid_ = 1000;
};

void executePlan(abi::cudnnHandle_t handle,
                 GraphPlan& plan,
                 const std::vector<std::int64_t>& uids,
                 const std::vector<void*>& pointers,
                 StreamScratch& scratch)
{
    void* workspace = scratch.get(static_cast<std::size_t>(plan.workspace_bytes));

    BackendDescriptor pack(abi::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    pack.set(abi::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, abi::CUDNN_TYPE_INT64, static_cast<std::int64_t>(uids.size()), uids.data(), "CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS");
    pack.set(abi::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, abi::CUDNN_TYPE_VOID_PTR, static_cast<std::int64_t>(pointers.size()), pointers.data(), "CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS");
    pack.set(abi::CUDNN_ATTR_VARIANT_PACK_WORKSPACE, abi::CUDNN_TYPE_VOID_PTR, 1, &workspace, "CUDNN_ATTR_VARIANT_PACK_WORKSPACE");
    pack.finalize("variant pack finalize");

    checkCudnn(
        dnn_dyn::p_cudnnBackendExecute(handle, plan.plan->get(), pack.get()),
        "cudnnBackendExecute");
}

void appendKey(std::ostringstream& key, const TensorDesc& d)
{
    key << static_cast<int>(d.type) << ':';
    for (auto v : d.dims) key << v << ',';
    key << '/';
    for (auto v : d.effectiveStrides()) key << v << ',';
    key << ';';
}

}

struct CuDnnBackend::NativeState {
    std::recursive_mutex mutex;
    std::map<int, abi::cudnnHandle_t> handles;
    std::map<std::string, std::unique_ptr<cudnn_detail::GraphPlan>> plans;

    ~NativeState()
    {
        plans.clear();
        for (auto& h : handles) {
            if (h.second && dnn_dyn::p_cudnnDestroy)
                dnn_dyn::p_cudnnDestroy(h.second);
        }
    }
};

CuDnnBackend::CuDnnBackend()
{
    requireCuda();
    native_.reset(new NativeState());
}

CuDnnBackend::~CuDnnBackend() = default;

const char* CuDnnBackend::name() const noexcept
{
    return "cuDNN/NVIDIA GPU";
}

ExecutionBackend CuDnnBackend::executionBackend() const noexcept
{
    return ExecutionBackend::CUDA;
}

void CuDnnBackend::convolutionForward(const TensorDesc& xd,
                                      const float* x,
                                      const TensorDesc& wd,
                                      const float* w,
                                      const ConvolutionDesc& c,
                                      const TensorDesc& yd,
                                      float* y)
{
    std::cerr << "[MY CLAP] CUDA convolution called from application\n";
	requireCuda();

    DeviceBuffer dx(legacyBytes(xd));
    DeviceBuffer dw(legacyBytes(wd));
    DeviceBuffer dy(legacyBytes(yd));

    dx.copyFromHost(x, dx.size());
    dw.copyFromHost(w, dw.size());

    std::cout << "[CLAP_DNN][CUDA] convolutionForward -> cudnnConvolutionForward\n";

    convolutionForward(xd, dx.get(), wd, dw.get(), c, yd, dy.get(), legacyContext());

    synchronizeLegacy();
    dy.copyToHost(y, dy.size());
}

void CuDnnBackend::convolutionBackwardData(const TensorDesc& dyd,
                                           const float* dy,
                                           const TensorDesc& wd,
                                           const float* w,
                                           const ConvolutionDesc& c,
                                           const TensorDesc& dxd,
                                           float* dx)
{
    requireCuda();

    DeviceBuffer d_dy(legacyBytes(dyd));
    DeviceBuffer d_w(legacyBytes(wd));
    DeviceBuffer d_dx(legacyBytes(dxd));

    d_dy.copyFromHost(dy, d_dy.size());
    d_w.copyFromHost(w, d_w.size());

    std::cout << "[CLAP_DNN][CUDA] convolutionBackwardData -> cudnnConvolutionBackwardData\n";

    convolutionBackwardData(dyd, d_dy.get(), wd, d_w.get(), c, dxd, d_dx.get(), legacyContext());

    synchronizeLegacy();
    d_dx.copyToHost(dx, d_dx.size());
}

void CuDnnBackend::convolutionBackwardWeights(const TensorDesc& xd,
                                              const float* x,
                                              const TensorDesc& dyd,
                                              const float* dy,
                                              const ConvolutionDesc& c,
                                              const TensorDesc& dwd,
                                              float* dw)
{
    requireCuda();

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_dy(legacyBytes(dyd));
    DeviceBuffer d_dw(legacyBytes(dwd));

    d_x.copyFromHost(x, d_x.size());
    d_dy.copyFromHost(dy, d_dy.size());

    std::cout << "[CLAP_DNN][CUDA] convolutionBackwardWeights -> cudnnConvolutionBackwardFilter\n";

    convolutionBackwardWeights(xd, d_x.get(), dyd, d_dy.get(), c, dwd, d_dw.get(), legacyContext());

    synchronizeLegacy();
    d_dw.copyToHost(dw, d_dw.size());
}

void CuDnnBackend::convolutionBackwardBias(const TensorDesc& dyd,
                                           const float* dy,
                                           const TensorDesc& dbd,
                                           float* db)
{
    requireCuda();

    DeviceBuffer d_dy(legacyBytes(dyd));
    DeviceBuffer d_db(legacyBytes(dbd));

    d_dy.copyFromHost(dy, d_dy.size());

    std::cout << "[CLAP_DNN][CUDA] convolutionBackwardBias -> cudnnConvolutionBackwardBias\n";

    convolutionBackwardBias(dyd, d_dy.get(), dbd, d_db.get(), legacyContext());

    synchronizeLegacy();
    d_db.copyToHost(db, d_db.size());
}

void CuDnnBackend::convolutionTransposeForward(const TensorDesc& xd,
                                               const float* x,
                                               const TensorDesc& wd,
                                               const float* w,
                                               const ConvolutionDesc& c,
                                               const TensorDesc& yd,
                                               float* y)
{
    requireCuda();

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_w(legacyBytes(wd));
    DeviceBuffer d_y(legacyBytes(yd));

    d_x.copyFromHost(x, d_x.size());
    d_w.copyFromHost(w, d_w.size());

    std::cout << "[CLAP_DNN][CUDA] convolutionTransposeForward -> cudnnConvolutionBackwardData\n";

    convolutionTransposeForward(xd, d_x.get(), wd, d_w.get(), c, yd, d_y.get(), legacyContext());

    synchronizeLegacy();
    d_y.copyToHost(y, d_y.size());
}

void CuDnnBackend::fusedConvolutionBiasActivation(const TensorDesc& xd,
                                                  const float* x,
                                                  const TensorDesc& wd,
                                                  const float* w,
                                                  const float* bias,
                                                  const ConvolutionDesc& c,
                                                  const ActivationDesc& a,
                                                  const TensorDesc& yd,
                                                  float* y)
{
    requireCuda();

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_w(legacyBytes(wd));
    DeviceBuffer d_bias(channelCount(yd) * sizeof(float));
    DeviceBuffer d_y(legacyBytes(yd));

    d_x.copyFromHost(x, d_x.size());
    d_w.copyFromHost(w, d_w.size());
    d_bias.copyFromHost(bias, d_bias.size());

    std::cout << "[CLAP_DNN][CUDA] fusedConvolutionBiasActivation -> cudnnConvolutionBiasActivationForward\n";

    fusedConvolutionBiasActivation(xd, d_x.get(), wd, d_w.get(), d_bias.get(), c, a, yd, d_y.get(), legacyContext());

    synchronizeLegacy();
    d_y.copyToHost(y, d_y.size());
}

void CuDnnBackend::activationForward(const ActivationDesc& a,
                                     const TensorDesc& xd,
                                     const float* x,
                                     const TensorDesc& yd,
                                     float* y)
{
    requireCuda();

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_y(legacyBytes(yd));
    d_x.copyFromHost(x, d_x.size());

    std::cout << "[CLAP_DNN][CUDA] activationForward -> cudnnActivationForward\n";

    activationForward(a, xd, d_x.get(), yd, d_y.get(), legacyContext());

    synchronizeLegacy();
    d_y.copyToHost(y, d_y.size());
}

void CuDnnBackend::activationBackward(const ActivationDesc& a,
                                      const TensorDesc& xd,
                                      const float* x,
                                      const TensorDesc& dyd,
                                      const float* dy,
                                      const TensorDesc& dxd,
                                      float* dx)
{
    requireCuda();

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_dy(legacyBytes(dyd));
    DeviceBuffer d_dx(legacyBytes(dxd));

    d_x.copyFromHost(x, d_x.size());
    d_dy.copyFromHost(dy, d_dy.size());

    std::cout << "[CLAP_DNN][CUDA] activationBackward -> cudnnActivationBackward\n";

    activationBackward(a, xd, d_x.get(), dyd, d_dy.get(), dxd, d_dx.get(), legacyContext());

    synchronizeLegacy();
    d_dx.copyToHost(dx, d_dx.size());
}

void CuDnnBackend::poolingForward(const PoolingDesc& p,
                                  const TensorDesc& xd,
                                  const float* x,
                                  const TensorDesc& yd,
                                  float* y)
{
    requireCuda();

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_y(legacyBytes(yd));
    d_x.copyFromHost(x, d_x.size());

    std::cout << "[CLAP_DNN][CUDA] poolingForward -> cudnnPoolingForward\n";

    poolingForward(p, xd, d_x.get(), yd, d_y.get(), legacyContext());

    synchronizeLegacy();
    d_y.copyToHost(y, d_y.size());
}

void CuDnnBackend::poolingBackward(const PoolingDesc& p,
                                   const TensorDesc& xd,
                                   const float* x,
                                   const TensorDesc& yd,
                                   const float* y,
                                   const TensorDesc& dyd,
                                   const float* dy,
                                   const TensorDesc& dxd,
                                   float* dx)
{
    requireCuda();

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_y(legacyBytes(yd));
    DeviceBuffer d_dy(legacyBytes(dyd));
    DeviceBuffer d_dx(legacyBytes(dxd));

    d_x.copyFromHost(x, d_x.size());
    d_y.copyFromHost(y, d_y.size());
    d_dy.copyFromHost(dy, d_dy.size());

    std::cout << "[CLAP_DNN][CUDA] poolingBackward -> cudnnPoolingBackward\n";

    poolingBackward(p, xd, d_x.get(), yd, d_y.get(), dyd, d_dy.get(), dxd, d_dx.get(), legacyContext());

    synchronizeLegacy();
    d_dx.copyToHost(dx, d_dx.size());
}

void CuDnnBackend::softmaxForward(const TensorDesc& xd,
                                  const float* x,
                                  const TensorDesc& yd,
                                  float* y)
{
    requireCuda();

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_y(legacyBytes(yd));
    d_x.copyFromHost(x, d_x.size());

    std::cout << "[CLAP_DNN][CUDA] softmaxForward -> cudnnSoftmaxForward\n";

    SoftmaxDesc softmax;
    softmax.axis = 1;
    softmaxForward(softmax, xd, d_x.get(), yd, d_y.get(), legacyContext());

    synchronizeLegacy();
    d_y.copyToHost(y, d_y.size());
}

void CuDnnBackend::softmaxBackward(const TensorDesc& yd,
                                   const float* y,
                                   const TensorDesc& dyd,
                                   const float* dy,
                                   const TensorDesc& dxd,
                                   float* dx)
{
    requireCuda();

    DeviceBuffer d_y(legacyBytes(yd));
    DeviceBuffer d_dy(legacyBytes(dyd));
    DeviceBuffer d_dx(legacyBytes(dxd));

    d_y.copyFromHost(y, d_y.size());
    d_dy.copyFromHost(dy, d_dy.size());

    std::cout << "[CLAP_DNN][CUDA] softmaxBackward -> cudnnSoftmaxBackward\n";

    SoftmaxDesc softmax;
    softmax.axis = 1;
    softmaxBackward(softmax, yd, d_y.get(), dyd, d_dy.get(), dxd, d_dx.get(), legacyContext());

    synchronizeLegacy();
    d_dx.copyToHost(dx, d_dx.size());
}

void CuDnnBackend::batchNormForwardTraining(const BatchNormDesc& bn,
                                            const TensorDesc& xd,
                                            const float* x,
                                            const float* scale,
                                            const float* bias,
                                            float* running_mean,
                                            float* running_variance,
                                            float* saved_mean,
                                            float* saved_variance,
                                            const TensorDesc& yd,
                                            float* y)
{
    requireCuda();

    const std::size_t c = channelCount(xd);

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_y(legacyBytes(yd));
    DeviceBuffer d_scale(c * sizeof(float));
    DeviceBuffer d_bias(c * sizeof(float));
    DeviceBuffer d_rm(c * sizeof(float));
    DeviceBuffer d_rv(c * sizeof(float));
    DeviceBuffer d_sm(c * sizeof(float));
    DeviceBuffer d_siv(c * sizeof(float));

    d_x.copyFromHost(x, d_x.size());
    d_scale.copyFromHost(scale, d_scale.size());
    d_bias.copyFromHost(bias, d_bias.size());

    if (running_mean) d_rm.copyFromHost(running_mean, d_rm.size()); else d_rm.zero();
    if (running_variance) d_rv.copyFromHost(running_variance, d_rv.size()); else d_rv.zero();

    std::cout << "[CLAP_DNN][CUDA] batchNormForwardTraining -> cudnnBatchNormalizationForwardTraining\n";

    batchNormForwardTraining(
        bn,
        xd, d_x.get(),
        d_scale.get(), d_bias.get(),
        d_rm.get(), d_rv.get(),
        d_sm.get(), d_siv.get(),
        yd, d_y.get(),
        legacyContext());

    synchronizeLegacy();

    d_y.copyToHost(y, d_y.size());
    if (running_mean) d_rm.copyToHost(running_mean, d_rm.size());
    if (running_variance) d_rv.copyToHost(running_variance, d_rv.size());
    if (saved_mean) d_sm.copyToHost(saved_mean, d_sm.size());

    if (saved_variance) {
        std::vector<float> inv(c);
        d_siv.copyToHost(inv.data(), d_siv.size());
        for (std::size_t i = 0; i < c; ++i)
            saved_variance[i] = std::max(0.0f, 1.0f / (inv[i] * inv[i]) - static_cast<float>(bn.epsilon));
    }
}

void CuDnnBackend::batchNormForwardInference(const BatchNormDesc& bn,
                                             const TensorDesc& xd,
                                             const float* x,
                                             const float* scale,
                                             const float* bias,
                                             const float* mean,
                                             const float* variance,
                                             const TensorDesc& yd,
                                             float* y)
{
    requireCuda();

    const std::size_t c = channelCount(xd);

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_y(legacyBytes(yd));
    DeviceBuffer d_scale(c * sizeof(float));
    DeviceBuffer d_bias(c * sizeof(float));
    DeviceBuffer d_mean(c * sizeof(float));
    DeviceBuffer d_var(c * sizeof(float));

    d_x.copyFromHost(x, d_x.size());
    d_scale.copyFromHost(scale, d_scale.size());
    d_bias.copyFromHost(bias, d_bias.size());
    d_mean.copyFromHost(mean, d_mean.size());
    d_var.copyFromHost(variance, d_var.size());

    std::cout << "[CLAP_DNN][CUDA] batchNormForwardInference -> cudnnBatchNormalizationForwardInference\n";

    batchNormForwardInference(
        bn,
        xd, d_x.get(),
        d_scale.get(), d_bias.get(),
        d_mean.get(), d_var.get(),
        yd, d_y.get(),
        legacyContext());

    synchronizeLegacy();
    d_y.copyToHost(y, d_y.size());
}

void CuDnnBackend::batchNormBackward(const BatchNormDesc& bn,
                                     const TensorDesc& xd,
                                     const float* x,
                                     const TensorDesc& dyd,
                                     const float* dy,
                                     const float* scale,
                                     const float* saved_mean,
                                     const float* saved_variance,
                                     const TensorDesc& dxd,
                                     float* dx,
                                     float* dscale,
                                     float* dbias)
{
    requireCuda();

    const std::size_t c = channelCount(xd);

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_dy(legacyBytes(dyd));
    DeviceBuffer d_dx(legacyBytes(dxd));
    DeviceBuffer d_scale(c * sizeof(float));
    DeviceBuffer d_mean(c * sizeof(float));
    DeviceBuffer d_inv(c * sizeof(float));
    DeviceBuffer d_dscale(c * sizeof(float));
    DeviceBuffer d_dbias(c * sizeof(float));

    d_x.copyFromHost(x, d_x.size());
    d_dy.copyFromHost(dy, d_dy.size());
    d_scale.copyFromHost(scale, d_scale.size());
    d_mean.copyFromHost(saved_mean, d_mean.size());

    std::vector<float> inv(c);
    for (std::size_t i = 0; i < c; ++i)
        inv[i] = 1.0f / std::sqrt(saved_variance[i] + static_cast<float>(bn.epsilon));
    d_inv.copyFromHost(inv.data(), d_inv.size());

    std::cout << "[CLAP_DNN][CUDA] batchNormBackward -> cudnnBatchNormalizationBackward\n";

    batchNormBackward(
        bn,
        xd, d_x.get(),
        dyd, d_dy.get(),
        d_scale.get(),
        d_mean.get(), d_inv.get(),
        dxd, d_dx.get(),
        d_dscale.get(), d_dbias.get(),
        legacyContext());

    synchronizeLegacy();
    d_dx.copyToHost(dx, d_dx.size());
    d_dscale.copyToHost(dscale, d_dscale.size());
    d_dbias.copyToHost(dbias, d_dbias.size());
}

void CuDnnBackend::layerNormForward(const LayerNormDesc& ln,
                                    const TensorDesc& xd,
                                    const float* x,
                                    const float* scale,
                                    const float* bias,
                                    const TensorDesc& yd,
                                    float* y,
                                    float* saved_mean,
                                    float* saved_rstd)
{
    requireCuda();

    if (xd.elements() != yd.elements() || xd.dims.empty())
        throw std::runtime_error("layerNormForward requires matching non-empty tensors");

    const std::size_t inner = static_cast<std::size_t>(xd.dims.back());
    const std::size_t outer = xd.elements() / inner;
    const TensorDesc scale_desc({static_cast<std::int64_t>(inner)});

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_scale(inner * sizeof(float));
    DeviceBuffer d_bias(inner * sizeof(float));
    DeviceBuffer d_y(legacyBytes(yd));
    DeviceBuffer d_mean(outer * sizeof(float));
    DeviceBuffer d_rstd(outer * sizeof(float));

    d_x.copyFromHost(x, d_x.size());
    d_scale.copyFromHost(scale, d_scale.size());
    d_bias.copyFromHost(bias, d_bias.size());

    std::cout << "[CLAP_DNN][CUDA] layerNormForward -> runtime CUDA GPU kernel\n";

    layerNormForward(
        ln,
        xd, d_x.get(),
        scale_desc, d_scale.get(), d_bias.get(),
        yd, d_y.get(),
        d_mean.get(), d_rstd.get(),
        legacyContext());

    synchronizeLegacy();
    d_y.copyToHost(y, d_y.size());
    if (saved_mean) d_mean.copyToHost(saved_mean, d_mean.size());
    if (saved_rstd) d_rstd.copyToHost(saved_rstd, d_rstd.size());
}

void CuDnnBackend::layerNormBackward(const LayerNormDesc& ln,
                                     const TensorDesc& xd,
                                     const float* x,
                                     const TensorDesc& dyd,
                                     const float* dy,
                                     const float* scale,
                                     const float* saved_mean,
                                     const float* saved_rstd,
                                     const TensorDesc& dxd,
                                     float* dx,
                                     float* dscale,
                                     float* dbias)
{
    requireCuda();

    if (xd.elements() != dyd.elements() || xd.elements() != dxd.elements() || xd.dims.empty())
        throw std::runtime_error("layerNormBackward requires matching non-empty tensors");

    const std::size_t inner = static_cast<std::size_t>(xd.dims.back());
    const std::size_t outer = xd.elements() / inner;
    const TensorDesc scale_desc({static_cast<std::int64_t>(inner)});

    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_dy(legacyBytes(dyd));
    DeviceBuffer d_scale(inner * sizeof(float));
    DeviceBuffer d_mean(outer * sizeof(float));
    DeviceBuffer d_rstd(outer * sizeof(float));
    DeviceBuffer d_dx(legacyBytes(dxd));
    DeviceBuffer d_dscale(inner * sizeof(float));
    DeviceBuffer d_dbias(inner * sizeof(float));

    d_x.copyFromHost(x, d_x.size());
    d_dy.copyFromHost(dy, d_dy.size());
    d_scale.copyFromHost(scale, d_scale.size());
    d_mean.copyFromHost(saved_mean, d_mean.size());
    d_rstd.copyFromHost(saved_rstd, d_rstd.size());

    std::cout << "[CLAP_DNN][CUDA] layerNormBackward -> runtime CUDA GPU kernel\n";

    layerNormBackward(
        ln,
        xd, d_x.get(),
        dyd, d_dy.get(),
        scale_desc, d_scale.get(),
        d_mean.get(), d_rstd.get(),
        dxd, d_dx.get(),
        d_dscale.get(), d_dbias.get(),
        legacyContext());

    synchronizeLegacy();
    d_dx.copyToHost(dx, d_dx.size());
    d_dscale.copyToHost(dscale, d_dscale.size());
    d_dbias.copyToHost(dbias, d_dbias.size());
}

void CuDnnBackend::dropoutForward(const DropoutDesc& dropout,
                                  const TensorDesc& xd,
                                  const float* x,
                                  const TensorDesc& yd,
                                  float* y,
                                  std::uint8_t* mask)
{
    requireCuda();

    const std::size_t reserve_size = dropoutReserveSpaceSize(xd, legacyContext());

    DeviceBuffer reserve(reserve_size);
    DeviceBuffer d_x(legacyBytes(xd));
    DeviceBuffer d_y(legacyBytes(yd));

    d_x.copyFromHost(x, d_x.size());

    std::cout << "[CLAP_DNN][CUDA] dropoutForward -> cudnnDropoutForward\n";

    dropoutForward(dropout, xd, d_x.get(), yd, d_y.get(), reserve.get(), reserve.size(), legacyContext());

    synchronizeLegacy();
    d_y.copyToHost(y, d_y.size());

    dropout_reserve_.resize(reserve_size);
    reserve.copyToHost(dropout_reserve_.data(), reserve_size);
    dropout_probability_ = dropout.probability;
    dropout_seed_ = dropout.seed;
    dropout_shape_ = xd.dims;

    if (mask) {
        for (std::size_t i = 0; i < xd.elements(); ++i)
            mask[i] = static_cast<std::uint8_t>(y[i] != 0.0f || x[i] == 0.0f);
    }
}

void CuDnnBackend::dropoutBackward(const DropoutDesc& dropout,
                                   const TensorDesc& dyd,
                                   const float* dy,
                                   const std::uint8_t*,
                                   const TensorDesc& dxd,
                                   float* dx)
{
    requireCuda();

    if (dropout_reserve_.empty() || dropout_shape_ != dyd.dims)
        throw std::runtime_error("dropoutBackward requires dropoutForward on the same backend and tensor shape first");

    DeviceBuffer reserve(dropout_reserve_.size());
    DeviceBuffer d_dy(legacyBytes(dyd));
    DeviceBuffer d_dx(legacyBytes(dxd));

    reserve.copyFromHost(dropout_reserve_.data(), dropout_reserve_.size());
    d_dy.copyFromHost(dy, d_dy.size());

    std::cout << "[CLAP_DNN][CUDA] dropoutBackward -> cudnnDropoutBackward\n";

    DropoutDesc saved = dropout;
    saved.probability = dropout_probability_;
    saved.seed = dropout_seed_;
    dropoutBackward(saved, dyd, d_dy.get(), reserve.get(), reserve.size(), dxd, d_dx.get(), legacyContext());

    synchronizeLegacy();
    d_dx.copyToHost(dx, d_dx.size());
}

void CuDnnBackend::lrnForward(const LrnDesc& lrn, const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y)
{
    requireCuda();

    DeviceBuffer X(legacyBytes(xd));
    DeviceBuffer Y(legacyBytes(yd));
    X.copyFromHost(x, X.size());

    lrnForward(lrn, xd, X.get(), yd, Y.get(), legacyContext());

    synchronizeLegacy();
    Y.copyToHost(y, Y.size());
}

void CuDnnBackend::lrnBackward(const LrnDesc& lrn, const TensorDesc& xd, const float* x, const TensorDesc& yd, const float* y, const TensorDesc& dyd, const float* dy, const TensorDesc& dxd, float* dx)
{
    requireCuda();

    DeviceBuffer X(legacyBytes(xd));
    DeviceBuffer Y(legacyBytes(yd));
    DeviceBuffer DY(legacyBytes(dyd));
    DeviceBuffer DX(legacyBytes(dxd));
    X.copyFromHost(x, X.size());
    Y.copyFromHost(y, Y.size());
    DY.copyFromHost(dy, DY.size());

    lrnBackward(lrn, xd, X.get(), yd, Y.get(), dyd, DY.get(), dxd, DX.get(), legacyContext());

    synchronizeLegacy();
    DX.copyToHost(dx, DX.size());
}

namespace {

template <class Op>
void stageBinary(const TensorDesc& ad, const float* a,
                 const TensorDesc& bd, const float* b,
                 const TensorDesc& yd, float* y,
                 Op op)
{
    requireCuda();

    DeviceBuffer A0(legacyBytes(ad));
    DeviceBuffer B0(legacyBytes(bd));
    DeviceBuffer Y0(legacyBytes(yd));
    A0.copyFromHost(a, A0.size());
    B0.copyFromHost(b, B0.size());

    op(A0.get(), B0.get(), Y0.get());

    synchronizeLegacy();
    Y0.copyToHost(y, Y0.size());
}

template <class Op>
void stageUnary(const TensorDesc& xd, const float* x,
                const TensorDesc& yd, float* y,
                Op op)
{
    requireCuda();

    DeviceBuffer X0(legacyBytes(xd));
    DeviceBuffer Y0(legacyBytes(yd));
    X0.copyFromHost(x, X0.size());

    op(X0.get(), Y0.get());

    synchronizeLegacy();
    Y0.copyToHost(y, Y0.size());
}

}

void CuDnnBackend::tensorAdd(const TensorDesc& ad, const float* a, const TensorDesc& bd, const float* b, const TensorDesc& yd, float* y)
{
    stageBinary(ad, a, bd, b, yd, y, [&](void* A, void* B, void* Y) {
        tensorAdd(ad, A, bd, B, yd, Y, legacyContext());
    });
}

void CuDnnBackend::tensorMultiply(const TensorDesc& ad, const float* a, const TensorDesc& bd, const float* b, const TensorDesc& yd, float* y)
{
    stageBinary(ad, a, bd, b, yd, y, [&](void* A, void* B, void* Y) {
        tensorMultiply(ad, A, bd, B, yd, Y, legacyContext());
    });
}

void CuDnnBackend::tensorMin(const TensorDesc& ad, const float* a, const TensorDesc& bd, const float* b, const TensorDesc& yd, float* y)
{
    stageBinary(ad, a, bd, b, yd, y, [&](void* A, void* B, void* Y) {
        tensorMin(ad, A, bd, B, yd, Y, legacyContext());
    });
}

void CuDnnBackend::tensorMax(const TensorDesc& ad, const float* a, const TensorDesc& bd, const float* b, const TensorDesc& yd, float* y)
{
    stageBinary(ad, a, bd, b, yd, y, [&](void* A, void* B, void* Y) {
        tensorMax(ad, A, bd, B, yd, Y, legacyContext());
    });
}

void CuDnnBackend::tensorReduceSum(const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y)
{
    stageUnary(xd, x, yd, y, [&](void* X, void* Y) {
        tensorReduceSum(xd, X, yd, Y, legacyContext());
    });
}

void CuDnnBackend::tensorReduceProduct(const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y)
{
    stageUnary(xd, x, yd, y, [&](void* X, void* Y) {
        tensorReduceProduct(xd, X, yd, Y, legacyContext());
    });
}

void CuDnnBackend::convolutionForward(const TensorDesc& xd,
                                      const void* x,
                                      const TensorDesc& wd,
                                      const void* w,
                                      const ConvolutionDesc& c,
                                      const TensorDesc& yd,
                                      void* y,
                                      const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "convolutionForward");
    xd.require4d("convolutionForward");
    yd.require4d("convolutionForward");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor x_desc(xd);
    FilterDescriptor w_desc(wd);
    TensorDescriptor y_desc(yd);
    ConvolutionDescriptor conv_desc(c);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dnn_contract::trace("CUDA", "convolutionForward", "cudnnConvolutionForward");

    checkCudnn(
        dnn_dyn::p_cudnnConvolutionForward(
            handle, &alpha, x_desc, x, w_desc, w, conv_desc,
            abi::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            nullptr, 0, &beta, y_desc, y),
        "cudnnConvolutionForward");
}

void CuDnnBackend::convolutionBackwardData(const TensorDesc& dyd,
                                           const void* dy,
                                           const TensorDesc& wd,
                                           const void* w,
                                           const ConvolutionDesc& c,
                                           const TensorDesc& dxd,
                                           void* dx,
                                           const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "convolutionBackwardData");
    dyd.require4d("convolutionBackwardData");
    dxd.require4d("convolutionBackwardData");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor dy_desc(dyd);
    FilterDescriptor w_desc(wd);
    TensorDescriptor dx_desc(dxd);
    ConvolutionDescriptor conv_desc(c);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dnn_contract::trace("CUDA", "convolutionBackwardData", "cudnnConvolutionBackwardData");

    checkCudnn(
        dnn_dyn::p_cudnnConvolutionBackwardData(
            handle, &alpha, w_desc, w, dy_desc, dy, conv_desc,
            abi::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            nullptr, 0, &beta, dx_desc, dx),
        "cudnnConvolutionBackwardData");
}

void CuDnnBackend::convolutionBackwardWeights(const TensorDesc& xd,
                                              const void* x,
                                              const TensorDesc& dyd,
                                              const void* dy,
                                              const ConvolutionDesc& c,
                                              const TensorDesc& dwd,
                                              void* dw,
                                              const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "convolutionBackwardWeights");
    xd.require4d("convolutionBackwardWeights");
    dyd.require4d("convolutionBackwardWeights");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor x_desc(xd);
    TensorDescriptor dy_desc(dyd);
    FilterDescriptor dw_desc(dwd);
    ConvolutionDescriptor conv_desc(c);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dnn_contract::trace("CUDA", "convolutionBackwardWeights", "cudnnConvolutionBackwardFilter");

    checkCudnn(
        dnn_dyn::p_cudnnConvolutionBackwardFilter(
            handle, &alpha, x_desc, x, dy_desc, dy, conv_desc,
            abi::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            nullptr, 0, &beta, dw_desc, dw),
        "cudnnConvolutionBackwardFilter");
}

void CuDnnBackend::convolutionBackwardBias(const TensorDesc& dyd,
                                           const void* dy,
                                           const TensorDesc& dbd,
                                           void* db,
                                           const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "convolutionBackwardBias");
    dyd.require4d("convolutionBackwardBias");

    if (dbd.elements() != static_cast<std::size_t>(dyd.dims[1]))
        throw std::runtime_error("convolutionBackwardBias: db must hold one value per channel");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor dy_desc(dyd);
    TensorDescriptor db_desc(dbd.type, 1, static_cast<int>(dyd.dims[1]), 1, 1);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dnn_contract::trace("CUDA", "convolutionBackwardBias", "cudnnConvolutionBackwardBias");

    checkCudnn(
        dnn_dyn::p_cudnnConvolutionBackwardBias(
            handle, &alpha, dy_desc, dy, &beta, db_desc, db),
        "cudnnConvolutionBackwardBias");
}

void CuDnnBackend::convolutionTransposeForward(const TensorDesc& xd,
                                               const void* x,
                                               const TensorDesc& wd,
                                               const void* w,
                                               const ConvolutionDesc& c,
                                               const TensorDesc& yd,
                                               void* y,
                                               const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "convolutionTransposeForward");
    xd.require4d("convolutionTransposeForward");
    yd.require4d("convolutionTransposeForward");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor x_desc(xd);
    FilterDescriptor w_desc(wd);
    TensorDescriptor y_desc(yd);
    ConvolutionDescriptor conv_desc(c);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dnn_contract::trace("CUDA", "convolutionTransposeForward", "cudnnConvolutionBackwardData");

    checkCudnn(
        dnn_dyn::p_cudnnConvolutionBackwardData(
            handle, &alpha, w_desc, w, x_desc, x, conv_desc,
            abi::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            nullptr, 0, &beta, y_desc, y),
        "cudnnConvolutionBackwardData(ConvTranspose2D)");
}

void CuDnnBackend::fusedConvolutionBiasActivation(const TensorDesc& xd,
                                                  const void* x,
                                                  const TensorDesc& wd,
                                                  const void* w,
                                                  const void* bias,
                                                  const ConvolutionDesc& c,
                                                  const ActivationDesc& a,
                                                  const TensorDesc& yd,
                                                  void* y,
                                                  const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "fusedConvolutionBiasActivation");
    xd.require4d("fusedConvolutionBiasActivation");
    yd.require4d("fusedConvolutionBiasActivation");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor x_desc(xd);
    FilterDescriptor w_desc(wd);
    TensorDescriptor y_desc(yd);
    TensorDescriptor bias_desc(yd.type, 1, static_cast<int>(yd.dims[1]), 1, 1);
    ConvolutionDescriptor conv_desc(c);
    ActivationDescriptor activation_desc(a);

    const float alpha1 = 1.0f;
    const float alpha2 = 0.0f;

    dnn_contract::trace("CUDA", "fusedConvolutionBiasActivation", "cudnnConvolutionBiasActivationForward");

    checkCudnn(
        dnn_dyn::p_cudnnConvolutionBiasActivationForward(
            handle,
            &alpha1,
            x_desc, x,
            w_desc, w,
            conv_desc,
            abi::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            nullptr, 0,
            &alpha2,
            y_desc, y,
            bias_desc, bias,
            activation_desc,
            y_desc, y),
        "cudnnConvolutionBiasActivationForward");
}

void CuDnnBackend::activationForward(const ActivationDesc& a,
                                     const TensorDesc& xd,
                                     const void* x,
                                     const TensorDesc& yd,
                                     void* y,
                                     const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "activationForward");
    dnn_contract::requireSameShape(xd, yd, "activationForward");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor x_desc(xd);
    TensorDescriptor y_desc(yd);
    ActivationDescriptor activation_desc(a);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dnn_contract::trace("CUDA", "activationForward", "cudnnActivationForward");

    checkCudnn(
        dnn_dyn::p_cudnnActivationForward(
            handle, activation_desc, &alpha, x_desc, x,
            &beta, y_desc, y),
        "cudnnActivationForward");
}

void CuDnnBackend::activationBackward(const ActivationDesc& a,
                                      const TensorDesc& xd,
                                      const void* x,
                                      const TensorDesc& dyd,
                                      const void* dy,
                                      const TensorDesc& dxd,
                                      void* dx,
                                      const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "activationBackward");
    dnn_contract::requireSameShape(xd, dyd, "activationBackward");
    dnn_contract::requireSameShape(xd, dxd, "activationBackward");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);
    StreamScratch scratch(ctx);

    TensorDescriptor x_desc(xd);
    TensorDescriptor dy_desc(dyd);
    TensorDescriptor dx_desc(dxd);
    ActivationDescriptor activation_desc(a);

    void* y = scratch.get(xd.bytes());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    checkCudnn(
        dnn_dyn::p_cudnnActivationForward(
            handle, activation_desc, &alpha, x_desc, x,
            &beta, x_desc, y),
        "cudnnActivationForward(for backward)");

    dnn_contract::trace("CUDA", "activationBackward", "cudnnActivationBackward");

    checkCudnn(
        dnn_dyn::p_cudnnActivationBackward(
            handle, activation_desc, &alpha,
            x_desc, y, dy_desc, dy, x_desc, x,
            &beta, dx_desc, dx),
        "cudnnActivationBackward");
}

void CuDnnBackend::poolingForward(const PoolingDesc& p,
                                  const TensorDesc& xd,
                                  const void* x,
                                  const TensorDesc& yd,
                                  void* y,
                                  const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "poolingForward");
    xd.require4d("poolingForward");
    yd.require4d("poolingForward");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor x_desc(xd);
    TensorDescriptor y_desc(yd);
    PoolingDescriptor pooling_desc(p);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dnn_contract::trace("CUDA", "poolingForward", "cudnnPoolingForward");

    checkCudnn(
        dnn_dyn::p_cudnnPoolingForward(
            handle, pooling_desc, &alpha, x_desc, x,
            &beta, y_desc, y),
        "cudnnPoolingForward");
}

void CuDnnBackend::poolingBackward(const PoolingDesc& p,
                                   const TensorDesc& xd,
                                   const void* x,
                                   const TensorDesc& yd,
                                   const void* y,
                                   const TensorDesc& dyd,
                                   const void* dy,
                                   const TensorDesc& dxd,
                                   void* dx,
                                   const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "poolingBackward");
    xd.require4d("poolingBackward");
    yd.require4d("poolingBackward");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor x_desc(xd);
    TensorDescriptor y_desc(yd);
    TensorDescriptor dy_desc(dyd);
    TensorDescriptor dx_desc(dxd);
    PoolingDescriptor pooling_desc(p);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dnn_contract::trace("CUDA", "poolingBackward", "cudnnPoolingBackward");

    checkCudnn(
        dnn_dyn::p_cudnnPoolingBackward(
            handle, pooling_desc, &alpha,
            y_desc, y, dy_desc, dy,
            x_desc, x, &beta, dx_desc, dx),
        "cudnnPoolingBackward");
}

namespace {

dnn_contract::AxisView softmaxView(const TensorDesc& d, int axis, const char* where)
{
    dnn_contract::AxisView view;
    if (!dnn_contract::makeAxisView(d, axis, view))
        throw DnnUnsupportedError(std::string("CLAP_DNN cuDNN ") + where +
                                  ": the dimensions around the softmax axis cannot be collapsed into a cuDNN [N, C, H, 1] view");
    return view;
}

void requireSameView(const dnn_contract::AxisView& a, const dnn_contract::AxisView& b, const char* where)
{
    if (a.outer != b.outer || a.axis != b.axis || a.inner != b.inner)
        throw std::runtime_error(std::string(where) + ": tensor shapes do not match");
}

}

void CuDnnBackend::softmaxForward(const SoftmaxDesc& softmax,
                                  const TensorDesc& xd,
                                  const void* x,
                                  const TensorDesc& yd,
                                  void* y,
                                  const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "softmaxForward");

    const int axis = xd.normalizeAxis(softmax.axis, "softmaxForward");
    const auto xv = softmaxView(xd, axis, "softmaxForward");
    const auto yv = softmaxView(yd, yd.normalizeAxis(softmax.axis, "softmaxForward"), "softmaxForward");
    requireSameView(xv, yv, "softmaxForward");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor x_desc(xd.type, xv.outer, xv.axis, xv.inner, xv.outer_stride, xv.axis_stride, xv.inner_stride);
    TensorDescriptor y_desc(yd.type, yv.outer, yv.axis, yv.inner, yv.outer_stride, yv.axis_stride, yv.inner_stride);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dnn_contract::trace("CUDA", softmax.log_softmax ? "logSoftmaxForward" : "softmaxForward", "cudnnSoftmaxForward");

    checkCudnn(
        dnn_dyn::p_cudnnSoftmaxForward(
            handle,
            softmax.log_softmax ? abi::CUDNN_SOFTMAX_LOG : abi::CUDNN_SOFTMAX_ACCURATE,
            abi::CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha, x_desc, x, &beta, y_desc, y),
        "cudnnSoftmaxForward");
}

void CuDnnBackend::softmaxBackward(const SoftmaxDesc& softmax,
                                   const TensorDesc& yd,
                                   const void* y,
                                   const TensorDesc& dyd,
                                   const void* dy,
                                   const TensorDesc& dxd,
                                   void* dx,
                                   const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "softmaxBackward");

    const auto yv = softmaxView(yd, yd.normalizeAxis(softmax.axis, "softmaxBackward"), "softmaxBackward");
    const auto dyv = softmaxView(dyd, dyd.normalizeAxis(softmax.axis, "softmaxBackward"), "softmaxBackward");
    const auto dxv = softmaxView(dxd, dxd.normalizeAxis(softmax.axis, "softmaxBackward"), "softmaxBackward");
    requireSameView(yv, dyv, "softmaxBackward");
    requireSameView(yv, dxv, "softmaxBackward");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor y_desc(yd.type, yv.outer, yv.axis, yv.inner, yv.outer_stride, yv.axis_stride, yv.inner_stride);
    TensorDescriptor dy_desc(dyd.type, dyv.outer, dyv.axis, dyv.inner, dyv.outer_stride, dyv.axis_stride, dyv.inner_stride);
    TensorDescriptor dx_desc(dxd.type, dxv.outer, dxv.axis, dxv.inner, dxv.outer_stride, dxv.axis_stride, dxv.inner_stride);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dnn_contract::trace("CUDA", softmax.log_softmax ? "logSoftmaxBackward" : "softmaxBackward", "cudnnSoftmaxBackward");

    checkCudnn(
        dnn_dyn::p_cudnnSoftmaxBackward(
            handle,
            softmax.log_softmax ? abi::CUDNN_SOFTMAX_LOG : abi::CUDNN_SOFTMAX_ACCURATE,
            abi::CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha, y_desc, y, dy_desc, dy,
            &beta, dx_desc, dx),
        "cudnnSoftmaxBackward");
}

void CuDnnBackend::batchNormForwardTraining(const BatchNormDesc& bn,
                                            const TensorDesc& xd,
                                            const void* x,
                                            const void* scale,
                                            const void* bias,
                                            void* running_mean,
                                            void* running_variance,
                                            void* saved_mean,
                                            void* saved_variance,
                                            const TensorDesc& yd,
                                            void* y,
                                            const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "batchNormForwardTraining");
    xd.require4d("batchNormForwardTraining");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor x_desc(xd);
    TensorDescriptor y_desc(yd);
    TensorDescriptor bn_desc(1, static_cast<int>(xd.dims[1]), 1, 1);

    checkCudnn(
        dnn_dyn::p_cudnnDeriveBNTensorDescriptor(
            bn_desc, x_desc, abi::CUDNN_BATCHNORM_SPATIAL),
        "cudnnDeriveBNTensorDescriptor");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dnn_contract::trace("CUDA", "batchNormForwardTraining", "cudnnBatchNormalizationForwardTraining");

    checkCudnn(
        dnn_dyn::p_cudnnBatchNormalizationForwardTraining(
            handle,
            abi::CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            x_desc, x, y_desc, y,
            bn_desc,
            scale, bias,
            1.0,
            running_mean, running_variance,
            bn.epsilon,
            saved_mean, saved_variance),
        "cudnnBatchNormalizationForwardTraining");
}

void CuDnnBackend::batchNormForwardInference(const BatchNormDesc& bn,
                                             const TensorDesc& xd,
                                             const void* x,
                                             const void* scale,
                                             const void* bias,
                                             const void* mean,
                                             const void* variance,
                                             const TensorDesc& yd,
                                             void* y,
                                             const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "batchNormForwardInference");
    xd.require4d("batchNormForwardInference");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor x_desc(xd);
    TensorDescriptor y_desc(yd);
    TensorDescriptor bn_desc(1, static_cast<int>(xd.dims[1]), 1, 1);

    checkCudnn(
        dnn_dyn::p_cudnnDeriveBNTensorDescriptor(
            bn_desc, x_desc, abi::CUDNN_BATCHNORM_SPATIAL),
        "cudnnDeriveBNTensorDescriptor");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dnn_contract::trace("CUDA", "batchNormForwardInference", "cudnnBatchNormalizationForwardInference");

    checkCudnn(
        dnn_dyn::p_cudnnBatchNormalizationForwardInference(
            handle,
            abi::CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            x_desc, x, y_desc, y,
            bn_desc,
            scale, bias, mean, variance,
            bn.epsilon),
        "cudnnBatchNormalizationForwardInference");
}

void CuDnnBackend::batchNormBackward(const BatchNormDesc& bn,
                                     const TensorDesc& xd,
                                     const void* x,
                                     const TensorDesc& dyd,
                                     const void* dy,
                                     const void* scale,
                                     const void* saved_mean,
                                     const void* saved_variance,
                                     const TensorDesc& dxd,
                                     void* dx,
                                     void* dscale,
                                     void* dbias,
                                     const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "batchNormBackward");
    xd.require4d("batchNormBackward");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor x_desc(xd);
    TensorDescriptor dy_desc(dyd);
    TensorDescriptor dx_desc(dxd);
    TensorDescriptor bn_desc(1, static_cast<int>(xd.dims[1]), 1, 1);

    checkCudnn(
        dnn_dyn::p_cudnnDeriveBNTensorDescriptor(
            bn_desc, x_desc, abi::CUDNN_BATCHNORM_SPATIAL),
        "cudnnDeriveBNTensorDescriptor");

    const float alpha_data = 1.0f;
    const float beta_data = 0.0f;
    const float alpha_param = 1.0f;
    const float beta_param = 0.0f;

    dnn_contract::trace("CUDA", "batchNormBackward", "cudnnBatchNormalizationBackward");

    checkCudnn(
        dnn_dyn::p_cudnnBatchNormalizationBackward(
            handle,
            abi::CUDNN_BATCHNORM_SPATIAL,
            &alpha_data, &beta_data,
            &alpha_param, &beta_param,
            x_desc, x,
            dy_desc, dy,
            dx_desc, dx,
            bn_desc,
            scale,
            dscale, dbias,
            bn.epsilon,
            saved_mean, saved_variance),
        "cudnnBatchNormalizationBackward");
}

namespace {

dnn_contract::RowsView packedRows(const TensorDesc& d, const char* where)
{
    dnn_contract::RowsView view;
    if (!dnn_contract::makeRowsView(d, view) || (view.rows > 1 && view.row_stride != view.inner))
        throw DnnUnsupportedError(std::string("CLAP_DNN cuDNN ") + where +
                                  ": the runtime-compiled LayerNorm kernel requires packed rows");
    return view;
}

void requireFloat32(const TensorDesc& d, const char* where)
{
    if (d.type != DataType::Float32)
        throw DnnUnsupportedError(std::string("CLAP_DNN cuDNN ") + where +
                                  ": the CUDA LayerNorm path (CLAP runtime-compiled kernel) supports Float32 only");
}

}

void CuDnnBackend::layerNormForward(const LayerNormDesc& ln,
                                    const TensorDesc& xd,
                                    const void* x,
                                    const TensorDesc& sd,
                                    const void* scale,
                                    const void* bias,
                                    const TensorDesc& yd,
                                    void* y,
                                    void* saved_mean,
                                    void* saved_rstd,
                                    const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "layerNormForward");
    requireFloat32(xd, "layerNormForward");
    requireFloat32(yd, "layerNormForward");
    requireFloat32(sd, "layerNormForward");
    dnn_contract::requireSameShape(xd, yd, "layerNormForward");

    const auto rows = packedRows(xd, "layerNormForward");
    packedRows(yd, "layerNormForward");

    if (sd.elements() != static_cast<std::size_t>(rows.inner) || !sd.isPacked())
        throw std::runtime_error("layerNormForward: scale/bias must be packed with one value per normalized element");

    std::lock_guard<std::recursive_mutex> lock(native_->mutex);

    const int inner = toInt(rows.inner, "layerNormForward");
    const int outer = toInt(rows.rows, "layerNormForward");

    auto& jit = layerNormJit();
    float eps = static_cast<float>(ln.epsilon);
    void* x_ptr = const_cast<void*>(x);
    void* scale_ptr = const_cast<void*>(scale);
    void* bias_ptr = const_cast<void*>(bias);
    void* y_ptr = y;
    void* mean_ptr = saved_mean;
    void* rstd_ptr = saved_rstd;

    void* args[] = {
        &x_ptr, &scale_ptr, &bias_ptr, &y_ptr, &mean_ptr, &rstd_ptr,
        const_cast<int*>(&outer), const_cast<int*>(&inner), &eps
    };

    dnn_contract::trace("CUDA", "layerNormForward", "runtime CUDA GPU kernel");

    if (dnn_dyn::p_cuLaunchKernel(jit.forward,
                              static_cast<unsigned>(outer), 1, 1,
                              1, 1, 1,
                              0, ctx.stream, args, nullptr) != abi::CUDA_SUCCESS)
        throw std::runtime_error("cuLaunchKernel failed for LayerNorm forward");
}

void CuDnnBackend::layerNormBackward(const LayerNormDesc&,
                                     const TensorDesc& xd,
                                     const void* x,
                                     const TensorDesc& dyd,
                                     const void* dy,
                                     const TensorDesc& sd,
                                     const void* scale,
                                     const void* saved_mean,
                                     const void* saved_rstd,
                                     const TensorDesc& dxd,
                                     void* dx,
                                     void* dscale,
                                     void* dbias,
                                     const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "layerNormBackward");
    requireFloat32(xd, "layerNormBackward");
    requireFloat32(dyd, "layerNormBackward");
    requireFloat32(dxd, "layerNormBackward");
    requireFloat32(sd, "layerNormBackward");
    dnn_contract::requireSameShape(xd, dyd, "layerNormBackward");
    dnn_contract::requireSameShape(xd, dxd, "layerNormBackward");

    if (!saved_mean || !saved_rstd)
        throw std::runtime_error("layerNormBackward requires saved_mean and saved_rstd from layerNormForward");

    const auto rows = packedRows(xd, "layerNormBackward");
    packedRows(dyd, "layerNormBackward");
    packedRows(dxd, "layerNormBackward");

    if (sd.elements() != static_cast<std::size_t>(rows.inner) || !sd.isPacked())
        throw std::runtime_error("layerNormBackward: scale must be packed with one value per normalized element");

    std::lock_guard<std::recursive_mutex> lock(native_->mutex);
    StreamScratch scratch(ctx);

    const int inner = toInt(rows.inner, "layerNormBackward");
    const int outer = toInt(rows.rows, "layerNormBackward");
    const std::size_t param_bytes = static_cast<std::size_t>(inner) * sizeof(float);

    if (!dscale) dscale = scratch.get(param_bytes);
    if (!dbias) dbias = scratch.get(param_bytes);

    checkCuda(dnn_dyn::p_cudaMemsetAsync(dscale, 0, param_bytes, ctx.stream), "cudaMemsetAsync(dscale)");
    checkCuda(dnn_dyn::p_cudaMemsetAsync(dbias, 0, param_bytes, ctx.stream), "cudaMemsetAsync(dbias)");

    auto& jit = layerNormJit();
    void* x_ptr = const_cast<void*>(x);
    void* dy_ptr = const_cast<void*>(dy);
    void* scale_ptr = const_cast<void*>(scale);
    void* mean_ptr = const_cast<void*>(saved_mean);
    void* rstd_ptr = const_cast<void*>(saved_rstd);
    void* dx_ptr = dx;
    void* dscale_ptr = dscale;
    void* dbias_ptr = dbias;

    void* args[] = {
        &x_ptr, &dy_ptr, &scale_ptr, &mean_ptr, &rstd_ptr,
        &dx_ptr, &dscale_ptr, &dbias_ptr,
        const_cast<int*>(&outer), const_cast<int*>(&inner)
    };

    dnn_contract::trace("CUDA", "layerNormBackward", "runtime CUDA GPU kernel");

    if (dnn_dyn::p_cuLaunchKernel(jit.backward,
                              static_cast<unsigned>(outer), 1, 1,
                              1, 1, 1,
                              0, ctx.stream, args, nullptr) != abi::CUDA_SUCCESS)
        throw std::runtime_error("cuLaunchKernel failed for LayerNorm backward");
}

namespace {

struct NormView {
    dnn_contract::RowsView rows;
    std::vector<std::int64_t> dims;
    std::vector<std::int64_t> strides;
};

NormView normView(const TensorDesc& d, const char* where)
{
    NormView v;
    if (!dnn_contract::makeRowsView(d, v.rows))
        throw DnnUnsupportedError(std::string("CLAP_DNN cuDNN ") + where +
                                  ": rows of the normalized tensor are not uniformly strided");
    v.dims = {v.rows.rows, v.rows.inner, 1, 1};
    v.strides = {v.rows.row_stride, 1, 1, 1};
    return v;
}

void requireCudnn9(const char* what)
{
    if (dnn_dyn::p_cudnnGetVersion() < 90000)
        throw DnnUnsupportedError(std::string("CLAP_DNN cuDNN: ") + what + " requires cuDNN 9 or newer");
}

}

void CuDnnBackend::rmsNormForward(const RmsNormDesc& norm,
                                  const TensorDesc& xd,
                                  const void* x,
                                  const TensorDesc& wd,
                                  const void* weight,
                                  const TensorDesc& yd,
                                  void* y,
                                  void* saved_rstd,
                                  const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "rmsNormForward");
    requireCudnn9("RMSNorm");
    dnn_contract::normalizeLastAxis(xd, norm.axis, "rmsNormForward");
    dnn_contract::requireSameShape(xd, yd, "rmsNormForward");

    const NormView xv = normView(xd, "rmsNormForward");
    const NormView yv = normView(yd, "rmsNormForward");
    const std::int64_t inner = xv.rows.inner;

    if (wd.elements() != static_cast<std::size_t>(inner) || !wd.isPacked())
        throw std::runtime_error("rmsNormForward: weight must be packed with one value per normalized element");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);
    StreamScratch scratch(ctx);

    const bool training = saved_rstd != nullptr;
    abi::cudnnHandle_t h = handle;

    std::ostringstream key;
    key << "rms_fwd|" << h << '|' << training << '|';
    cudnn_detail::appendKey(key, xd);
    cudnn_detail::appendKey(key, wd);
    cudnn_detail::appendKey(key, yd);

    auto& slot = native_->plans[key.str()];
    if (!slot) {
        cudnn_detail::GraphBuilder g;

        auto X = g.tensor(1, toCudnnType(xd.type), xv.dims, xv.strides);
        auto W = g.tensor(2, toCudnnType(wd.type), {1, inner, 1, 1}, {inner, 1, 1, 1});
        auto E = g.tensor(3, abi::CUDNN_DATA_FLOAT, {1, 1, 1, 1}, {1, 1, 1, 1}, true);
        auto Y = g.tensor(4, toCudnnType(yd.type), yv.dims, yv.strides);
        auto R = training
            ? g.tensor(5, abi::CUDNN_DATA_FLOAT, {xv.rows.rows, 1, 1, 1}, {1, 1, 1, 1})
            : nullptr;

        g.normForward(
            abi::CUDNN_RMS_NORM,
            training ? abi::CUDNN_NORM_FWD_TRAINING : abi::CUDNN_NORM_FWD_INFERENCE,
            X, W, E, Y, R);

        slot = g.build(h, "RMSNorm forward");
    }

    float epsilon = static_cast<float>(norm.epsilon);

    std::vector<std::int64_t> uids = {1, 2, 3, 4};
    std::vector<void*> pointers = {
        const_cast<void*>(x),
        const_cast<void*>(weight),
        &epsilon,
        y
    };

    if (training) {
        uids.push_back(5);
        pointers.push_back(saved_rstd);
    }

    dnn_contract::trace("CUDA", "rmsNormForward", "cudnnBackendExecute(NORM_FORWARD, CUDNN_RMS_NORM)");
    cudnn_detail::executePlan(h, *slot, uids, pointers, scratch);
}

void CuDnnBackend::rmsNormBackward(const RmsNormDesc& norm,
                                   const TensorDesc& xd,
                                   const void* x,
                                   const TensorDesc& dyd,
                                   const void* dy,
                                   const TensorDesc& wd,
                                   const void* weight,
                                   const void* saved_rstd,
                                   const TensorDesc& dxd,
                                   void* dx,
                                   void* dweight,
                                   const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "rmsNormBackward");
    requireCudnn9("RMSNorm");
    dnn_contract::normalizeLastAxis(xd, norm.axis, "rmsNormBackward");
    dnn_contract::requireSameShape(xd, dyd, "rmsNormBackward");
    dnn_contract::requireSameShape(xd, dxd, "rmsNormBackward");

    if (!saved_rstd)
        throw std::runtime_error("rmsNormBackward requires saved_rstd from rmsNormForward");

    const NormView xv = normView(xd, "rmsNormBackward");
    const NormView dyv = normView(dyd, "rmsNormBackward");
    const NormView dxv = normView(dxd, "rmsNormBackward");
    const std::int64_t inner = xv.rows.inner;

    if (wd.elements() != static_cast<std::size_t>(inner) || !wd.isPacked())
        throw std::runtime_error("rmsNormBackward: weight must be packed with one value per normalized element");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);
    StreamScratch scratch(ctx);
    abi::cudnnHandle_t h = handle;

    if (!dweight)
        dweight = scratch.get(wd.bytes());

    std::ostringstream key;
    key << "rms_bwd|" << h << '|';
    cudnn_detail::appendKey(key, xd);
    cudnn_detail::appendKey(key, dyd);
    cudnn_detail::appendKey(key, wd);
    cudnn_detail::appendKey(key, dxd);

    auto& slot = native_->plans[key.str()];
    if (!slot) {
        cudnn_detail::GraphBuilder g;

        auto X = g.tensor(1, toCudnnType(xd.type), xv.dims, xv.strides);
        auto DY = g.tensor(2, toCudnnType(dyd.type), dyv.dims, dyv.strides);
        auto W = g.tensor(3, toCudnnType(wd.type), {1, inner, 1, 1}, {inner, 1, 1, 1});
        auto R = g.tensor(4, abi::CUDNN_DATA_FLOAT, {xv.rows.rows, 1, 1, 1}, {1, 1, 1, 1});
        auto DX = g.tensor(5, toCudnnType(dxd.type), dxv.dims, dxv.strides);
        auto DW = g.tensor(6, toCudnnType(wd.type), {1, inner, 1, 1}, {inner, 1, 1, 1});

        g.normBackward(abi::CUDNN_RMS_NORM, X, DY, W, R, DX, DW);

        slot = g.build(h, "RMSNorm backward");
    }

    std::vector<std::int64_t> uids = {1, 2, 3, 4, 5, 6};
    std::vector<void*> pointers = {
        const_cast<void*>(x),
        const_cast<void*>(dy),
        const_cast<void*>(weight),
        const_cast<void*>(saved_rstd),
        dx,
        dweight
    };

    dnn_contract::trace("CUDA", "rmsNormBackward", "cudnnBackendExecute(NORM_BACKWARD, CUDNN_RMS_NORM)");
    cudnn_detail::executePlan(h, *slot, uids, pointers, scratch);
}

std::size_t CuDnnBackend::dropoutReserveSpaceSize(const TensorDesc& xd, const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "dropoutReserveSpaceSize");

    TensorDescriptor x_desc(xd);
    std::size_t reserve_size = 0;

    checkCudnn(dnn_dyn::p_cudnnDropoutGetReserveSpaceSize(x_desc, &reserve_size),
               "cudnnDropoutGetReserveSpaceSize");
    return reserve_size;
}

void CuDnnBackend::dropoutForward(const DropoutDesc& dropout,
                                  const TensorDesc& xd,
                                  const void* x,
                                  const TensorDesc& yd,
                                  void* y,
                                  void* reserve_space,
                                  std::size_t reserve_space_bytes,
                                  const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "dropoutForward");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);
    StreamScratch scratch(ctx);

    TensorDescriptor x_desc(xd);
    TensorDescriptor y_desc(yd);

    abi::cudnnDropoutDescriptor_t dropout_desc = nullptr;
    checkCudnn(dnn_dyn::p_cudnnCreateDropoutDescriptor(&dropout_desc),
               "cudnnCreateDropoutDescriptor");

    try {
        std::size_t states_size = 0;
        checkCudnn(dnn_dyn::p_cudnnDropoutGetStatesSize(handle, &states_size),
                   "cudnnDropoutGetStatesSize");

        void* states = scratch.get(states_size);

        checkCudnn(
            dnn_dyn::p_cudnnSetDropoutDescriptor(
                dropout_desc,
                handle,
                dropout.probability,
                states, states_size,
                static_cast<unsigned long long>(dropout.seed)),
            "cudnnSetDropoutDescriptor");

        dnn_contract::trace("CUDA", "dropoutForward", "cudnnDropoutForward");

        checkCudnn(
            dnn_dyn::p_cudnnDropoutForward(
                handle, dropout_desc,
                x_desc, x, y_desc, y,
                reserve_space, reserve_space_bytes),
            "cudnnDropoutForward");

        dnn_dyn::p_cudnnDestroyDropoutDescriptor(dropout_desc);
    }
    catch (...) {
        if (dropout_desc && dnn_dyn::p_cudnnDestroyDropoutDescriptor)
            dnn_dyn::p_cudnnDestroyDropoutDescriptor(dropout_desc);
        throw;
    }
}

void CuDnnBackend::dropoutBackward(const DropoutDesc& dropout,
                                   const TensorDesc& dyd,
                                   const void* dy,
                                   const void* reserve_space,
                                   std::size_t reserve_space_bytes,
                                   const TensorDesc& dxd,
                                   void* dx,
                                   const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "dropoutBackward");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);

    TensorDescriptor dy_desc(dyd);
    TensorDescriptor dx_desc(dxd);

    abi::cudnnDropoutDescriptor_t dropout_desc = nullptr;
    checkCudnn(dnn_dyn::p_cudnnCreateDropoutDescriptor(&dropout_desc),
               "cudnnCreateDropoutDescriptor");

    try {
        checkCudnn(
            dnn_dyn::p_cudnnSetDropoutDescriptor(
                dropout_desc,
                handle,
                dropout.probability,
                nullptr, 0,
                static_cast<unsigned long long>(dropout.seed)),
            "cudnnSetDropoutDescriptor");

        dnn_contract::trace("CUDA", "dropoutBackward", "cudnnDropoutBackward");

        checkCudnn(
            dnn_dyn::p_cudnnDropoutBackward(
                handle, dropout_desc,
                dy_desc, dy, dx_desc, dx,
                const_cast<void*>(reserve_space), reserve_space_bytes),
            "cudnnDropoutBackward");

        dnn_dyn::p_cudnnDestroyDropoutDescriptor(dropout_desc);
    }
    catch (...) {
        if (dropout_desc && dnn_dyn::p_cudnnDestroyDropoutDescriptor)
            dnn_dyn::p_cudnnDestroyDropoutDescriptor(dropout_desc);
        throw;
    }
}

void CuDnnBackend::lrnForward(const LrnDesc& lrn, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "lrnForward");
    xd.require4d("lrnForward");

    ScopedHandle h(native_->mutex, native_->handles, ctx);
    TensorDescriptor Xd(xd);
    TensorDescriptor Yd(yd);
    abi::cudnnLRNDescriptor_t d = nullptr;

    checkCudnn(dnn_dyn::p_cudnnCreateLRNDescriptor(&d), "cudnnCreateLRNDescriptor");
    try {
        checkCudnn(
            dnn_dyn::p_cudnnSetLRNDescriptor(
                d, static_cast<unsigned>(lrn.local_size), lrn.alpha, lrn.beta, lrn.k),
            "cudnnSetLRNDescriptor");

        float one = 1;
        float zero = 0;
        dnn_contract::trace("CUDA", "lrnForward", "cudnnLRNCrossChannelForward");
        checkCudnn(
            dnn_dyn::p_cudnnLRNCrossChannelForward(
                h, d, abi::CUDNN_LRN_CROSS_CHANNEL_DIM1,
                &one, Xd, x, &zero, Yd, y),
            "cudnnLRNCrossChannelForward");
        dnn_dyn::p_cudnnDestroyLRNDescriptor(d);
    }
    catch (...) {
        if (d)
            dnn_dyn::p_cudnnDestroyLRNDescriptor(d);
        throw;
    }
}

void CuDnnBackend::lrnBackward(const LrnDesc& lrn, const TensorDesc& xd, const void* x, const TensorDesc& yd, const void* y, const TensorDesc& dyd, const void* dy, const TensorDesc& dxd, void* dx, const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "lrnBackward");
    xd.require4d("lrnBackward");

    ScopedHandle h(native_->mutex, native_->handles, ctx);
    TensorDescriptor Xd(xd);
    TensorDescriptor Yd(yd);
    TensorDescriptor DYd(dyd);
    TensorDescriptor DXd(dxd);
    abi::cudnnLRNDescriptor_t d = nullptr;

    checkCudnn(dnn_dyn::p_cudnnCreateLRNDescriptor(&d), "cudnnCreateLRNDescriptor");
    try {
        checkCudnn(
            dnn_dyn::p_cudnnSetLRNDescriptor(
                d, static_cast<unsigned>(lrn.local_size), lrn.alpha, lrn.beta, lrn.k),
            "cudnnSetLRNDescriptor");

        float one = 1;
        float zero = 0;
        dnn_contract::trace("CUDA", "lrnBackward", "cudnnLRNCrossChannelBackward");
        checkCudnn(
            dnn_dyn::p_cudnnLRNCrossChannelBackward(
                h, d, abi::CUDNN_LRN_CROSS_CHANNEL_DIM1,
                &one, Yd, y, DYd, dy, Xd, x, &zero, DXd, dx),
            "cudnnLRNCrossChannelBackward");
        dnn_dyn::p_cudnnDestroyLRNDescriptor(d);
    }
    catch (...) {
        if (d)
            dnn_dyn::p_cudnnDestroyLRNDescriptor(d);
        throw;
    }
}

static void cudnnOp(abi::cudnnHandle_t h, abi::cudnnOpTensorOp_t op, const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y)
{
    TensorDescriptor A(ad);
    TensorDescriptor B(bd);
    TensorDescriptor Y(yd);
    abi::cudnnOpTensorDescriptor_t d = nullptr;

    checkCudnn(dnn_dyn::p_cudnnCreateOpTensorDescriptor(&d), "cudnnCreateOpTensorDescriptor");
    try {
        checkCudnn(
            dnn_dyn::p_cudnnSetOpTensorDescriptor(
                d, op, abi::CUDNN_DATA_FLOAT, abi::CUDNN_PROPAGATE_NAN),
            "cudnnSetOpTensorDescriptor");

        float one = 1;
        float zero = 0;
        dnn_contract::trace("CUDA", "tensorOp", "cudnnOpTensor");
        checkCudnn(
            dnn_dyn::p_cudnnOpTensor(
                h, d, &one, A, a, &one, B, b, &zero, Y, y),
            "cudnnOpTensor");
        dnn_dyn::p_cudnnDestroyOpTensorDescriptor(d);
    }
    catch (...) {
        if (d)
            dnn_dyn::p_cudnnDestroyOpTensorDescriptor(d);
        throw;
    }
}

void CuDnnBackend::tensorAdd(const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "tensorAdd");
    ScopedHandle h(native_->mutex, native_->handles, ctx);
    cudnnOp(h, abi::CUDNN_OP_TENSOR_ADD, ad, a, bd, b, yd, y);
}

void CuDnnBackend::tensorMultiply(const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "tensorMultiply");
    ScopedHandle h(native_->mutex, native_->handles, ctx);
    cudnnOp(h, abi::CUDNN_OP_TENSOR_MUL, ad, a, bd, b, yd, y);
}

void CuDnnBackend::tensorMin(const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "tensorMin");
    ScopedHandle h(native_->mutex, native_->handles, ctx);
    cudnnOp(h, abi::CUDNN_OP_TENSOR_MIN, ad, a, bd, b, yd, y);
}

void CuDnnBackend::tensorMax(const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "tensorMax");
    ScopedHandle h(native_->mutex, native_->handles, ctx);
    cudnnOp(h, abi::CUDNN_OP_TENSOR_MAX, ad, a, bd, b, yd, y);
}

static void cudnnReduce(abi::cudnnHandle_t h, StreamScratch& scratch, abi::cudnnReduceTensorOp_t op, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y)
{
    TensorDescriptor X(xd);
    TensorDescriptor Y(yd);
    abi::cudnnReduceTensorDescriptor_t d = nullptr;

    checkCudnn(dnn_dyn::p_cudnnCreateReduceTensorDescriptor(&d), "cudnnCreateReduceTensorDescriptor");
    try {
        checkCudnn(
            dnn_dyn::p_cudnnSetReduceTensorDescriptor(
                d, op, abi::CUDNN_DATA_FLOAT, abi::CUDNN_PROPAGATE_NAN_REDUCE,
                abi::CUDNN_REDUCE_TENSOR_NO_INDICES, abi::CUDNN_32BIT_INDICES),
            "cudnnSetReduceTensorDescriptor");

        std::size_t ws = 0;
        checkCudnn(
            dnn_dyn::p_cudnnGetReductionWorkspaceSize(h, d, X, Y, &ws),
            "cudnnGetReductionWorkspaceSize");
        void* W = scratch.get(ws);

        float one = 1;
        float zero = 0;
        dnn_contract::trace("CUDA", "tensorReduce", "cudnnReduceTensor");
        checkCudnn(
            dnn_dyn::p_cudnnReduceTensor(
                h, d, nullptr, 0, W, ws, &one, X, x, &zero, Y, y),
            "cudnnReduceTensor");
        dnn_dyn::p_cudnnDestroyReduceTensorDescriptor(d);
    }
    catch (...) {
        if (d)
            dnn_dyn::p_cudnnDestroyReduceTensorDescriptor(d);
        throw;
    }
}

void CuDnnBackend::tensorReduceSum(const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "tensorReduceSum");
    ScopedHandle h(native_->mutex, native_->handles, ctx);
    StreamScratch scratch(ctx);
    cudnnReduce(h, scratch, abi::CUDNN_REDUCE_TENSOR_ADD, xd, x, yd, y);
}

void CuDnnBackend::tensorReduceProduct(const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "tensorReduceProduct");
    ScopedHandle h(native_->mutex, native_->handles, ctx);
    StreamScratch scratch(ctx);
    cudnnReduce(h, scratch, abi::CUDNN_REDUCE_TENSOR_MUL, xd, x, yd, y);
}

namespace {

enum SdpaUid : std::int64_t {
    SDPA_Q = 1,
    SDPA_K = 2,
    SDPA_V = 3,
    SDPA_O = 4,
    SDPA_MASK = 5,
    SDPA_SCALE = 6,
    SDPA_NEG_INF = 7
};

float sdpaNegativeInfinity()
{
    if (dnn_dyn::p_cudnnGetVersion() < 91000)
        return std::numeric_limits<float>::lowest();
    return -std::numeric_limits<float>::infinity();
}

std::unique_ptr<cudnn_detail::GraphPlan> buildSdpaPlan(abi::cudnnHandle_t handle,
                                                       const dnn_contract::AttentionShape& shape,
                                                       bool causal,
                                                       const TensorDesc& qd,
                                                       const TensorDesc& kd,
                                                       const TensorDesc& vd,
                                                       const TensorDesc* mask_desc,
                                                       const TensorDesc& od)
{
    using cudnn_detail::GraphBuilder;

    const abi::cudnnDataType_t dt = toCudnnType(qd.type);
    const auto qs = qd.effectiveStrides();
    const auto ks = kd.effectiveStrides();
    const auto vs = vd.effectiveStrides();
    const auto os = od.effectiveStrides();

    const std::int64_t b = shape.batch;
    const std::int64_t hq = shape.q_heads;
    const std::int64_t hk = shape.kv_heads;
    const std::int64_t sq = shape.q_len;
    const std::int64_t skv = shape.kv_len;
    const std::int64_t d = shape.head_dim;
    const std::int64_t dv = shape.value_dim;

    GraphBuilder g;

    auto Q = g.tensor(SDPA_Q, dt, {b, hq, sq, d}, qs);
    auto KT = g.tensor(SDPA_K, dt, {b, hk, d, skv}, {ks[0], ks[1], ks[3], ks[2]});
    auto V = g.tensor(SDPA_V, dt, {b, hk, skv, dv}, vs);
    auto O = g.tensor(SDPA_O, dt, {b, hq, sq, dv}, os);

    const std::vector<std::int64_t> score = {b, hq, sq, skv};
    const std::vector<std::int64_t> row_stat = {b, hq, sq, 1};

    auto S = g.virtualTensor(abi::CUDNN_DATA_FLOAT, score);
    g.matmul(Q, KT, S);

    auto SCALE = g.tensor(SDPA_SCALE, abi::CUDNN_DATA_FLOAT, {1, 1, 1, 1}, {1, 1, 1, 1}, true);
    auto scaled = g.virtualTensor(abi::CUDNN_DATA_FLOAT, score);
    g.pointwise(abi::CUDNN_POINTWISE_MUL, S, SCALE, nullptr, scaled);

    auto current = scaled;

    if (mask_desc) {
        auto MASK = g.tensor(SDPA_MASK, toCudnnType(mask_desc->type), mask_desc->dims, mask_desc->effectiveStrides());
        auto masked = g.virtualTensor(abi::CUDNN_DATA_FLOAT, score);
        g.pointwise(abi::CUDNN_POINTWISE_ADD, current, MASK, nullptr, masked);
        current = masked;
    }

    if (causal) {
        auto row = g.virtualTensor(abi::CUDNN_DATA_INT32, score);
        g.pointwise(abi::CUDNN_POINTWISE_GEN_INDEX, current, nullptr, nullptr, row, abi::CUDNN_DATA_FLOAT, 2);

        auto col = g.virtualTensor(abi::CUDNN_DATA_INT32, score);
        g.pointwise(abi::CUDNN_POINTWISE_GEN_INDEX, current, nullptr, nullptr, col, abi::CUDNN_DATA_FLOAT, 3);

        auto keep = g.virtualTensor(abi::CUDNN_DATA_BOOLEAN, score);
        g.pointwise(abi::CUDNN_POINTWISE_CMP_GE, row, col, nullptr, keep, abi::CUDNN_DATA_BOOLEAN);

        auto NEG_INF = g.tensor(SDPA_NEG_INF, abi::CUDNN_DATA_FLOAT, {1, 1, 1, 1}, {1, 1, 1, 1}, true);
        auto selected = g.virtualTensor(abi::CUDNN_DATA_FLOAT, score);
        g.pointwise(abi::CUDNN_POINTWISE_BINARY_SELECT, current, NEG_INF, keep, selected);
        current = selected;
    }

    auto row_max = g.virtualTensor(abi::CUDNN_DATA_FLOAT, row_stat);
    g.reduction(abi::CUDNN_REDUCE_TENSOR_MAX, current, row_max);

    auto shifted = g.virtualTensor(abi::CUDNN_DATA_FLOAT, score);
    g.pointwise(abi::CUDNN_POINTWISE_SUB, current, row_max, nullptr, shifted);

    auto exp_scores = g.virtualTensor(abi::CUDNN_DATA_FLOAT, score);
    g.pointwise(abi::CUDNN_POINTWISE_EXP, shifted, nullptr, nullptr, exp_scores);

    auto row_sum = g.virtualTensor(abi::CUDNN_DATA_FLOAT, row_stat);
    g.reduction(abi::CUDNN_REDUCE_TENSOR_ADD, exp_scores, row_sum);

    auto log_sum = g.virtualTensor(abi::CUDNN_DATA_FLOAT, row_stat);
    g.pointwise(abi::CUDNN_POINTWISE_LOG, row_sum, nullptr, nullptr, log_sum);

    auto stats = g.virtualTensor(abi::CUDNN_DATA_FLOAT, row_stat);
    g.pointwise(abi::CUDNN_POINTWISE_ADD, row_max, log_sum, nullptr, stats);

    auto probs = g.virtualTensor(abi::CUDNN_DATA_FLOAT, score);
    g.pointwise(abi::CUDNN_POINTWISE_DIV, exp_scores, row_sum, nullptr, probs);

    g.matmul(probs, V, O);

    return g.build(handle, "scaled dot-product attention");
}

}

void CuDnnBackend::scaledDotProductAttentionForward(const AttentionDesc& attention,
                                                    const TensorDesc& qd,
                                                    const void* q,
                                                    const TensorDesc& kd,
                                                    const void* k,
                                                    const TensorDesc& vd,
                                                    const void* v,
                                                    const TensorDesc* mask_desc,
                                                    const void* mask,
                                                    const TensorDesc& od,
                                                    void* o,
                                                    const ExecutionContext& ctx)
{
    requireCudaContext(ctx, "scaledDotProductAttentionForward");

    if ((mask_desc == nullptr) != (mask == nullptr))
        throw std::runtime_error("scaledDotProductAttentionForward: mask_desc and mask must both be set or both be null");

    const auto shape = dnn_contract::validateAttention(attention, qd, kd, vd, mask_desc, od);

    if (qd.type == DataType::Float32)
        dnn_contract::unsupported("cuDNN", "fused flash attention supports Float16/BFloat16 Q/K/V only");
    if (attention.dropout > 0.0f)
        dnn_contract::unsupported("cuDNN", "attention dropout needs the cuDNN RNG graph node, which this backend does not build");
    requireCudnn9("fused scaled dot-product attention");

    ScopedHandle handle(native_->mutex, native_->handles, ctx);
    StreamScratch scratch(ctx);
    abi::cudnnHandle_t h = handle;

    std::ostringstream key;
    key << "sdpa|" << h << '|' << attention.causal << '|';
    cudnn_detail::appendKey(key, qd);
    cudnn_detail::appendKey(key, kd);
    cudnn_detail::appendKey(key, vd);
    cudnn_detail::appendKey(key, od);
    if (mask_desc)
        cudnn_detail::appendKey(key, *mask_desc);

    auto& slot = native_->plans[key.str()];
    if (!slot)
        slot = buildSdpaPlan(h, shape, attention.causal, qd, kd, vd, mask_desc, od);

    float scale = static_cast<float>(shape.scale);
    float neg_inf = sdpaNegativeInfinity();

    std::vector<std::int64_t> uids = {SDPA_Q, SDPA_K, SDPA_V, SDPA_O, SDPA_SCALE};
    std::vector<void*> pointers = {
        const_cast<void*>(q),
        const_cast<void*>(k),
        const_cast<void*>(v),
        o,
        &scale
    };

    if (mask_desc) {
        uids.push_back(SDPA_MASK);
        pointers.push_back(const_cast<void*>(mask));
    }
    if (attention.causal) {
        uids.push_back(SDPA_NEG_INF);
        pointers.push_back(&neg_inf);
    }

    dnn_contract::trace("CUDA", "scaledDotProductAttentionForward", "cudnnBackendExecute(fused flash attention graph)");
    cudnn_detail::executePlan(h, *slot, uids, pointers, scratch);
}

DnnSupport CuDnnBackend::supports(const DnnCapabilityQuery& query, const ExecutionContext& ctx) const
{
    if (ctx.backend != ExecutionBackend::CUDA)
        return DnnSupport::no("the cuDNN backend executes on CUDA device memory (ExecutionBackend::CUDA)");

    try {
        const TensorDesc& x = query.tensor(0);

        for (const auto& t : query.tensors) {
            if (t.dims.empty() || t.dims.size() > 8)
                return DnnSupport::no("cuDNN tensors must have rank 1..8");
            t.spanElements();
        }

        switch (query.operation) {
            case DnnOperation::ActivationForward:
            case DnnOperation::ActivationBackward:
                return DnnSupport::yes(query.activation.mode == ActivationMode::SiLU
                    ? "cudnnActivation* with CUDNN_ACTIVATION_SWISH (beta = 1)"
                    : "cudnnActivation*");

            case DnnOperation::SoftmaxForward:
            case DnnOperation::SoftmaxBackward: {
                const int axis = x.normalizeAxis(query.softmax.axis, "supports(softmax)");
                dnn_contract::AxisView view;
                if (!dnn_contract::makeAxisView(x, axis, view))
                    return DnnSupport::no("the dimensions around the softmax axis cannot be collapsed into a cuDNN [N, C, H, 1] view");
                if (view.outer > INT_MAX || view.axis > INT_MAX || view.inner > INT_MAX)
                    return DnnSupport::no("softmax extents exceed the cuDNN int descriptor range");
                return DnnSupport::yes(query.softmax.log_softmax
                    ? "cudnnSoftmax* with CUDNN_SOFTMAX_LOG"
                    : "cudnnSoftmax* with CUDNN_SOFTMAX_ACCURATE");
            }

            case DnnOperation::LayerNormForward:
            case DnnOperation::LayerNormBackward: {
                dnn_contract::RowsView rows;
                if (x.type != DataType::Float32)
                    return DnnSupport::no("CUDA LayerNorm runs the existing CLAP runtime-compiled kernel, which is Float32-only");
                if (!dnn_contract::makeRowsView(x, rows) || (rows.rows > 1 && rows.row_stride != rows.inner))
                    return DnnSupport::no("CUDA LayerNorm requires packed rows");
                return DnnSupport::yes("existing CLAP runtime-compiled CUDA LayerNorm kernel (not a cuDNN primitive)");
            }

            case DnnOperation::RmsNormForward:
            case DnnOperation::RmsNormBackward: {
                if (dnn_dyn::p_cudnnGetVersion() < 90000)
                    return DnnSupport::no("RMSNorm requires cuDNN 9 or newer");
                dnn_contract::normalizeLastAxis(x, query.rms_norm.axis, "supports(rmsNorm)");
                dnn_contract::RowsView rows;
                if (!dnn_contract::makeRowsView(x, rows))
                    return DnnSupport::no("rows of the normalized tensor are not uniformly strided");
                return DnnSupport::yes("cuDNN graph NORM operation in CUDNN_RMS_NORM mode (engine availability is checked at plan creation)");
            }

            case DnnOperation::ScaledDotProductAttentionForward: {
                if (dnn_dyn::p_cudnnGetVersion() < 90000)
                    return DnnSupport::no("fused SDPA requires cuDNN 9 or newer");
                const TensorDesc* mask = query.has_mask ? &query.tensor(4) : nullptr;
                const auto shape = dnn_contract::validateAttention(
                    query.attention, query.tensor(0), query.tensor(1), query.tensor(2), mask, query.tensor(3));
                if (x.type == DataType::Float32)
                    return DnnSupport::no("cuDNN fused flash attention supports Float16/BFloat16 Q/K/V only");
                if (query.attention.dropout > 0.0f)
                    return DnnSupport::no("attention dropout needs the cuDNN RNG graph node, which this backend does not build");
                if (shape.head_dim % 8 != 0 || shape.value_dim % 8 != 0 || shape.head_dim > 256 || shape.value_dim > 256)
                    return DnnSupport::no("cuDNN fused flash attention requires head dimensions that are multiples of 8 and at most 256");
                return DnnSupport::yes("cuDNN graph fused flash attention (engine availability is checked at plan creation)");
            }

            case DnnOperation::FusedConvolutionBiasActivation:
                if (query.activation.mode != ActivationMode::ReLU)
                    return DnnSupport::no("cudnnConvolutionBiasActivationForward supports ReLU (and identity) activations only");
                if (x.rank() != 4)
                    return DnnSupport::no("convolution expects 4D NCHW tensors");
                return DnnSupport::yes("cudnnConvolutionBiasActivationForward");

            case DnnOperation::DropoutForward:
            case DnnOperation::DropoutBackward:
                return DnnSupport::yes("cudnnDropout* with caller-owned reserve space");

            case DnnOperation::TensorAdd:
            case DnnOperation::TensorMultiply:
            case DnnOperation::TensorMin:
            case DnnOperation::TensorMax:
                return DnnSupport::yes("cudnnOpTensor");

            case DnnOperation::TensorReduceSum:
            case DnnOperation::TensorReduceProduct:
                return DnnSupport::yes("cudnnReduceTensor");

            default:
                if (x.rank() != 4)
                    return DnnSupport::no(std::string(dnnOperationName(query.operation)) + " expects 4D NCHW tensors");
                return DnnSupport::yes();
        }
    }
    catch (const std::exception& e) {
        return DnnSupport::no(e.what());
    }
}

}
