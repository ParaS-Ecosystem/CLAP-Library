#pragma once

#include <dnn.hh>
#include "clap/dnn_dyn_backends.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace clap_example {

inline void checkStatus(int status, const char* what)
{
    if (status != 0)
        throw std::runtime_error(std::string(what) + " failed (status=" + std::to_string(status) + ")");
}

class Device {
public:
    explicit Device(clap::IDnnBackend& backend, bool use_caller_stream = true)
    {
        ctx.backend = backend.executionBackend();

        if (!use_caller_stream)
            return;

        if (ctx.backend == clap::ExecutionBackend::CUDA) {
            checkStatus(clap::dnn_dyn::p_cudaStreamCreate(&ctx.stream), "cudaStreamCreate");
        } else if (ctx.backend == clap::ExecutionBackend::ROCM) {
            checkStatus(clap::dnn_dyn::p_hipStreamCreate(&ctx.stream), "hipStreamCreate");
        } else {
            checkStatus(clap::dnn_dyn::p_dnnl_engine_create(&cpu_engine_, clap::abi::dnnl_cpu, 0), "dnnl_engine_create");
            checkStatus(clap::dnn_dyn::p_dnnl_stream_create(&ctx.stream, cpu_engine_, clap::abi::dnnl_stream_in_order), "dnnl_stream_create");
        }
    }

    ~Device()
    {
        for (void* p : allocations_)
            release(p);

        if (!ctx.stream)
            return;

        if (ctx.backend == clap::ExecutionBackend::CUDA) {
            clap::dnn_dyn::p_cudaStreamDestroy(ctx.stream);
        } else if (ctx.backend == clap::ExecutionBackend::ROCM) {
            clap::dnn_dyn::p_hipStreamDestroy(ctx.stream);
        } else {
            clap::dnn_dyn::p_dnnl_stream_destroy(ctx.stream);
            clap::dnn_dyn::p_dnnl_engine_destroy(cpu_engine_);
        }
    }

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    bool onGpu() const { return ctx.backend != clap::ExecutionBackend::CPU; }

    void* allocate(std::size_t bytes)
    {
        void* p = nullptr;
        if (bytes == 0)
            bytes = 1;

        if (ctx.backend == clap::ExecutionBackend::CUDA)
            checkStatus(clap::dnn_dyn::p_cudaMalloc(&p, bytes), "cudaMalloc");
        else if (ctx.backend == clap::ExecutionBackend::ROCM)
            checkStatus(clap::dnn_dyn::p_hipMalloc(&p, bytes), "hipMalloc");
        else
            p = std::calloc(1, bytes);

        if (!p)
            throw std::runtime_error("allocation failed");
        allocations_.push_back(p);
        return p;
    }

    void upload(void* dst, const void* src, std::size_t bytes)
    {
        if (ctx.backend == clap::ExecutionBackend::CUDA)
            checkStatus(clap::dnn_dyn::p_cudaMemcpy(dst, src, bytes, clap::abi::cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        else if (ctx.backend == clap::ExecutionBackend::ROCM)
            checkStatus(clap::dnn_dyn::p_hipMemcpy(dst, src, bytes, clap::abi::hipMemcpyHostToDevice), "hipMemcpy H2D");
        else
            std::memcpy(dst, src, bytes);
    }

    void synchronize()
    {
        if (ctx.backend == clap::ExecutionBackend::CUDA)
            checkStatus(clap::dnn_dyn::p_cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize");
        else if (ctx.backend == clap::ExecutionBackend::ROCM)
            checkStatus(clap::dnn_dyn::p_hipStreamSynchronize(ctx.stream), "hipStreamSynchronize");
        else if (ctx.stream)
            checkStatus(clap::dnn_dyn::p_dnnl_stream_wait(ctx.stream), "dnnl_stream_wait");
    }

    void download(void* dst, const void* src, std::size_t bytes)
    {
        synchronize();
        if (ctx.backend == clap::ExecutionBackend::CUDA)
            checkStatus(clap::dnn_dyn::p_cudaMemcpy(dst, src, bytes, clap::abi::cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        else if (ctx.backend == clap::ExecutionBackend::ROCM)
            checkStatus(clap::dnn_dyn::p_hipMemcpy(dst, src, bytes, clap::abi::hipMemcpyDeviceToHost), "hipMemcpy D2H");
        else
            std::memcpy(dst, src, bytes);
    }

    template <class T>
    void* uploadVector(const std::vector<T>& host)
    {
        void* p = allocate(host.size() * sizeof(T));
        upload(p, host.data(), host.size() * sizeof(T));
        return p;
    }

    template <class T>
    std::vector<T> downloadVector(const void* p, std::size_t count)
    {
        std::vector<T> host(count);
        download(host.data(), p, count * sizeof(T));
        return host;
    }

    clap::ExecutionContext ctx;

private:
    void release(void* p)
    {
        if (ctx.backend == clap::ExecutionBackend::CUDA)
            clap::dnn_dyn::p_cudaFree(p);
        else if (ctx.backend == clap::ExecutionBackend::ROCM)
            clap::dnn_dyn::p_hipFree(p);
        else
            std::free(p);
    }

    clap::abi::dnnl_engine_t cpu_engine_ = nullptr;
    std::vector<void*> allocations_;
};

inline std::uint16_t floatToBFloat16(float f)
{
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u)
        return static_cast<std::uint16_t>((u >> 16) | 0x40u);
    const std::uint32_t rounding = 0x7fffu + ((u >> 16) & 1u);
    return static_cast<std::uint16_t>((u + rounding) >> 16);
}

inline float bfloat16ToFloat(std::uint16_t h)
{
    const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline std::uint16_t floatToHalf(float f)
{
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    const std::uint32_t sign = (u >> 16) & 0x8000u;
    std::uint32_t mag = u & 0x7fffffffu;

    if (mag > 0x7f800000u)
        return static_cast<std::uint16_t>(sign | 0x7e00u);
    if (mag >= 0x477ff000u)
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (mag < 0x38800000u) {
        float v;
        std::memcpy(&v, &mag, sizeof(v));
        const float scaled = v * 16777216.0f;
        return static_cast<std::uint16_t>(sign | static_cast<std::uint32_t>(std::nearbyint(scaled)));
    }

    const std::uint32_t rounding = 0xfffu + ((mag >> 13) & 1u);
    mag += rounding;
    return static_cast<std::uint16_t>(sign | ((mag - 0x38000000u) >> 13));
}

inline float halfToFloat(std::uint16_t h)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1fu;
    const std::uint32_t man = h & 0x3ffu;
    std::uint32_t u;

    if (exp == 0) {
        const float v = std::ldexp(static_cast<float>(man), -24);
        std::memcpy(&u, &v, sizeof(u));
        u |= sign;
    } else if (exp == 31) {
        u = sign | 0x7f800000u | (man << 13);
    } else {
        u = sign | ((exp + 112u) << 23) | (man << 13);
    }

    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline std::vector<std::uint8_t> encode(const std::vector<float>& values, clap::DataType type)
{
    std::vector<std::uint8_t> out(values.size() * clap::dataTypeSize(type));
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (type == clap::DataType::Float32) {
            std::memcpy(&out[i * 4], &values[i], 4);
        } else {
            const std::uint16_t h = type == clap::DataType::BFloat16
                ? floatToBFloat16(values[i])
                : floatToHalf(values[i]);
            std::memcpy(&out[i * 2], &h, 2);
        }
    }
    return out;
}

inline std::vector<float> decode(const std::vector<std::uint8_t>& raw, clap::DataType type)
{
    const std::size_t n = raw.size() / clap::dataTypeSize(type);
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (type == clap::DataType::Float32) {
            std::memcpy(&out[i], &raw[i * 4], 4);
        } else {
            std::uint16_t h;
            std::memcpy(&h, &raw[i * 2], 2);
            out[i] = type == clap::DataType::BFloat16 ? bfloat16ToFloat(h) : halfToFloat(h);
        }
    }
    return out;
}

}
