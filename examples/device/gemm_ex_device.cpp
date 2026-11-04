#include <blas.hh>
#include <clap/dyn_backends.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace clap;

namespace {

constexpr int M = 32;
constexpr int N = 24;
constexpr int K = 16;
constexpr int BATCHES = 3;

std::uint16_t float_to_fp16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000;
    const std::uint32_t exponent = (bits >> 23) & 0xff;
    std::uint32_t mantissa = bits & 0x7fffff;

    if (exponent == 0xff) {
        return static_cast<std::uint16_t>(
            sign | 0x7c00 | (mantissa ? 0x0200 : 0));
    }

    int half_exp = static_cast<int>(exponent) - 127 + 15;

    if (half_exp >= 31)
        return static_cast<std::uint16_t>(sign | 0x7c00);

    if (half_exp <= 0) {
        if (half_exp < -10)
            return static_cast<std::uint16_t>(sign);

        mantissa |= 0x800000;
        const unsigned shift = 14 - half_exp;
        std::uint32_t result = mantissa >> shift;
        const std::uint32_t remainder =
            mantissa & ((1u << shift) - 1);
        const std::uint32_t halfway = 1u << (shift - 1);

        if (remainder > halfway ||
            (remainder == halfway && (result & 1)))
            ++result;

        return static_cast<std::uint16_t>(sign | result);
    }

    std::uint32_t result = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fff;

    if (remainder > 0x1000 ||
        (remainder == 0x1000 && (result & 1))) {
        ++result;
        if (result == 0x400) {
            result = 0;
            ++half_exp;
            if (half_exp >= 31)
                return static_cast<std::uint16_t>(sign | 0x7c00);
        }
    }

    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(half_exp) << 10) | result);
}

float fp16_to_float(std::uint16_t value) {
    const std::uint32_t sign =
        (static_cast<std::uint32_t>(value & 0x8000)) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1f;
    std::uint32_t mantissa = value & 0x3ff;
    std::uint32_t bits;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                --e;
            }
            mantissa &= 0x3ff;
            bits = sign |
                   (static_cast<std::uint32_t>(e + 127) << 23) |
                   (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000 | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) |
               (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    if ((bits & 0x7f800000) == 0x7f800000 &&
        (bits & 0x007fffff))
        return static_cast<std::uint16_t>((bits >> 16) | 0x0040);

    bits += 0x7fff + ((bits >> 16) & 1);
    return static_cast<std::uint16_t>(bits >> 16);
}

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::size_t type_size(BlasDataType type) {
    return type == BlasDataType::Float32
               ? sizeof(float)
               : sizeof(std::uint16_t);
}

const char* type_name(BlasDataType type) {
    switch (type) {
        case BlasDataType::Float16:  return "FP16";
        case BlasDataType::BFloat16: return "BF16";
        case BlasDataType::Float32:  return "FP32";
    }
    return "Unknown";
}

void encode_values(const std::vector<float>& input,
                   BlasDataType type,
                   std::vector<unsigned char>& output) {
    output.resize(input.size() * type_size(type));

    if (type == BlasDataType::Float32) {
        std::memcpy(output.data(), input.data(),
                    input.size() * sizeof(float));
        return;
    }

    for (std::size_t i = 0; i < input.size(); ++i) {
        const std::uint16_t value =
            type == BlasDataType::Float16
                ? float_to_fp16(input[i])
                : float_to_bf16(input[i]);

        std::memcpy(output.data() + i * sizeof(value),
                    &value, sizeof(value));
    }
}

float decode_value(const unsigned char* data,
                   BlasDataType type,
                   std::size_t index) {
    if (type == BlasDataType::Float32) {
        float value;
        std::memcpy(&value, data + index * sizeof(float),
                    sizeof(value));
        return value;
    }

    std::uint16_t value;
    std::memcpy(&value, data + index * sizeof(value),
                sizeof(value));

    return type == BlasDataType::Float16
               ? fp16_to_float(value)
               : bf16_to_float(value);
}

class DeviceBuffer {
public:
    DeviceBuffer(BackendType backend, std::size_t bytes)
        : backend_(backend), pointer_(nullptr) {
        int status = -1;

        if (backend_ == BackendType::CUDA) {
            if (!dyn::p_cudaMalloc)
                throw std::runtime_error("CUDA allocation function is unavailable");
            status = dyn::p_cudaMalloc(&pointer_, bytes);
        } else if (backend_ == BackendType::AMD) {
            if (!dyn::p_hipMalloc)
                throw std::runtime_error("HIP allocation function is unavailable");
            status = dyn::p_hipMalloc(&pointer_, bytes);
        }

        if (status != 0 || !pointer_)
            throw std::runtime_error("Device memory allocation failed");
    }

    ~DeviceBuffer() {
        if (!pointer_)
            return;

        if (backend_ == BackendType::CUDA && dyn::p_cudaFree)
            dyn::p_cudaFree(pointer_);
        else if (backend_ == BackendType::AMD && dyn::p_hipFree)
            dyn::p_hipFree(pointer_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const {
        return pointer_;
    }

private:
    BackendType backend_;
    void* pointer_;
};

void copy_to_device(BackendType backend, void* destination,
                    const void* source, std::size_t bytes) {
    int status = -1;

    if (backend == BackendType::CUDA) {
        status = dyn::p_cudaMemcpy(
            destination, source, bytes, cudaMemcpyHostToDevice);
    } else {
        status = dyn::p_hipMemcpy(
            destination, source, bytes, hipMemcpyHostToDevice);
    }

    if (status != 0)
        throw std::runtime_error("Host-to-device copy failed");
}

void copy_from_device(BackendType backend, void* destination,
                      const void* source, std::size_t bytes) {
    int status = -1;

    if (backend == BackendType::CUDA) {
        status = dyn::p_cudaMemcpy(
            destination, source, bytes, cudaMemcpyDeviceToHost);
    } else {
        status = dyn::p_hipMemcpy(
            destination, source, bytes, hipMemcpyDeviceToHost);
    }

    if (status != 0)
        throw std::runtime_error("Device-to-host copy failed");
}

void run_case(IBlasBackend& backend, BackendType selected_backend,
              BlasDataType input_type, BlasDataType output_type) {
    const std::size_t stride_a = M * K;
    const std::size_t stride_b = K * N;
    const std::size_t stride_c = M * N;

    const std::size_t count_a = stride_a * BATCHES;
    const std::size_t count_b = stride_b * BATCHES;
    const std::size_t count_c = stride_c * BATCHES;

    std::vector<float> a_values(count_a);
    std::vector<float> b_values(count_b);
    std::vector<float> c_values(count_c, 0.0f);

    for (std::size_t i = 0; i < count_a; ++i)
        a_values[i] =
            static_cast<float>(static_cast<int>((i * 3) % 11) - 5) * 0.125f;

    for (std::size_t i = 0; i < count_b; ++i)
        b_values[i] =
            static_cast<float>(static_cast<int>((i * 5) % 9) - 4) * 0.1f;

    std::vector<unsigned char> host_a;
    std::vector<unsigned char> host_b;
    std::vector<unsigned char> host_c;

    encode_values(a_values, input_type, host_a);
    encode_values(b_values, input_type, host_b);
    encode_values(c_values, output_type, host_c);

    DeviceBuffer device_a(selected_backend,
                          count_a * type_size(input_type));
    DeviceBuffer device_b(selected_backend,
                          count_b * type_size(input_type));
    DeviceBuffer device_c(selected_backend,
                          count_c * type_size(output_type));

    copy_to_device(selected_backend, device_a.get(),
                   host_a.data(), host_a.size());
    copy_to_device(selected_backend, device_b.get(),
                   host_b.data(), host_b.size());
    copy_to_device(selected_backend, device_c.get(),
                   host_c.data(), host_c.size());

    backend.gemm_ex_strided_batched_device(
        Layout::ColMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        M, N, K, 1.0f,
        device_a.get(), input_type, M, stride_a,
        device_b.get(), input_type, K, stride_b,
        0.0f,
        device_c.get(), output_type, M, stride_c,
        BATCHES, nullptr);

    copy_from_device(selected_backend, host_c.data(),
                     device_c.get(), host_c.size());

    float max_error = 0.0f;

    for (int batch = 0; batch < BATCHES; ++batch) {
        for (int column = 0; column < N; ++column) {
            for (int row = 0; row < M; ++row) {
                float expected = 0.0f;

                for (int p = 0; p < K; ++p) {
                    const std::size_t index_a =
                        batch * stride_a + row + p * M;
                    const std::size_t index_b =
                        batch * stride_b + p + column * K;

                    expected +=
                        decode_value(host_a.data(), input_type, index_a) *
                        decode_value(host_b.data(), input_type, index_b);
                }

                const std::size_t index_c =
                    batch * stride_c + row + column * M;

                const float actual =
                    decode_value(host_c.data(), output_type, index_c);

                max_error = std::max(max_error,
                                     std::fabs(actual - expected));
            }
        }
    }

    const float tolerance =
        output_type == BlasDataType::Float32 ? 2.0e-4f :
        output_type == BlasDataType::Float16 ? 2.0e-2f : 8.0e-2f;

    std::printf("A/B: %-4s  C: %-4s  batches: %d  max error: %.7g  %s\n",
                type_name(input_type), type_name(output_type),
                BATCHES, max_error,
                max_error <= tolerance ? "PASS" : "FAIL");

    if (!std::isfinite(max_error) || max_error > tolerance)
        throw std::runtime_error("Numerical validation failed");
}

BackendType select_backend(int argc, char** argv) {
    BackendType selected = BackendType::CPU;

    const char* env_backend = std::getenv("CLAP_BACKEND");

    if (env_backend) {
        const std::string value(env_backend);

        if (value == "CUDA")
            selected = BackendType::CUDA;
        else if (value == "AMD")
            selected = BackendType::AMD;
    } else {
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);

            if (arg == "-cuda") {
                selected = BackendType::CUDA;
                break;
            }
            if (arg == "-rocm") {
                selected = BackendType::AMD;
                break;
            }
            if (arg == "-cpu") {
                selected = BackendType::CPU;
                break;
            }
        }
    }

    return selected;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const BackendType selected_backend = select_backend(argc, argv);

        if (selected_backend == BackendType::CPU) {
            std::fprintf(stderr,
                         "Select a GPU backend using -cuda, -rocm, "
                         "or CLAP_BACKEND=CUDA|AMD\n");
            return 1;
        }

        auto backend = BlasFactory::create(selected_backend);

        std::printf("Mixed-precision strided-batched GEMM\n");
        std::printf("Dimensions: M=%d N=%d K=%d batches=%d\n",
                    M, N, K, BATCHES);
        std::printf("Backend: %s\n",
                    selected_backend == BackendType::CUDA ? "CUDA" : "ROCm");

        run_case(*backend, selected_backend,
                 BlasDataType::Float16, BlasDataType::Float16);
        run_case(*backend, selected_backend,
                 BlasDataType::Float16, BlasDataType::Float32);
        run_case(*backend, selected_backend,
                 BlasDataType::BFloat16, BlasDataType::BFloat16);
        run_case(*backend, selected_backend,
                 BlasDataType::BFloat16, BlasDataType::Float32);
        run_case(*backend, selected_backend,
                 BlasDataType::Float32, BlasDataType::Float32);

        std::printf("All mixed-precision GEMM cases passed.\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "gemm_ex_device: %s\n", error.what());
        return 1;
    }
}
