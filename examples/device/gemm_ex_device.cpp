#include <blas.hh>
#include <clap/dyn_backends.hpp>


#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
    switch (type) {
        case BlasDataType::Float16:
        case BlasDataType::BFloat16:
            return sizeof(std::uint16_t);

        case BlasDataType::Float32:
        case BlasDataType::I32:
            return sizeof(std::uint32_t);

        case BlasDataType::I8:
        case BlasDataType::Float8E4M3:
        case BlasDataType::Float8E5M2:
        case BlasDataType::Float8E4M3Fnuz:
        case BlasDataType::Float8E5M2Fnuz:
            return sizeof(std::uint8_t);
    }
    return 0;
}

const char* type_name(BlasDataType type) {
    switch (type) {
        case BlasDataType::Float16:       return "FP16";
        case BlasDataType::BFloat16:      return "BF16";
        case BlasDataType::Float32:       return "FP32";
        case BlasDataType::I8:            return "INT8";
        case BlasDataType::I32:           return "INT32";
        case BlasDataType::Float8E4M3:    return "FP8-E4M3";
        case BlasDataType::Float8E5M2:    return "FP8-E5M2";
        case BlasDataType::Float8E4M3Fnuz:return "FP8-E4M3-FNUZ";
        case BlasDataType::Float8E5M2Fnuz:return "FP8-E5M2-FNUZ";
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

// Software E4M3 FNUZ conversion for the exactly representable
// test values used below. This avoids a compile-time HIP dependency.
std::uint8_t float_to_fp8_e4m3_fnuz(float value) {
    static constexpr std::uint8_t encodings[] = {
        0x00, 0x28, 0x30, 0x34, 0x38, 0x3A, 0x3C
    };

    if (!std::isfinite(value))
        throw std::invalid_argument("Non-finite FP8 test input");

    const float scaled = std::fabs(value) * 8.0f;
    const int magnitude = static_cast<int>(std::lround(scaled));

    if (magnitude < 0 || magnitude > 6 ||
        std::fabs(scaled - static_cast<float>(magnitude)) > 1.0e-5f) {
        throw std::invalid_argument(
            "FP8 test value is outside the supported exact-value set");
    }

    std::uint8_t encoded = encodings[magnitude];
    if (std::signbit(value) && magnitude != 0)
        encoded |= 0x80;

    return encoded;
}

float fp8_e4m3_fnuz_to_float(std::uint8_t value) {
    // FNUZ reserves 0x80 as its only NaN representation.
    if (value == 0x80)
        return std::numeric_limits<float>::quiet_NaN();

    const int exponent = (value >> 3) & 0x0F;
    const int mantissa = value & 0x07;

    const float decoded = exponent == 0
        ? std::ldexp(static_cast<float>(mantissa), -10)
        : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                     exponent - 8);

    return (value & 0x80) ? -decoded : decoded;
}

float quantize_output(float value, BlasDataType type) {
    switch (type) {
        case BlasDataType::Float32:
            return value;
        case BlasDataType::Float16:
            return fp16_to_float(float_to_fp16(value));
        case BlasDataType::BFloat16:
            return bf16_to_float(float_to_bf16(value));
        default:
            throw std::invalid_argument(
                "Unsupported FP8 GEMM output datatype");
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


std::size_t matrix_offset(
    Layout layout, int row, int column, int ld) {
    return layout == Layout::ColMajor
        ? static_cast<std::size_t>(row) +
              static_cast<std::size_t>(column) * ld
        : static_cast<std::size_t>(row) * ld + column;
}

void run_int8_case(
    IBlasBackend& backend,
    BackendType selected_backend,
    Layout layout,
    Transpose transA,
    Transpose transB,
    int alpha,
    int beta) {

    const int a_rows = transA == Transpose::NoTrans ? M : K;
    const int a_cols = transA == Transpose::NoTrans ? K : M;
    const int b_rows = transB == Transpose::NoTrans ? K : N;
    const int b_cols = transB == Transpose::NoTrans ? N : K;

    // Round physical leading dimensions up to multiples of four for
    // NVIDIA INT8 GEMM. Keep the logical matrix dimensions unchanged.
    const auto round_up_4 = [](int value) {
        return (value + 3) & ~3;
    };

    const int lda =
        round_up_4(layout == Layout::ColMajor ? a_rows : a_cols);
    const int ldb =
        round_up_4(layout == Layout::ColMajor ? b_rows : b_cols);
    const int ldc =
        round_up_4(layout == Layout::ColMajor ? M : N);

    const std::size_t stride_a =
        layout == Layout::ColMajor
            ? static_cast<std::size_t>(lda) * a_cols
            : static_cast<std::size_t>(a_rows) * lda;

    const std::size_t stride_b =
        layout == Layout::ColMajor
            ? static_cast<std::size_t>(ldb) * b_cols
            : static_cast<std::size_t>(b_rows) * ldb;

    const std::size_t stride_c =
        layout == Layout::ColMajor
            ? static_cast<std::size_t>(ldc) * N
            : static_cast<std::size_t>(M) * ldc;

    std::vector<std::int8_t> host_a(stride_a * BATCHES, 0);
    std::vector<std::int8_t> host_b(stride_b * BATCHES, 0);
    std::vector<std::int32_t> host_c(stride_c * BATCHES, 0);

    // Initialize the physically stored matrices.
    for (int batch = 0; batch < BATCHES; ++batch) {
        for (int row = 0; row < a_rows; ++row) {
            for (int column = 0; column < a_cols; ++column) {
                const int value =
                    ((batch * 7 + row * 3 + column * 5) % 11) - 5;

                host_a[batch * stride_a +
                       matrix_offset(layout, row, column, lda)] =
                    static_cast<std::int8_t>(value);
            }
        }

        for (int row = 0; row < b_rows; ++row) {
            for (int column = 0; column < b_cols; ++column) {
                const int value =
                    ((batch * 3 + row * 2 + column * 7) % 9) - 4;

                host_b[batch * stride_b +
                       matrix_offset(layout, row, column, ldb)] =
                    static_cast<std::int8_t>(value);
            }
        }

        for (int row = 0; row < M; ++row) {
            for (int column = 0; column < N; ++column) {
                const int value =
                    ((batch + row * 2 + column * 3) % 7) - 3;

                host_c[batch * stride_c +
                       matrix_offset(layout, row, column, ldc)] =
                    static_cast<std::int32_t>(value);
            }
        }
    }

    const std::vector<std::int32_t> initial_c = host_c;

    DeviceBuffer device_a(
        selected_backend, host_a.size() * sizeof(std::int8_t));
    DeviceBuffer device_b(
        selected_backend, host_b.size() * sizeof(std::int8_t));
    DeviceBuffer device_c(
        selected_backend, host_c.size() * sizeof(std::int32_t));

    copy_to_device(
        selected_backend, device_a.get(),
        host_a.data(), host_a.size() * sizeof(std::int8_t));

    copy_to_device(
        selected_backend, device_b.get(),
        host_b.data(), host_b.size() * sizeof(std::int8_t));

    copy_to_device(
        selected_backend, device_c.get(),
        host_c.data(), host_c.size() * sizeof(std::int32_t));

    backend.gemm_ex_strided_batched_device(
        layout, transA, transB,
        M, N, K, static_cast<float>(alpha),
        device_a.get(), BlasDataType::I8,
        lda, static_cast<long long>(stride_a),
        device_b.get(), BlasDataType::I8,
        ldb, static_cast<long long>(stride_b),
        static_cast<float>(beta),
        device_c.get(), BlasDataType::I32,
        ldc, static_cast<long long>(stride_c),
        BATCHES, nullptr);

    copy_from_device(
        selected_backend, host_c.data(),
        device_c.get(), host_c.size() * sizeof(std::int32_t));

    std::int64_t max_error = 0;

    for (int batch = 0; batch < BATCHES; ++batch) {
        for (int column = 0; column < N; ++column) {
            for (int row = 0; row < M; ++row) {
                std::int64_t sum = 0;

                for (int p = 0; p < K; ++p) {
                    const int a_row =
                        transA == Transpose::NoTrans ? row : p;
                    const int a_column =
                        transA == Transpose::NoTrans ? p : row;

                    const int b_row =
                        transB == Transpose::NoTrans ? p : column;
                    const int b_column =
                        transB == Transpose::NoTrans ? column : p;

                    const std::size_t index_a =
                        batch * stride_a +
                        matrix_offset(layout, a_row, a_column, lda);

                    const std::size_t index_b =
                        batch * stride_b +
                        matrix_offset(layout, b_row, b_column, ldb);

                    sum += static_cast<std::int32_t>(host_a[index_a]) *
                           static_cast<std::int32_t>(host_b[index_b]);
                }

                const std::size_t index_c =
                    batch * stride_c +
                    matrix_offset(layout, row, column, ldc);

                const std::int64_t expected =
                    static_cast<std::int64_t>(alpha) * sum +
                    static_cast<std::int64_t>(beta) *
                        initial_c[index_c];

                const std::int64_t actual = host_c[index_c];
                const std::int64_t error =
                    std::llabs(actual - expected);

                max_error = std::max(max_error, error);
            }
        }
    }

    const char* layout_name =
        layout == Layout::ColMajor ? "ColMajor" : "RowMajor";

    const char ta =
        transA == Transpose::NoTrans ? 'N' : 'T';
    const char tb =
        transB == Transpose::NoTrans ? 'N' : 'T';

    std::printf(
        "INT8->I32 %-8s %c%c alpha=%d beta=%d "
        "max error=%lld %s\n",
        layout_name, ta, tb, alpha, beta,
        static_cast<long long>(max_error),
        max_error == 0 ? "PASS" : "FAIL");

    if (max_error != 0)
        throw std::runtime_error("INT8 GEMM numerical validation failed");
}


bool run_fp8_case(
    IBlasBackend& backend,
    BackendType selected_backend,
    Transpose transA,
    Transpose transB,
    BlasDataType output_type) {

    constexpr float alpha = 1.25f;
    constexpr float beta = 0.5f;
    constexpr float scale_a = 0.5f;
    constexpr float scale_b = 0.25f;

    const int a_rows = transA == Transpose::NoTrans ? M : K;
    const int a_cols = transA == Transpose::NoTrans ? K : M;
    const int b_rows = transB == Transpose::NoTrans ? K : N;
    const int b_cols = transB == Transpose::NoTrans ? N : K;

    const int lda = a_rows + 4;
    const int ldb = b_rows + 4;
    const int ldc = M + 4;

    const std::size_t size_a = static_cast<std::size_t>(lda) * a_cols;
    const std::size_t size_b = static_cast<std::size_t>(ldb) * b_cols;
    const std::size_t size_c = static_cast<std::size_t>(ldc) * N;

    std::vector<std::uint8_t> host_a(size_a, 0);
    std::vector<std::uint8_t> host_b(size_b, 0);
    std::vector<float> initial_c(size_c, 0.0f);

    for (int row = 0; row < a_rows; ++row) {
        for (int column = 0; column < a_cols; ++column) {
            const float value =
                static_cast<float>((row * 3 + column * 5) % 13 - 6)
                * 0.125f;

            host_a[matrix_offset(
                Layout::ColMajor, row, column, lda)] =
                float_to_fp8_e4m3_fnuz(value);
        }
    }

    for (int row = 0; row < b_rows; ++row) {
        for (int column = 0; column < b_cols; ++column) {
            const float value =
                static_cast<float>((row * 7 + column * 2) % 11 - 5)
                * 0.125f;

            host_b[matrix_offset(
                Layout::ColMajor, row, column, ldb)] =
                float_to_fp8_e4m3_fnuz(value);
        }
    }

    for (int row = 0; row < M; ++row) {
        for (int column = 0; column < N; ++column) {
            initial_c[matrix_offset(
                Layout::ColMajor, row, column, ldc)] =
                static_cast<float>((row + column * 3) % 9 - 4) * 0.0625f;
        }
    }

    std::vector<unsigned char> host_c;
    encode_values(initial_c, output_type, host_c);

    // The reference must use the actual initial C values stored in
    // the selected output format, because beta multiplies that buffer.
    for (std::size_t i = 0; i < size_c; ++i)
        initial_c[i] = decode_value(host_c.data(), output_type, i);

    DeviceBuffer device_a(selected_backend, size_a);
    DeviceBuffer device_b(selected_backend, size_b);
    DeviceBuffer device_c(
        selected_backend, size_c * type_size(output_type));
    DeviceBuffer device_scale_a(selected_backend, sizeof(float));
    DeviceBuffer device_scale_b(selected_backend, sizeof(float));

    copy_to_device(selected_backend, device_a.get(),
                   host_a.data(), host_a.size());
    copy_to_device(selected_backend, device_b.get(),
                   host_b.data(), host_b.size());
    copy_to_device(selected_backend, device_c.get(),
                   host_c.data(), host_c.size());
    copy_to_device(selected_backend, device_scale_a.get(),
                   &scale_a, sizeof(scale_a));
    copy_to_device(selected_backend, device_scale_b.get(),
                   &scale_b, sizeof(scale_b));

    const bool supported = backend.gemm_scaled_device(
        Layout::ColMajor, transA, transB,
        M, N, K, alpha,
        device_a.get(), BlasDataType::Float8E4M3Fnuz, lda,
        static_cast<const float*>(device_scale_a.get()),
        device_b.get(), BlasDataType::Float8E4M3Fnuz, ldb,
        static_cast<const float*>(device_scale_b.get()),
        beta,
        device_c.get(), output_type, ldc,
        nullptr);

    const char ta = transA == Transpose::NoTrans ? 'N' : 'T';
    const char tb = transB == Transpose::NoTrans ? 'N' : 'T';

    if (!supported) {
        std::printf(
            "FP8-E4M3-FNUZ->%-4s ColMajor %c%c SKIP (no algorithm)\\n",
            type_name(output_type), ta, tb);
        return false;
    }

    copy_from_device(selected_backend, host_c.data(),
                     device_c.get(), host_c.size());

    float max_error = 0.0f;

    for (int column = 0; column < N; ++column) {
        for (int row = 0; row < M; ++row) {
            float sum = 0.0f;

            for (int q = 0; q < K; ++q) {
                const int ar = transA == Transpose::NoTrans ? row : q;
                const int ac = transA == Transpose::NoTrans ? q : row;
                const int br = transB == Transpose::NoTrans ? q : column;
                const int bc = transB == Transpose::NoTrans ? column : q;

                const std::size_t ia =
                    matrix_offset(Layout::ColMajor, ar, ac, lda);
                const std::size_t ib =
                    matrix_offset(Layout::ColMajor, br, bc, ldb);

                sum += fp8_e4m3_fnuz_to_float(host_a[ia]) *
                       fp8_e4m3_fnuz_to_float(host_b[ib]);
            }

            const std::size_t ic =
                matrix_offset(Layout::ColMajor, row, column, ldc);

            const float accumulated =
                alpha * scale_a * scale_b * sum +
                beta * initial_c[ic];

            const float expected =
                quantize_output(accumulated, output_type);
            const float actual =
                decode_value(host_c.data(), output_type, ic);

            max_error = std::max(
                max_error, std::fabs(actual - expected));
        }
    }

    const float tolerance =
        output_type == BlasDataType::Float32 ? 2.0e-3f :
        output_type == BlasDataType::Float16 ? 2.0e-3f : 1.0e-2f;

    std::printf(
        "FP8-E4M3-FNUZ->%-4s ColMajor %c%c "
        "scaleA=%.3g scaleB=%.3g max error=%.7g %s\n",
        type_name(output_type), ta, tb, scale_a, scale_b, max_error,
        std::isfinite(max_error) && max_error <= tolerance ? "PASS" : "FAIL");

    if (!std::isfinite(max_error) || max_error > tolerance)
        throw std::runtime_error("FP8 scaled GEMM numerical validation failed");

    return true;
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

        const Layout layouts[] = {
            Layout::ColMajor,
            Layout::RowMajor
        };

        const Transpose transposes[] = {
            Transpose::NoTrans,
            Transpose::Trans
        };

        const int scalar_cases[][2] = {
            {1, 0},
            {2, 1}
        };

        int int8_cases = 0;

        for (Layout layout : layouts) {
            for (Transpose ta : transposes) {
                for (Transpose tb : transposes) {
                    for (const auto& scalars : scalar_cases) {
                        run_int8_case(
                            *backend, selected_backend,
                            layout, ta, tb,
                            scalars[0], scalars[1]);

                        ++int8_cases;
                    }
                }
            }
        }

        std::printf("INT8/INT32 cases passed: %d/16\n", int8_cases);

        int fp8_cases = 0;
        int fp8_unsupported = 0;

        if (selected_backend == BackendType::AMD) {
            const BlasDataType fp8_outputs[] = {
                BlasDataType::Float32,
                BlasDataType::Float16,
                BlasDataType::BFloat16
            };

            for (BlasDataType output_type : fp8_outputs) {
                for (Transpose ta : transposes) {
                    for (Transpose tb : transposes) {
                        const bool supported = run_fp8_case(
                            *backend, selected_backend, ta, tb, output_type);

                        if (supported)
                            ++fp8_cases;
                        else
                            ++fp8_unsupported;

                        // FP32 support is already established by the
                        // baseline tests and is mandatory on this device.
                        if (!supported &&
                            output_type == BlasDataType::Float32) {
                            throw std::runtime_error(
                                "Previously working FP8 FP32 algorithm disappeared");
                        }
                    }
                }
            }

            std::printf(
                "FP8 E4M3 FNUZ: %d passed, %d unsupported, 12 attempted\n",
                fp8_cases, fp8_unsupported);
        } else {
            std::printf(
                "FP8 FNUZ scaled GEMM tests skipped: ROCm hipBLASLt only.\\n");
        }

        std::printf("All mixed-precision GEMM cases passed.\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "gemm_ex_device: %s\n", error.what());
        return 1;
    }
}
