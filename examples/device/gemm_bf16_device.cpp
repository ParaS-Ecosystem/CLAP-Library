#include <blas.hh>
#include <clap/dyn_backends.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace clap;

namespace {

struct Shape {
    int m, n, k;
};

// A small shape and a decode-sized shape; each case runs twice so the second
// call reuses the cached GEMM plan.
constexpr Shape SHAPES[] = {
    {5, 3, 4},
    {64, 48, 96}
};

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

float make_value(int seed) {
    return static_cast<float>((seed * 7) % 17 - 8) * 0.0625f;
}

std::size_t matrix_index(Layout layout, int row, int column, int ld) {
    if (layout == Layout::ColMajor)
        return static_cast<std::size_t>(row) +
               static_cast<std::size_t>(column) * ld;

    return static_cast<std::size_t>(row) * ld + column;
}

class DeviceBuffer {
public:
    DeviceBuffer(BackendType backend, std::size_t bytes)
        : backend_(backend), pointer_(nullptr) {
        int status = -1;

        if (backend_ == BackendType::CUDA) {
            if (!dyn::p_cudaMalloc)
                throw std::runtime_error("CUDA allocation unavailable");

            status = dyn::p_cudaMalloc(&pointer_, bytes);
        } else if (backend_ == BackendType::AMD) {
            if (!dyn::p_hipMalloc)
                throw std::runtime_error("HIP allocation unavailable");

            status = dyn::p_hipMalloc(&pointer_, bytes);
        }

        if (status != 0 || !pointer_)
            throw std::runtime_error("GPU allocation failed");
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

// Runs C = alpha * op(A) * op(B) + beta * C for every batch, twice, and checks
// both results against a double-precision reference computed from the BF16
// inputs. batches == 0 calls gemm_bf16_device; otherwise
// gemm_strided_batched_bf16_device.
void run_case(IBlasBackend& backend, BackendType selected_backend,
              const Shape& shape, Layout layout, Transpose ta, Transpose tb,
              int batches) {

    const int m = shape.m, n = shape.n, k = shape.k;
    const int count = std::max(batches, 1);

    const int a_rows = ta == Transpose::NoTrans ? m : k;
    const int a_cols = ta == Transpose::NoTrans ? k : m;
    const int b_rows = tb == Transpose::NoTrans ? k : n;
    const int b_cols = tb == Transpose::NoTrans ? n : k;

    // Padded leading dimensions exercise the stride arguments.
    const int lda = (layout == Layout::ColMajor ? a_rows : a_cols) + 2;
    const int ldb = (layout == Layout::ColMajor ? b_rows : b_cols) + 1;
    const int ldc = (layout == Layout::ColMajor ? m : n) + 3;

    const long long strideA =
        static_cast<long long>(lda) * (layout == Layout::ColMajor ? a_cols : a_rows) + 5;
    const long long strideB =
        static_cast<long long>(ldb) * (layout == Layout::ColMajor ? b_cols : b_rows) + 7;
    const long long strideC =
        static_cast<long long>(ldc) * (layout == Layout::ColMajor ? n : m) + 3;

    std::vector<std::uint16_t> hostA(static_cast<std::size_t>(strideA) * count);
    std::vector<std::uint16_t> hostB(static_cast<std::size_t>(strideB) * count);
    std::vector<std::uint16_t> hostC(static_cast<std::size_t>(strideC) * count);

    for (std::size_t i = 0; i < hostA.size(); ++i)
        hostA[i] = float_to_bf16(make_value(static_cast<int>(i % 97) + 1));
    for (std::size_t i = 0; i < hostB.size(); ++i)
        hostB[i] = float_to_bf16(make_value(static_cast<int>(i % 89) + 3));
    for (std::size_t i = 0; i < hostC.size(); ++i)
        hostC[i] = float_to_bf16(make_value(static_cast<int>(i % 83) + 5));

    const float alpha = 0.75f;
    const float beta = -0.25f;

    DeviceBuffer deviceA(selected_backend, hostA.size() * sizeof(std::uint16_t));
    DeviceBuffer deviceB(selected_backend, hostB.size() * sizeof(std::uint16_t));
    DeviceBuffer deviceC(selected_backend, hostC.size() * sizeof(std::uint16_t));

    copy_to_device(selected_backend, deviceA.get(),
                   hostA.data(), hostA.size() * sizeof(std::uint16_t));
    copy_to_device(selected_backend, deviceB.get(),
                   hostB.data(), hostB.size() * sizeof(std::uint16_t));

    double max_error = 0.0;

    for (int repeat = 0; repeat < 2; ++repeat) {
        copy_to_device(selected_backend, deviceC.get(),
                       hostC.data(), hostC.size() * sizeof(std::uint16_t));

        if (batches == 0) {
            backend.gemm_bf16_device(
                layout, ta, tb, m, n, k, alpha,
                deviceA.get(), lda, deviceB.get(), ldb,
                beta, deviceC.get(), ldc, nullptr);
        } else {
            backend.gemm_strided_batched_bf16_device(
                layout, ta, tb, m, n, k, alpha,
                deviceA.get(), lda, strideA, deviceB.get(), ldb, strideB,
                beta, deviceC.get(), ldc, strideC, batches, nullptr);
        }

        std::vector<std::uint16_t> result(hostC.size());
        copy_from_device(selected_backend, result.data(),
                         deviceC.get(), result.size() * sizeof(std::uint16_t));

        for (int batch = 0; batch < count; ++batch) {
            const std::uint16_t* a = hostA.data() + batch * strideA;
            const std::uint16_t* b = hostB.data() + batch * strideB;

            for (int row = 0; row < m; ++row) {
                for (int column = 0; column < n; ++column) {
                    double sum = 0.0;

                    for (int inner = 0; inner < k; ++inner) {
                        const std::size_t ia = ta == Transpose::NoTrans
                            ? matrix_index(layout, row, inner, lda)
                            : matrix_index(layout, inner, row, lda);
                        const std::size_t ib = tb == Transpose::NoTrans
                            ? matrix_index(layout, inner, column, ldb)
                            : matrix_index(layout, column, inner, ldb);

                        sum += static_cast<double>(bf16_to_float(a[ia])) *
                               static_cast<double>(bf16_to_float(b[ib]));
                    }

                    const std::size_t ic =
                        batch * strideC + matrix_index(layout, row, column, ldc);
                    const double expected =
                        alpha * sum + beta * bf16_to_float(hostC[ic]);
                    const double error =
                        std::abs(bf16_to_float(result[ic]) - expected) /
                        std::max(1.0, std::abs(expected));

                    max_error = std::max(max_error, error);
                }
            }
        }

        // Elements outside the matrices (padding) must be left unchanged.
        for (int batch = 0; batch < count; ++batch) {
            for (int row = 0; row < (layout == Layout::ColMajor ? ldc : m); ++row) {
                for (int column = 0; column < (layout == Layout::ColMajor ? n : ldc); ++column) {
                    const bool inside = layout == Layout::ColMajor
                        ? row < m : column < n;
                    if (inside)
                        continue;

                    const std::size_t ic =
                        batch * strideC + matrix_index(layout, row, column, ldc);
                    if (result[ic] != hostC[ic])
                        throw std::runtime_error("BF16 GEMM wrote outside C");
                }
            }
        }
    }

    // BF16 output has 8 significant bits.
    const double tolerance = 1.0e-2;

    std::printf("%s %c%c  M=%-3d N=%-3d K=%-3d %-9s max error: %.3g  %s\n",
                layout == Layout::ColMajor ? "col" : "row",
                ta == Transpose::NoTrans ? 'N' : 'T',
                tb == Transpose::NoTrans ? 'N' : 'T',
                m, n, k,
                batches == 0 ? "gemm" : (batches == 1 ? "batches=1" : "batches=3"),
                max_error, max_error <= tolerance ? "PASS" : "FAIL");

    if (!(max_error <= tolerance))
        throw std::runtime_error("BF16 device GEMM result outside tolerance");
}

BackendType select_backend(int argc, char** argv) {
    BackendType selected = BackendType::CPU;

    const char* env = std::getenv("CLAP_BACKEND");

    if (env) {
        const std::string value(env);

        if (value == "CUDA")
            selected = BackendType::CUDA;
        else if (value == "AMD" || value == "ROCM")
            selected = BackendType::AMD;
    }

    if (!env) {
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);

            if (arg == "-cuda")
                selected = BackendType::CUDA;
            else if (arg == "-rocm")
                selected = BackendType::AMD;
            else if (arg == "-cpu")
                selected = BackendType::CPU;
        }
    }

    return selected;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const BackendType selected = select_backend(argc, argv);

        if (selected == BackendType::CPU)
            throw std::runtime_error(
                "Select a GPU backend using -cuda, -rocm, "
                "or CLAP_BACKEND=CUDA|AMD");

        auto backend = BlasFactory::create(selected);

        std::cout << "BF16 device GEMM\n";
        std::cout << "Backend: "
                  << (selected == BackendType::CUDA ? "CUDA" : "ROCm")
                  << '\n';

        const Layout layouts[] = {Layout::ColMajor, Layout::RowMajor};
        const Transpose transposes[] = {Transpose::NoTrans, Transpose::Trans};
        const int batch_counts[] = {0, 1, 3};

        int cases = 0;

        for (const Shape& shape : SHAPES)
            for (Layout layout : layouts)
                for (Transpose ta : transposes)
                    for (Transpose tb : transposes)
                        for (int batches : batch_counts) {
                            run_case(*backend, selected, shape, layout, ta, tb, batches);
                            ++cases;
                        }

        std::cout << "\nAll " << cases << " BF16 device GEMM cases passed.\n";
        return 0;

    } catch (const std::exception& error) {
        std::fprintf(stderr, "gemm_bf16_device: %s\n", error.what());
        return 1;
    }
}
