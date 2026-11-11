
#include <blas.hh>
#include <clap/dyn_backends.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace clap;

namespace {

constexpr int M = 5;
constexpr int N = 3;
constexpr int K = 4;
constexpr int BATCHES = 3;

template <typename T>
struct RealType {
    using type = T;
};

template <typename T>
struct RealType<std::complex<T>> {
    using type = T;
};

template <typename T>
struct IsComplex : std::false_type {};

template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <typename T>
using RealTypeT = typename RealType<T>::type;

template <typename T>
std::complex<double> to_acc(const T& value) {
    if constexpr (IsComplex<T>::value) {
        return {static_cast<double>(value.real()),
                static_cast<double>(value.imag())};
    } else {
        return {static_cast<double>(value), 0.0};
    }
}

template <typename T>
T make_value(int seed) {
    using R = RealTypeT<T>;

    const R real = static_cast<R>((seed * 7) % 17 - 8) *
                   static_cast<R>(0.0625);

    if constexpr (IsComplex<T>::value) {
        const R imag = static_cast<R>((seed * 5) % 13 - 6) *
                       static_cast<R>(0.03125);
        return T(real, imag);
    } else {
        return static_cast<T>(real);
    }
}

template <typename T>
T make_scalar(double real, double imag = 0.0) {
    if constexpr (IsComplex<T>::value) {
        using R = RealTypeT<T>;
        return T(static_cast<R>(real), static_cast<R>(imag));
    } else {
        return static_cast<T>(real);
    }
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

template <typename T>
void call_batched_gemm(
    IBlasBackend& backend,
    Layout layout, Transpose ta, Transpose tb,
    int m, int n, int k, T alpha,
    const T* A, int lda, long long strideA,
    const T* B, int ldb, long long strideB,
    T beta, T* C, int ldc, long long strideC,
    int batches) {

    if constexpr (IsComplex<T>::value) {
        backend.gemm_strided_batched_device(
            layout, ta, tb, m, n, k, &alpha,
            A, lda, strideA, B, ldb, strideB,
            &beta, C, ldc, strideC, batches, nullptr);
    } else {
        backend.gemm_strided_batched_device(
            layout, ta, tb, m, n, k, alpha,
            A, lda, strideA, B, ldb, strideB,
            beta, C, ldc, strideC, batches, nullptr);
    }
}

template <typename T>
std::complex<double> get_a(
    const std::vector<T>& A, Layout layout, Transpose ta,
    int batch, int row, int inner,
    int m, int k, int lda, long long strideA) {

    const int physical_row =
        ta == Transpose::NoTrans ? row : inner;

    const int physical_column =
        ta == Transpose::NoTrans ? inner : row;

    const std::size_t index =
        static_cast<std::size_t>(batch) * strideA +
        matrix_index(layout, physical_row, physical_column, lda);

    auto value = to_acc(A[index]);

    if (ta == Transpose::ConjTrans)
        value = std::conj(value);

    return value;
}

template <typename T>
std::complex<double> get_b(
    const std::vector<T>& B, Layout layout, Transpose tb,
    int batch, int inner, int column,
    int n, int k, int ldb, long long strideB) {

    const int physical_row =
        tb == Transpose::NoTrans ? inner : column;

    const int physical_column =
        tb == Transpose::NoTrans ? column : inner;

    const std::size_t index =
        static_cast<std::size_t>(batch) * strideB +
        matrix_index(layout, physical_row, physical_column, ldb);

    auto value = to_acc(B[index]);

    if (tb == Transpose::ConjTrans)
        value = std::conj(value);

    return value;
}

template <typename T>
void run_case(
    IBlasBackend& backend,
    BackendType selected_backend,
    Layout layout, Transpose ta, Transpose tb) {

    const int a_rows = ta == Transpose::NoTrans ? M : K;
    const int a_cols = ta == Transpose::NoTrans ? K : M;

    const int b_rows = tb == Transpose::NoTrans ? K : N;
    const int b_cols = tb == Transpose::NoTrans ? N : K;

    const int lda = layout == Layout::ColMajor
                        ? a_rows + 2 : a_cols + 2;

    const int ldb = layout == Layout::ColMajor
                        ? b_rows + 2 : b_cols + 2;

    const int ldc = layout == Layout::ColMajor
                        ? M + 2 : N + 2;

    const std::size_t extentA =
        layout == Layout::ColMajor
            ? static_cast<std::size_t>(lda) * a_cols
            : static_cast<std::size_t>(a_rows) * lda;

    const std::size_t extentB =
        layout == Layout::ColMajor
            ? static_cast<std::size_t>(ldb) * b_cols
            : static_cast<std::size_t>(b_rows) * ldb;

    const std::size_t extentC =
        layout == Layout::ColMajor
            ? static_cast<std::size_t>(ldc) * N
            : static_cast<std::size_t>(M) * ldc;

    // Include padding between consecutive matrices.
    const long long strideA = static_cast<long long>(extentA + 5);
    const long long strideB = static_cast<long long>(extentB + 7);
    const long long strideC = static_cast<long long>(extentC + 9);

    std::vector<T> hostA(strideA * BATCHES);
    std::vector<T> hostB(strideB * BATCHES);
    std::vector<T> hostC(strideC * BATCHES);

    for (std::size_t i = 0; i < hostA.size(); ++i)
        hostA[i] = make_value<T>(static_cast<int>(i % 97) + 1);

    for (std::size_t i = 0; i < hostB.size(); ++i)
        hostB[i] = make_value<T>(static_cast<int>(i % 89) + 3);

    for (std::size_t i = 0; i < hostC.size(); ++i)
        hostC[i] = make_value<T>(static_cast<int>(i % 83) + 5);

    const std::vector<T> initialC = hostC;

    const T alpha = make_scalar<T>(0.75, -0.20);
    const T beta  = make_scalar<T>(-0.15, 0.10);

    DeviceBuffer deviceA(selected_backend, hostA.size() * sizeof(T));
    DeviceBuffer deviceB(selected_backend, hostB.size() * sizeof(T));
    DeviceBuffer deviceC(selected_backend, hostC.size() * sizeof(T));

    copy_to_device(selected_backend, deviceA.get(),
                   hostA.data(), hostA.size() * sizeof(T));

    copy_to_device(selected_backend, deviceB.get(),
                   hostB.data(), hostB.size() * sizeof(T));

    copy_to_device(selected_backend, deviceC.get(),
                   hostC.data(), hostC.size() * sizeof(T));

    call_batched_gemm(
        backend, layout, ta, tb,
        M, N, K, alpha,
        static_cast<const T*>(deviceA.get()), lda, strideA,
        static_cast<const T*>(deviceB.get()), ldb, strideB,
        beta,
        static_cast<T*>(deviceC.get()), ldc, strideC,
        BATCHES);

    std::vector<T> resultC(hostC.size());
    std::vector<T> resultA(hostA.size());
    std::vector<T> resultB(hostB.size());

    copy_from_device(selected_backend, resultC.data(),
                     deviceC.get(), resultC.size() * sizeof(T));

    copy_from_device(selected_backend, resultA.data(),
                     deviceA.get(), resultA.size() * sizeof(T));

    copy_from_device(selected_backend, resultB.data(),
                     deviceB.get(), resultB.size() * sizeof(T));

    // Input matrices must not be modified.
    if (resultA != hostA || resultB != hostB)
        throw std::runtime_error("GEMM modified an input matrix");

    std::vector<unsigned char> written(resultC.size(), 0);

    const auto alpha_acc = to_acc(alpha);
    const auto beta_acc = to_acc(beta);

    const double tolerance =
        std::is_same_v<RealTypeT<T>, float> ? 3.0e-4 : 2.0e-11;

    double max_error = 0.0;

    for (int batch = 0; batch < BATCHES; ++batch) {
        for (int column = 0; column < N; ++column) {
            for (int row = 0; row < M; ++row) {

                std::complex<double> sum(0.0, 0.0);

                for (int inner = 0; inner < K; ++inner) {
                    sum += get_a(hostA, layout, ta, batch,
                                 row, inner, M, K, lda, strideA) *
                           get_b(hostB, layout, tb, batch,
                                 inner, column, N, K, ldb, strideB);
                }

                const std::size_t index =
                    static_cast<std::size_t>(batch) * strideC +
                    matrix_index(layout, row, column, ldc);

                const auto expected =
                    alpha_acc * sum +
                    beta_acc * to_acc(initialC[index]);

                const auto actual = to_acc(resultC[index]);

                const double error = std::abs(actual - expected);
                const double limit =
                    tolerance * (1.0 + std::abs(expected));

                max_error = std::max(max_error, error);

                if (!std::isfinite(error) || error > limit)
                    throw std::runtime_error(
                        "Numerical mismatch in batched GEMM");

                written[index] = 1;
            }
        }
    }

    // Verify that output padding and gaps remain unchanged.
    for (std::size_t i = 0; i < resultC.size(); ++i) {
        if (!written[i] && resultC[i] != initialC[i])
            throw std::runtime_error(
                "GEMM wrote outside the output matrix region");
    }

    std::cout << "  max error = " << max_error << '\n';
}

template <typename T>
void run_precision(
    IBlasBackend& backend,
    BackendType selected_backend,
    const char* name) {

    const Layout layouts[] = {
        Layout::ColMajor,
        Layout::RowMajor
    };

    const Transpose operations[] = {
        Transpose::NoTrans,
        Transpose::Trans,
        Transpose::ConjTrans
    };

    int passed = 0;

    for (Layout layout : layouts) {
        for (Transpose ta : operations) {
            for (Transpose tb : operations) {
                run_case<T>(
                    backend, selected_backend, layout, ta, tb);
                ++passed;
            }
        }
    }

    std::cout << name << ": "
              << passed << "/18 cases passed\n";
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

        std::cout << "Typed strided-batched device GEMM\n";
        std::cout << "M=" << M << " N=" << N
                  << " K=" << K
                  << " batches=" << BATCHES << '\n';

        std::cout << "Backend: "
                  << (selected == BackendType::CUDA ? "CUDA" : "ROCm")
                  << '\n';

        run_precision<float>(*backend, selected, "FP32");
        run_precision<double>(*backend, selected, "FP64");

        run_precision<std::complex<float>>(
            *backend, selected, "Complex FP32");

        run_precision<std::complex<double>>(
            *backend, selected, "Complex FP64");

        std::cout << "\nAll typed strided-batched GEMM tests passed.\n";
        return 0;

    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "gemm_strided_batched_device: %s\n",
            error.what());

        return 1;
    }
}
