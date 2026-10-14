#include "clap/lapack_factory.hpp"
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <vector>
#include <algorithm>

using namespace clap;
using namespace std::chrono;

using T = std::complex<float>;
using real_t = float;

int main() {
    int n, nrhs = 1;
    std::printf("Enter matrix size n: ");
    if (std::scanf("%d", &n) != 1 || n <= 0) {
        std::printf("Invalid n (must be positive).\n");
        return 1;
    }
    std::printf("Enter number of right-hand sides nrhs: ");
    if (std::scanf("%d", &nrhs) != 1 || nrhs <= 0) {
        std::printf("Invalid nrhs (must be positive).\n");
        return 1;
    }

    const int lda = n;
    const int ldb = n;
    const size_t countA = static_cast<size_t>(n) * n;
    const size_t countB = static_cast<size_t>(n) * nrhs;
    std::printf("n = %d", n);
    std::printf("  nrhs = %d\n", nrhs);

    std::vector<T> A(countA);
    std::vector<T> B(countB);
    std::vector<int> ipiv(static_cast<size_t>(n));

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[i + static_cast<size_t>(j) * lda] =
                (i == j) ? T(static_cast<real_t>(n + 2), 0)
                         : T(static_cast<real_t>(0.01 * ((i + 3*j) % 11 - 5)),
                             static_cast<real_t>(0.01 * ((2*i + j) % 9 - 4)));

    for (size_t i = 0; i < countB; ++i)
        B[i] = T(static_cast<real_t>(1 + (i % 7)),
                 static_cast<real_t>((i % 3) - 1));

    auto backend = LapackFactory::create(BackendType::CPU);
    int info = 0;
    const auto start = high_resolution_clock::now();
    backend->cgesv(n, nrhs, A.data(), lda,
                          ipiv.data(), B.data(), ldb, &info);
    const auto end = high_resolution_clock::now();

    if (info != 0) {
        std::printf("cgesv failed: info = %d\n", info);
        return 1;
    }
    std::printf("info = %d (0 = success)\n", info);
    printf("  B[0] = (%.6g, %.6g)\n", static_cast<double>(B[0].real()), static_cast<double>(B[0].imag()));
    const double elapsed = duration<double>(end - start).count();
    std::printf("Time = %.6f seconds\n", elapsed);
    return 0;
}
