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

using T = std::complex<double>;
using real_t = double;

int main() {
    int n, nrhs = 1;
    std::printf("Enter matrix size n: ");
    if (std::scanf("%d", &n) != 1 || n <= 0) {
        std::printf("Invalid n (must be positive).\n");
        return 1;
    }

    const int lda = n;
    const int ldb = n;
    const size_t countA = static_cast<size_t>(n) * n;
    const size_t countB = static_cast<size_t>(n) * nrhs;
    std::printf("n = %d", n);
    std::printf("\n");

    std::vector<T> A(countA);
    std::vector<int> ipiv(static_cast<size_t>(n));

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[i + static_cast<size_t>(j) * lda] =
                (i == j) ? T(static_cast<real_t>(n + 2), 0)
                         : T(static_cast<real_t>(0.01 * ((i + 3*j) % 11 - 5)),
                             static_cast<real_t>(0.01 * ((2*i + j) % 9 - 4)));

    auto backend = LapackFactory::create(BackendType::CPU);
    int info = 0;
    const auto start = high_resolution_clock::now();
    backend->zgetrf(Layout::ColMajor, n, n, A.data(), lda, ipiv.data(), &info);
    const auto end = high_resolution_clock::now();

    if (info != 0) {
        std::printf("zgetrf failed: info = %d\n", info);
        return 1;
    }
    std::printf("info = %d (0 = success)\n", info);
    std::printf("  A[0] = (%.8g, %.8g)\n", static_cast<double>(A[0].real()), static_cast<double>(A[0].imag()));
    const double elapsed = duration<double>(end - start).count();
    std::printf("Time = %.6f seconds\n", elapsed);
    return 0;
}
