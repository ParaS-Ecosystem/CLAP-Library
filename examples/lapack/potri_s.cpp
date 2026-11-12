#include "clap/lapack_factory.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
using namespace clap;
using namespace std::chrono;
using T = float;
using real_t = float;
static T v(int i, int j) { return T(real_t(0.02 * ((i + 3 * j) % 9 - 4))); }
static double norm(const T &x) { return std::abs(x); }
template <typename V> static V adj(const V &x) { return x; }
template <typename R> static std::complex<R> adj(const std::complex<R> &x) {
  return std::conj(x);
}
static double residual(const std::vector<T> &a, const std::vector<T> &b) {
  double d = 0, base = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    d = std::max(d, norm(a[i] - b[i]));
    base = std::max(base, norm(b[i]));
  }
  return d / (1.0 + base);
}
int main() {
  try {
    int n = 0, nrhs = 1;
    std::printf("Enter matrix size n: ");
    if (std::scanf("%d", &n) != 1 || n <= 0) {
      std::printf("Invalid n.\n");
      return 1;
    }

    const int lda = n, ldb = n;
    std::printf("n = %d  nrhs = %d\n", n, nrhs);
    auto backend = LapackFactory::create(BackendType::CPU);
    int info = 0;
    std::vector<T> L((size_t)n * n, T{}), A((size_t)n * n, T{});
    for (int j = 0; j < n; ++j)
      for (int i = j; i < n; ++i)
        L[(size_t)j * n + i] = (i == j) ? T(real_t(2.0 + 0.01 * n)) : v(i, j);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
          A[(size_t)j * n + i] +=
              L[(size_t)k * n + i] * adj(L[(size_t)k * n + j]);
    const auto A0 = A;
    backend->spotrf(Layout::ColMajor, Uplo::Lower, n, A.data(), lda, &info);
    if (info != 0) {
      std::printf("setup potrf info = %d\n", info);
      return 2;
    }
    const auto start = high_resolution_clock::now();
    backend->spotri(Layout::ColMajor, Uplo::Lower, n, A.data(), lda, &info);
    const auto end = high_resolution_clock::now();
    if (info != 0) {
      std::printf("potris failed: info = %d\n", info);
      return 2;
    }
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < j; ++i)
        A[(size_t)j * n + i] = adj(A[(size_t)i * n + j]);
    std::vector<T> I((size_t)n * n, T{});
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
          I[(size_t)j * n + i] += A0[(size_t)k * n + i] * A[(size_t)j * n + k];
    double err = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        err =
            std::max(err, norm(I[(size_t)j * n + i] - (i == j ? T(1) : T(0))));
    std::printf("inverse residual = %.6e\n", err);
    if (err > 2e-3)
      return 4;
    std::printf("info = %d (0 = success)\n", info);
    std::printf("Time = %.6f seconds\n", duration<double>(end - start).count());
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "potris error: %s\n", e.what());
    return 3;
  }
}
