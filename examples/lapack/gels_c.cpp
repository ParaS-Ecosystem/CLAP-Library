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
using T = std::complex<float>;
using real_t = float;
static T v(int i, int j) {
  return T(real_t(0.02 * ((i + 3 * j) % 9 - 4)),
           real_t(0.01 * ((2 * i + j) % 7 - 3)));
}
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
    std::printf("Enter number of right-hand sides nrhs: ");
    if (std::scanf("%d", &nrhs) != 1 || nrhs <= 0)
      return 1;
    const int lda = n, ldb = n;
    std::printf("n = %d  nrhs = %d\n", n, nrhs);
    auto backend = LapackFactory::create(BackendType::CPU);
    int info = 0;
    std::vector<T> A((size_t)n * n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        A[(size_t)j * n + i] = (i == j) ? T(real_t(n + 3)) : v(i, j);
    const auto A0 = A;
    const int m = n;
    std::vector<T> B((size_t)n * nrhs);
    for (int j = 0; j < nrhs; ++j)
      for (int i = 0; i < n; ++i)
        B[(size_t)j * n + i] = T(real_t(1 + (i + 2 * j) % 7));
    const auto B0 = B;
    const auto start = high_resolution_clock::now();
    backend->cgels(Layout::ColMajor, Transpose::NoTrans, m, n, nrhs, A.data(),
                   lda, B.data(), ldb, &info);
    const auto end = high_resolution_clock::now();
    if (info != 0) {
      std::printf("gelsc failed: info = %d\n", info);
      return 2;
    }
    std::vector<T> expected((size_t)n * nrhs, T{});
    for (int j = 0; j < nrhs; ++j)
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
          expected[(size_t)j * n + i] +=
              A0[(size_t)k * n + i] * B[(size_t)j * n + k];
    const double err = residual(expected, B0);
    std::printf("relative solve residual = %.6e\n", err);
    if (err > 2e-3)
      return 4;
    std::printf("info = %d (0 = success)\n", info);
    std::printf("Time = %.6f seconds\n", duration<double>(end - start).count());
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "gelsc error: %s\n", e.what());
    return 3;
  }
}
