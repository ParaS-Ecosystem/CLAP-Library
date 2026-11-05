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
using T = std::complex<double>;
using real_t = double;
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

    const int lda = n, ldb = n;
    std::printf("n = %d  nrhs = %d\n", n, nrhs);
    auto backend = LapackFactory::create(BackendType::CPU);
    int info = 0;
    std::vector<T> A((size_t)n * n, T{});
    for (int j = 0; j < n; ++j)
      for (int i = 0; i <= j; ++i) {
        T x = (i == j) ? T(real_t(2 + i)) : v(i, j);
        A[(size_t)j * n + i] = x;
        A[(size_t)i * n + j] = adj(x);
      }
    const auto A0 = A;
    std::vector<real_t> w(n);
    const auto start = high_resolution_clock::now();
    backend->zheev(Layout::ColMajor, Job::Vec, Uplo::Lower, n, A.data(), lda,
                   w.data(), &info);
    const auto end = high_resolution_clock::now();
    if (info != 0) {
      std::printf("heevz failed: info = %d\n", info);
      return 2;
    }
    double err = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        T aq{};
        for (int k = 0; k < n; ++k)
          aq += A0[(size_t)k * n + i] * A[(size_t)j * n + k];
        err = std::max(err, norm(aq - T(w[j]) * A[(size_t)j * n + i]));
      }
    std::printf("eigenpair residual = %.6e\n", err);
    if (err > 2e-9 * n)
      return 4;
    std::printf("info = %d (0 = success)\n", info);
    std::printf("Time = %.6f seconds\n", duration<double>(end - start).count());
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "heevz error: %s\n", e.what());
    return 3;
  }
}
