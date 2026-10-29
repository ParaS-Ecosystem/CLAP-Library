// Self-checking test of the bf16 device GEMMs (gemm_bf16_device and
// gemm_strided_batched_bf16_device) on device memory.
//
//   gemm_bf16_device [-rocm|-cuda]     (default -rocm)
//
// Operands are small integers, exact in bf16, so the expected result is
// exact. Exit status: 0 pass, 1 failure, 77 skipped (no GPU runtime).
#include <clap/blas_factory.hpp>

#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace clap;

namespace {

struct Runtime {
  int (*malloc_fn)(void**, size_t) = nullptr;
  int (*free_fn)(void*) = nullptr;
  int (*memcpy_fn)(void*, const void*, size_t, int) = nullptr;
  int (*sync_fn)() = nullptr;

  bool load(bool cuda) {
    const char* libs_hip[] = {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"};
    const char* libs_cuda[] = {"libcudart.so", "libcudart.so.12", "libcudart.so.13"};
    void* h = nullptr;
    for (const char* lib : cuda ? libs_cuda : libs_hip)
      if ((h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL)) != nullptr) break;
    if (h == nullptr) return false;
    const char* p = cuda ? "cuda" : "hip";
    auto sym = [&](const char* name) {
      return dlsym(h, (std::string(p) + name).c_str());
    };
    malloc_fn = reinterpret_cast<decltype(malloc_fn)>(sym("Malloc"));
    free_fn = reinterpret_cast<decltype(free_fn)>(sym("Free"));
    memcpy_fn = reinterpret_cast<decltype(memcpy_fn)>(sym("Memcpy"));
    sync_fn = reinterpret_cast<decltype(sync_fn)>(sym("DeviceSynchronize"));
    int count = 0;
    auto count_fn = reinterpret_cast<int (*)(int*)>(sym("GetDeviceCount"));
    return malloc_fn && free_fn && memcpy_fn && sync_fn && count_fn &&
           count_fn(&count) == 0 && count > 0;
  }
};

uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

float from_bf16(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

bool check_case(IBlasBackend& be, const Runtime& rt, Transpose ta, Transpose tb,
                int m, int n, int k, int batch, float beta) {
  const int lda = ta == Transpose::NoTrans ? k : m;
  const int ldb = tb == Transpose::NoTrans ? n : k;
  const long long sa = static_cast<long long>(m) * k, sb = static_cast<long long>(k) * n,
                  sc = static_cast<long long>(m) * n;
  std::vector<uint16_t> a(sa * batch), b(sb * batch), c(sc * batch);
  for (size_t i = 0; i < a.size(); ++i) a[i] = to_bf16(static_cast<float>(int(i * 7 % 5) - 2));
  for (size_t i = 0; i < b.size(); ++i) b[i] = to_bf16(static_cast<float>(int(i * 3 % 7) - 3));
  for (size_t i = 0; i < c.size(); ++i) c[i] = to_bf16(static_cast<float>(int(i % 3) - 1));

  std::vector<float> want(c.size());
  for (int z = 0; z < batch; ++z)
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        float acc = 0;
        for (int p = 0; p < k; ++p) {
          const float x = from_bf16(a[z * sa + (ta == Transpose::NoTrans ? i * lda + p : p * lda + i)]);
          const float y = from_bf16(b[z * sb + (tb == Transpose::NoTrans ? p * ldb + j : j * ldb + p)]);
          acc += x * y;
        }
        const size_t o = z * sc + i * n + j;
        want[o] = acc + beta * from_bf16(c[o]);
      }

  void *da, *db, *dc;
  rt.malloc_fn(&da, a.size() * 2);
  rt.malloc_fn(&db, b.size() * 2);
  rt.malloc_fn(&dc, c.size() * 2);
  rt.memcpy_fn(da, a.data(), a.size() * 2, 1);
  rt.memcpy_fn(db, b.data(), b.size() * 2, 1);
  rt.memcpy_fn(dc, c.data(), c.size() * 2, 1);
  const char* what = batch == 1 ? "gemm_bf16_device" : "gemm_strided_batched_bf16_device";
  bool ok = true;
  try {
    if (batch == 1)
      be.gemm_bf16_device(Layout::RowMajor, ta, tb, m, n, k, 1.0f, da, lda, db, ldb, beta, dc, n);
    else
      be.gemm_strided_batched_bf16_device(Layout::RowMajor, ta, tb, m, n, k, 1.0f, da, lda, sa,
                                          db, ldb, sb, beta, dc, n, sc, batch);
    rt.sync_fn();
    rt.memcpy_fn(c.data(), dc, c.size() * 2, 2);
    for (size_t i = 0; i < c.size() && ok; ++i)
      if (from_bf16(c[i]) != want[i]) {
        std::printf("FAIL %s m=%d n=%d k=%d batch=%d ta=%d tb=%d: C[%zu]=%g want %g\n", what, m,
                    n, k, batch, int(ta == Transpose::Trans), int(tb == Transpose::Trans), i,
                    from_bf16(c[i]), want[i]);
        ok = false;
      }
  } catch (const std::exception& e) {
    std::printf("FAIL %s threw: %s\n", what, e.what());
    ok = false;
  }
  rt.free_fn(da);
  rt.free_fn(db);
  rt.free_fn(dc);
  return ok;
}

}

int main(int argc, char** argv) {
  const bool cuda = argc > 1 && std::strcmp(argv[1], "-cuda") == 0;
  Runtime rt;
  if (!rt.load(cuda)) {
    std::printf("SKIP: no %s device runtime\n", cuda ? "CUDA" : "HIP");
    return 77;
  }
  std::unique_ptr<IBlasBackend> be;
  try {
    be = BlasFactory::create(cuda ? BackendType::CUDA : BackendType::AMD);
  } catch (const std::exception& e) {
    std::printf("SKIP: %s\n", e.what());
    return 77;
  }

  int run = 0, failed = 0;
  const Transpose ops[] = {Transpose::NoTrans, Transpose::Trans};
  for (Transpose ta : ops)
    for (Transpose tb : ops)
      for (int batch : {1, 3})
        for (float beta : {0.0f, 1.0f}) {
          ++run;
          if (!check_case(*be, rt, ta, tb, 33, 17, 40, batch, beta)) ++failed;
        }
  std::printf("%s: %d / %d cases pass\n", failed ? "FAIL" : "PASS", run - failed, run);
  return failed ? 1 : 0;
}
