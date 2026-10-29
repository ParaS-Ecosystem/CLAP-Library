// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// This file is part of the CLAP library, a component of the ParaS Ecosystem.
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License (LGPL) version 3
// as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library. If not, see <https://www.gnu.org/licenses/>.
// -----------------------------------------------------------------------------

#include "gemm_select.hpp"

#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>

namespace clap {
namespace rocm_gemm {
namespace {

using LtHandle = void*;
using LtDesc = void*;
using LtLayout = void*;
using LtPref = void*;
struct LtAlgo {
  uint8_t data[16];
  size_t max_workspace_bytes;
};
struct LtHeuristic {
  LtAlgo algo;
  size_t workspace_size;
  int state;
  float waves;
  int reserved[4];
};
constexpr int kOpN = 111, kOpT = 112;
constexpr int kCompute32F = 2;
constexpr int kR32F = 0, kR16BF = 14;
constexpr int kDescTransA = 0, kDescTransB = 1;
constexpr int kDescScaleA = 5, kDescScaleB = 6;
constexpr int kPrefMaxWorkspace = 1;
constexpr int kLayoutBatchCount = 0, kLayoutBatchStride = 1;
constexpr size_t kWorkspace = size_t{32} << 20;

struct Runtime {
  int (*create)(LtHandle*) = nullptr;
  int (*desc_create)(LtDesc*, int, int) = nullptr;
  int (*desc_set)(LtDesc, int, const void*, size_t) = nullptr;
  int (*layout_create)(LtLayout*, int, uint64_t, uint64_t, int64_t) = nullptr;
  int (*layout_set)(LtLayout, int, const void*, size_t) = nullptr;
  int (*pref_create)(LtPref*) = nullptr;
  int (*pref_set)(LtPref, int, const void*, size_t) = nullptr;
  int (*pref_destroy)(LtPref) = nullptr;
  int (*heuristic)(LtHandle, LtDesc, LtLayout, LtLayout, LtLayout, LtLayout, LtPref, int,
                   LtHeuristic*, int*) = nullptr;
  int (*matmul)(LtHandle, LtDesc, const void*, const void*, LtLayout, const void*, LtLayout,
                const void*, const void*, LtLayout, void*, LtLayout, const LtAlgo*, void*,
                size_t, void*) = nullptr;
  int (*get_device)(int*) = nullptr;
  int (*malloc_fn)(void**, size_t) = nullptr;
  int (*free_fn)(void*) = nullptr;
  int (*memset_async)(void*, int, size_t, void*) = nullptr;
  int (*event_create)(void**) = nullptr;
  int (*event_destroy)(void*) = nullptr;
  int (*event_record)(void*, void*) = nullptr;
  int (*event_sync)(void*) = nullptr;
  int (*event_elapsed)(float*, void*, void*) = nullptr;
  int (*is_capturing)(void*, int*) = nullptr;
  bool ok = false;

  Runtime() {
    void* lt = open({"libhipblaslt.so", "libhipblaslt.so.1", "libhipblaslt.so.0"});
    void* hip = open({"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"});
    if (!lt || !hip) return;
    ok = sym(lt, "hipblasLtCreate", create) && sym(lt, "hipblasLtMatmulDescCreate", desc_create) &&
         sym(lt, "hipblasLtMatmulDescSetAttribute", desc_set) &&
         sym(lt, "hipblasLtMatrixLayoutCreate", layout_create) &&
         sym(lt, "hipblasLtMatrixLayoutSetAttribute", layout_set) &&
         sym(lt, "hipblasLtMatmulPreferenceCreate", pref_create) &&
         sym(lt, "hipblasLtMatmulPreferenceSetAttribute", pref_set) &&
         sym(lt, "hipblasLtMatmulPreferenceDestroy", pref_destroy) &&
         sym(lt, "hipblasLtMatmulAlgoGetHeuristic", heuristic) &&
         sym(lt, "hipblasLtMatmul", matmul) && sym(hip, "hipGetDevice", get_device) &&
         sym(hip, "hipMalloc", malloc_fn) && sym(hip, "hipFree", free_fn) &&
         sym(hip, "hipMemsetAsync", memset_async) &&
         sym(hip, "hipEventCreate", event_create) && sym(hip, "hipEventDestroy", event_destroy) &&
         sym(hip, "hipEventRecord", event_record) && sym(hip, "hipEventSynchronize", event_sync) &&
         sym(hip, "hipEventElapsedTime", event_elapsed) &&
         sym(hip, "hipStreamIsCapturing", is_capturing);
  }
  static void* open(std::initializer_list<const char*> names) {
    for (const char* n : names)
      if (void* h = dlopen(n, RTLD_LAZY | RTLD_LOCAL)) return h;
    return nullptr;
  }
  template <typename F> static bool sym(void* lib, const char* name, F& fn) {
    fn = reinterpret_cast<F>(dlsym(lib, name));
    return fn != nullptr;
  }
};

bool debug() {
  static const bool on = [] {
    const char* v = std::getenv("CLAP_ROCM_GEMM_DEBUG");
    return v && *v && std::strcmp(v, "0") != 0;
  }();
  return on;
}

Runtime& runtime() {
  static Runtime rt;
  static const bool reported = [] {
    if (debug()) std::fprintf(stderr, "[clap rocm gemm] hipBLASLt %s\n", rt.ok ? "loaded" : "unavailable");
    return true;
  }();
  (void)reported;
  return rt;
}

enum class Mode { Auto, RocBlas, HipBlasLt };

Mode mode() {
  static const Mode m = [] {
    const char* v = std::getenv("CLAP_ROCM_GEMM");
    if (v && std::strcmp(v, "rocblas") == 0) return Mode::RocBlas;
    if (v && std::strcmp(v, "hipblaslt") == 0) return Mode::HipBlasLt;
    return Mode::Auto;
  }();
  return m;
}

struct Device {
  LtHandle handle = nullptr;
  void* workspace = nullptr;
  void* scratch = nullptr;
  size_t scratch_bytes = 0;
  bool ok = false;
};

struct Shape {
  LtLayout a = nullptr, b = nullptr, c = nullptr;
  LtAlgo own{};
  int own_state = 0;
};
struct ClassAlgo {
  LtAlgo algo{};
  bool ok = false;
};

using DescKey = std::tuple<int, int, int>;
using ShapeKey = std::tuple<int, int, int, int, int, int, int, int, int, int, int, long long,
                            long long, long long>;
using ClassKey = std::tuple<int, int, int, int, int, int, int, int>;

std::mutex mutex;
std::map<int, Device> devices;
std::map<DescKey, LtDesc> descs;
std::map<ShapeKey, Shape> shapes;
std::map<ClassKey, ClassAlgo> algos;
std::map<ClassKey, bool> choices;

int size_class(int m) {
  if (m <= 16) return m;
  int p = 16;
  while (p * 2 < m) p *= 2;
  return m <= p + p / 2 ? p + p / 2 : 2 * p;
}

Device* device(Runtime& rt, int dev) {
  auto it = devices.find(dev);
  if (it != devices.end()) return it->second.ok ? &it->second : nullptr;
  Device& d = devices[dev];
  d.ok = rt.create(&d.handle) == 0 && rt.malloc_fn(&d.workspace, kWorkspace) == 0;
  return d.ok ? &d : nullptr;
}

LtDesc desc(Runtime& rt, int type, int op_a, int op_b) {
  auto it = descs.find(DescKey{type, op_a, op_b});
  if (it != descs.end()) return it->second;
  LtDesc d = nullptr;
  if (rt.desc_create(&d, kCompute32F, kR32F) != 0) d = nullptr;
  if (d) {
    rt.desc_set(d, kDescTransA, &op_a, sizeof(op_a));
    rt.desc_set(d, kDescTransB, &op_b, sizeof(op_b));
  }
  descs.emplace(DescKey{type, op_a, op_b}, d);
  return d;
}

Shape* shape(Runtime& rt, const ShapeKey& key, int type, int op_a, int op_b, int m, int n, int k,
             int lda, int ldb, int ldc, int batch, long long sa, long long sb, long long sc) {
  auto it = shapes.find(key);
  if (it != shapes.end()) return it->second.c ? &it->second : nullptr;
  Shape& s = shapes[key];
  const bool ta = op_a != kOpN, tb = op_b != kOpN;
  if (rt.layout_create(&s.a, type, ta ? k : m, ta ? m : k, lda) != 0 ||
      rt.layout_create(&s.b, type, tb ? n : k, tb ? k : n, ldb) != 0 ||
      rt.layout_create(&s.c, type, m, n, ldc) != 0) {
    s.c = nullptr;
    return nullptr;
  }
  if (batch > 1) {
    const int32_t count = batch;
    const int64_t strides[3] = {sa, sb, sc};
    LtLayout layouts[3] = {s.a, s.b, s.c};
    for (int i = 0; i < 3; ++i)
      if (rt.layout_set(layouts[i], kLayoutBatchCount, &count, sizeof(count)) != 0 ||
          rt.layout_set(layouts[i], kLayoutBatchStride, &strides[i], sizeof(strides[i])) != 0) {
        s.c = nullptr;
        return nullptr;
      }
  }
  return &s;
}

bool heuristic(Runtime& rt, Device& d, LtDesc dsc, const Shape& s, LtAlgo& out) {
  LtPref pref = nullptr;
  if (rt.pref_create(&pref) != 0) return false;
  const uint64_t ws = kWorkspace;
  rt.pref_set(pref, kPrefMaxWorkspace, &ws, sizeof(ws));
  LtHeuristic h{};
  int found = 0;
  const int st = rt.heuristic(d.handle, dsc, s.a, s.b, s.c, s.c, pref, 1, &h, &found);
  rt.pref_destroy(pref);
  if (st != 0 || found < 1 || h.state != 0) return false;
  out = h.algo;
  return true;
}

int lt_gemm(Runtime& rt, Device& d, LtDesc dsc, const Shape& s, const LtAlgo& algo,
            const float* alpha, const void* A, const void* B, const float* beta, void* C,
            void* stream) {
  return rt.matmul(d.handle, dsc, alpha, A, s.a, B, s.b, beta, C, s.c, C, s.c, &algo,
                   d.workspace, kWorkspace, stream);
}

float time_us(Runtime& rt, void* stream, void* e0, void* e1, const std::function<int()>& fn) {
  float ms = -1;
  bool ok = rt.event_record(e0, stream) == 0;
  for (int i = 0; ok && i < 3; ++i) ok = fn() == 0;
  if (ok && rt.event_record(e1, stream) == 0 && rt.event_sync(e1) == 0 &&
      rt.event_elapsed(&ms, e0, e1) == 0)
    return ms * 1000.0f / 3;
  return -1;
}

std::pair<float, float> compare(Runtime& rt, void* stream, const std::function<int()>& lt,
                                const std::function<int()>& roc) {
  void *e0 = nullptr, *e1 = nullptr;
  if (rt.event_create(&e0) != 0 || rt.event_create(&e1) != 0) return {-1, -1};
  float best_lt = -1, best_roc = -1;
  if (lt() == 0 && lt() == 0 && roc() == 0 && roc() == 0) {
    for (int r = 0; r < 3; ++r) {
      const float a = time_us(rt, stream, e0, e1, lt), b = time_us(rt, stream, e0, e1, roc);
      if (a > 0 && (best_lt < 0 || a < best_lt)) best_lt = a;
      if (b > 0 && (best_roc < 0 || b < best_roc)) best_roc = b;
    }
  }
  rt.event_destroy(e0);
  rt.event_destroy(e1);
  return {best_lt, best_roc};
}

using ScaledKey = std::tuple<int, int, int, int, int, int, int, int, int, int, int, int>;
struct Scaled {
  LtDesc desc = nullptr;
  LtLayout a = nullptr, b = nullptr, c = nullptr;
  LtAlgo algo{};
  bool ok = false;
};
std::map<ScaledKey, Scaled> scaled;

}

bool run_scaled(int a_type, int b_type, int c_type, int op_a, int op_b, int m, int n, int k,
                const void* A, int lda, const void* B, int ldb, void* C, int ldc,
                const void* scale_a, const void* scale_b, void* stream) {
  if (a_type < 0 || b_type < 0 || c_type < 0 || m <= 0 || n <= 0 || k <= 0) return false;
  Runtime& rt = runtime();
  if (!rt.ok) return false;
  std::lock_guard<std::mutex> lock(mutex);
  int dev = 0;
  if (rt.get_device(&dev) != 0) return false;
  Device* d = device(rt, dev);
  if (!d) return false;
  const ScaledKey key{dev, a_type, b_type, c_type, op_a, op_b, m, n, k, lda, ldb, ldc};
  auto it = scaled.find(key);
  if (it == scaled.end()) {
    Scaled s;
    const bool ta = op_a != kOpN, tb = op_b != kOpN;
    s.ok = rt.desc_create(&s.desc, kCompute32F, kR32F) == 0 &&
           rt.desc_set(s.desc, kDescTransA, &op_a, sizeof(op_a)) == 0 &&
           rt.desc_set(s.desc, kDescTransB, &op_b, sizeof(op_b)) == 0 &&
           rt.desc_set(s.desc, kDescScaleA, &scale_a, sizeof(scale_a)) == 0 &&
           rt.desc_set(s.desc, kDescScaleB, &scale_b, sizeof(scale_b)) == 0 &&
           rt.layout_create(&s.a, a_type, ta ? k : m, ta ? m : k, lda) == 0 &&
           rt.layout_create(&s.b, b_type, tb ? n : k, tb ? k : n, ldb) == 0 &&
           rt.layout_create(&s.c, c_type, m, n, ldc) == 0;
    if (s.ok) {
      Shape sh;
      sh.a = s.a;
      sh.b = s.b;
      sh.c = s.c;
      s.ok = heuristic(rt, *d, s.desc, sh, s.algo);
    }
    if (debug())
      std::fprintf(stderr, "[clap rocm gemm] fp8 types=%d,%d->%d op=%d,%d m=%d n=%d k=%d: %s\n", a_type, b_type,
                   c_type, op_a, op_b, m, n, k, s.ok ? "hipBLASLt algorithm" : "none");
    it = scaled.emplace(key, s).first;
  }
  Scaled& s = it->second;
  if (!s.ok) return false;
  rt.desc_set(s.desc, kDescScaleA, &scale_a, sizeof(scale_a));
  rt.desc_set(s.desc, kDescScaleB, &scale_b, sizeof(scale_b));
  const float one = 1.f, zero = 0.f;
  return rt.matmul(d->handle, s.desc, &one, A, s.a, B, s.b, &zero, C, s.c, C, s.c, &s.algo,
                   d->workspace, kWorkspace, stream) == 0;
}

bool run(DType type, int op_a, int op_b, int m, int n, int k, const float* alpha, const void* A,
         int lda, const void* B, int ldb, const float* beta, void* C, int ldc, int batch,
         long long stride_a, long long stride_b, long long stride_c, void* stream,
         const std::function<int(void* c)>& rocblas) {
  if (mode() == Mode::RocBlas || m <= 0 || n <= 0 || k <= 0 || batch <= 0) return false;
  if (batch == 1) stride_a = stride_b = stride_c = 0;
  Runtime& rt = runtime();
  if (!rt.ok) return false;
  const int dt = type == DType::F32 ? kR32F : kR16BF;
  std::lock_guard<std::mutex> lock(mutex);
  int dev = 0;
  if (rt.get_device(&dev) != 0) return false;
  const ClassKey ckey{dev, dt, op_a, op_b, size_class(m), size_class(n), k, batch};
  auto choice = choices.find(ckey);
  if (mode() == Mode::Auto && choice != choices.end() && !choice->second) return false;
  Device* d = device(rt, dev);
  if (!d) return false;
  LtDesc dsc = desc(rt, dt, op_a, op_b);
  if (!dsc) return false;
  Shape* s = shape(rt, ShapeKey{dev, dt, op_a, op_b, m, n, k, lda, ldb, ldc, batch, stride_a,
                                stride_b, stride_c},
                   dt, op_a, op_b, m, n, k, lda, ldb, ldc, batch, stride_a, stride_b, stride_c);
  if (!s) return false;
  ClassAlgo& ca = algos[ckey];
  if (!ca.ok && !(ca.ok = heuristic(rt, *d, dsc, *s, ca.algo))) {
    if (debug())
      std::fprintf(stderr, "[clap rocm gemm] no hipBLASLt algorithm: type=%d op=%d,%d m=%d n=%d k=%d\n",
                   dt, op_a, op_b, m, n, k);
    return false;
  }
  auto gemm = [&](void* out) {
    if (s->own_state == 0) {
      const int st = lt_gemm(rt, *d, dsc, *s, ca.algo, alpha, A, B, beta, out, stream);
      if (st == 0) return 0;
      s->own_state = heuristic(rt, *d, dsc, *s, s->own) ? 1 : -1;
      if (debug())
        std::fprintf(stderr, "[clap rocm gemm] class algorithm rejected (status %d): m=%d n=%d k=%d, own %s\n",
                     st, m, n, k, s->own_state > 0 ? "found" : "none");
    }
    return s->own_state > 0 ? lt_gemm(rt, *d, dsc, *s, s->own, alpha, A, B, beta, out, stream) : -1;
  };
  if (mode() == Mode::Auto && choice == choices.end()) {
    int capturing = 0;
    if (rt.is_capturing(stream, &capturing) != 0 || capturing != 0) return false;
    const size_t elems = batch > 1 ? size_t(stride_c) * (batch - 1) + size_t(ldc) * n
                                   : size_t(ldc) * n;
    const size_t bytes = elems * (type == DType::F32 ? 4 : 2);
    if (d->scratch_bytes < bytes) {
      if (d->scratch) rt.free_fn(d->scratch);
      d->scratch = nullptr;
      d->scratch_bytes = 0;
      if (rt.malloc_fn(&d->scratch, bytes) != 0) return false;
      if (rt.memset_async(d->scratch, 0, bytes, stream) != 0) {
        rt.free_fn(d->scratch);
        d->scratch = nullptr;
        return false;
      }
      d->scratch_bytes = bytes;
    }
    const auto [lt, roc] = compare(rt, stream, [&] { return gemm(d->scratch); },
                                   [&] { return rocblas(d->scratch); });
    choice = choices.emplace(ckey, lt > 0 && (roc < 0 || lt < 0.97f * roc)).first;
    if (debug())
      std::fprintf(stderr, "[clap rocm gemm] choice type=%d op=%d,%d m=%d (class %d) n=%d (class %d) k=%d batch=%d: "
                   "hipBLASLt %.1f us, rocBLAS %.1f us -> %s\n", dt, op_a, op_b, m, size_class(m), n,
                   size_class(n), k, batch, lt, roc, choice->second ? "hipBLASLt" : "rocBLAS");
    if (!choice->second) return false;
  }
  return gemm(C) == 0;
}

}
}
