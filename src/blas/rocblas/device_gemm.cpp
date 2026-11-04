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

#include "clap/blas/rocblas/rocblas_backend.hpp"
#include "gemm_select.hpp"
#include <dlfcn.h>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cstdint>
#include <cmath>
#include <limits>

namespace clap {
namespace {

constexpr int kRocblasF16  = 150;
constexpr int kRocblasF32  = 151;
constexpr int kRocblasI8   = 160;
constexpr int kRocblasI32  = 162;
constexpr int kRocblasBF16 = 168;

int rocblas_data_type(BlasDataType type) {
  switch (type) {
    case BlasDataType::Float16:  return kRocblasF16;
    case BlasDataType::BFloat16: return kRocblasBF16;
    case BlasDataType::Float32:  return kRocblasF32;
    case BlasDataType::I8:       return kRocblasI8;
    case BlasDataType::I32:      return kRocblasI32;
    case BlasDataType::Float8E4M3:
    case BlasDataType::Float8E5M2:
    case BlasDataType::Float8E4M3Fnuz:
    case BlasDataType::Float8E5M2Fnuz:
      throw std::invalid_argument(
          "FP8 requires gemm_scaled_device");
  }
  throw std::invalid_argument("Unsupported BLAS datatype");
}

int32_t checked_int32_scalar(float value, const char* name) {
  const double converted = static_cast<double>(value);

  if (!std::isfinite(value) ||
      std::trunc(converted) != converted ||
      converted < std::numeric_limits<int32_t>::min() ||
      converted > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument(
        std::string("INT8 GEMM ") + name +
        " must be an exactly representable int32 value");
  }

  return static_cast<int32_t>(value);
}

template <typename T> T vendor_symbol(const char* symbol) {
  static void* library = [] {
    void* h = dlopen("librocblas.so", RTLD_LAZY | RTLD_LOCAL);
    if (!h) h = dlopen("librocblas.so.5", RTLD_LAZY | RTLD_LOCAL);
    if (!h) h = dlopen("librocblas.so.4", RTLD_LAZY | RTLD_LOCAL);
    if (!h) throw std::runtime_error("Unable to load rocBLAS for device GEMM");
    return h;
  }();
  auto fn = reinterpret_cast<T>(dlsym(library, symbol));
  if (!fn) throw std::runtime_error(std::string("Missing BLAS symbol: ") + symbol);
  return fn;
}

void validate(Layout layout, Transpose a, Transpose b, int m,int n,int k,
              int lda,int ldb,int ldc, const void* A,const void* B,void* C) {
  if (m<0 || n<0 || k<0) throw std::invalid_argument("Negative GEMM dimension");
  const int rowsA = (layout == Layout::ColMajor ? (a == Transpose::NoTrans ? m : k) : (a == Transpose::NoTrans ? k : m));
  const int rowsB = (layout == Layout::ColMajor ? (b == Transpose::NoTrans ? k : n) : (b == Transpose::NoTrans ? n : k));
  const int rowsC = (layout == Layout::ColMajor ? m : n);
  if(lda < std::max(1,rowsA) || ldb < std::max(1,rowsB) || ldc < std::max(1,rowsC))
    throw std::invalid_argument("Invalid GEMM leading dimension");
  if(m && n && (!A || !B || !C)) throw std::invalid_argument("Null GEMM device pointer");
}

struct Params {int m,n,k,lda,ldb,ldc; rocblas_operation a,b;};
Params params(Layout layout, Transpose transA, Transpose transB,
              int m,int n,int k,int lda,int ldb,int ldc) {
  Params p{m,n,k,lda,ldb,ldc,to_rocblas_trans(transA),to_rocblas_trans(transB)};
  if(layout == Layout::RowMajor) {
    std::swap(p.m,p.n); std::swap(p.a,p.b); std::swap(p.lda,p.ldb);
  }
  return p;
}

void set_stream(rocblas_handle handle,void *stream) {
  // Always update the handle. A nullptr stream means the default HIP stream.
  // Returning early would leave a previously assigned custom stream active.
  using Fn = rocblas_status (*)(rocblas_handle,void*);
  auto fn = vendor_symbol<Fn>("rocblas_set_stream");
  if(fn(handle,stream) != rocblas_status_success)
    throw std::runtime_error("Failed to set device GEMM stream");
}

void check(int status) {if(status != 0) throw std::runtime_error("device BLAS GEMM status="+std::to_string(status));}
}

void RocBlasBackend::gemm_device(Layout layout,Transpose transA,Transpose transB,
    int m,int n,int k,float alpha,const float* A,int lda,
    const float* B,int ldb,float beta,float* C,int ldc,void* stream) {

  validate(layout,transA,transB,m,n,k,lda,ldb,ldc,A,B,C);
  if(m==0 || n==0) return;

  set_stream(handle,stream);

  clap::rocblas_gemm_select::gemm_fp32(
      handle,layout,transA,transB,
      m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,stream);
}

void RocBlasBackend::gemm_device(Layout layout,Transpose transA,Transpose transB,
    int m,int n,int k,double alpha,const double* A,int lda,
    const double* B,int ldb,double beta,double* C,int ldc,void* stream) {
  validate(layout,transA,transB,m,n,k,lda,ldb,ldc,A,B,C);
  if(m==0 || n==0) return;
  set_stream(handle,stream);
  auto p=params(layout,transA,transB,m,n,k,lda,ldb,ldc);
  auto op=clap::dyn::p_rocblas_dgemm;
  if(!op) throw std::runtime_error("rocBLAS dgemm unavailable");
  check((int)op(handle,p.a,p.b,p.m,p.n,p.k,&alpha,
      reinterpret_cast<const double*>(layout==Layout::RowMajor ? B:A),p.lda,
      reinterpret_cast<const double*>(layout==Layout::RowMajor ? A:B),p.ldb,
      &beta,reinterpret_cast<double*>(C),p.ldc));
}

void RocBlasBackend::gemm_device(Layout layout,Transpose transA,Transpose transB,
    int m,int n,int k,const std::complex<float>* alpha,const std::complex<float>* A,int lda,
    const std::complex<float>* B,int ldb,const std::complex<float>* beta,std::complex<float>* C,int ldc,void* stream) {
  validate(layout,transA,transB,m,n,k,lda,ldb,ldc,A,B,C);
  if(m==0 || n==0) return;
  set_stream(handle,stream);
  auto p=params(layout,transA,transB,m,n,k,lda,ldb,ldc);
  auto op=clap::dyn::p_rocblas_cgemm;
  if(!op) throw std::runtime_error("rocBLAS cgemm unavailable");
  check((int)op(handle,p.a,p.b,p.m,p.n,p.k,alpha,
      reinterpret_cast<const roc_complex*>(layout==Layout::RowMajor ? B:A),p.lda,
      reinterpret_cast<const roc_complex*>(layout==Layout::RowMajor ? A:B),p.ldb,
      beta,reinterpret_cast<roc_complex*>(C),p.ldc));
}

void RocBlasBackend::gemm_device(Layout layout,Transpose transA,Transpose transB,
    int m,int n,int k,const std::complex<double>* alpha,const std::complex<double>* A,int lda,
    const std::complex<double>* B,int ldb,const std::complex<double>* beta,std::complex<double>* C,int ldc,void* stream) {
  validate(layout,transA,transB,m,n,k,lda,ldb,ldc,A,B,C);
  if(m==0 || n==0) return;
  set_stream(handle,stream);
  auto p=params(layout,transA,transB,m,n,k,lda,ldb,ldc);
  auto op=clap::dyn::p_rocblas_zgemm;
  if(!op) throw std::runtime_error("rocBLAS zgemm unavailable");
  check((int)op(handle,p.a,p.b,p.m,p.n,p.k,alpha,
      reinterpret_cast<const roc_DoubleComplex*>(layout==Layout::RowMajor ? B:A),p.lda,
      reinterpret_cast<const roc_DoubleComplex*>(layout==Layout::RowMajor ? A:B),p.ldb,
      beta,reinterpret_cast<roc_DoubleComplex*>(C),p.ldc));
}

void RocBlasBackend::gemm_bf16_device(Layout layout,Transpose ta,Transpose tb,
  int m,int n,int k,float alpha,const void* A,int lda,const void* B,int ldb,
  float beta,void* C,int ldc,void* stream) {

  validate(layout,ta,tb,m,n,k,lda,ldb,ldc,A,B,C);
  if(!m || !n) return;

  set_stream(handle,stream);

  clap::rocblas_gemm_select::gemm_bf16(
      handle,layout,ta,tb,
      m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,stream);
}

void RocBlasBackend::gemm_strided_batched_bf16_device(Layout layout, Transpose ta, Transpose tb,
     int m, int n, int k, float alpha, const void* A, int lda, long long strideA, const void* B,
     int ldb, long long strideB, float beta, void* C, int ldc, long long strideC,
     int batches, void* stream) {

    if (batches < 0 || strideA < 0 ||
        strideB < 0 || strideC < 0)
        throw std::invalid_argument("Invalid BF16 batched GEMM count/stride");

    validate(layout, ta, tb, m, n, k, lda, ldb, ldc, A, B, C);
    if (batches == 0 || m == 0 || n == 0)
        return;

    set_stream(handle, stream);
    auto p = params(layout, ta, tb, m, n, k, lda, ldb, ldc);
    using Fn = rocblas_status (*)(rocblas_handle, rocblas_operation, rocblas_operation,
                                  int, int, int, const void*, const void*, int, int,
                                  std::int64_t, const void*, int, int, std::int64_t,
                                  const void*, const void*, int, int, std::int64_t,
                                  void*, int, int, std::int64_t,
                                  int, int, int, int32_t, uint32_t );

    auto fn = vendor_symbol<Fn>("rocblas_gemm_strided_batched_ex");
    const void* a = (layout == Layout::RowMajor ? B : A);
    const void* b = (layout == Layout::RowMajor ? A : B);

    auto sa = (layout == Layout::RowMajor ? strideB : strideA);
    auto sb = (layout == Layout::RowMajor ? strideA : strideB);

    check((int)fn(handle, p.a, p.b, p.m, p.n, p.k, &alpha, a, kRocblasBF16, p.lda, sa,
                  b, kRocblasBF16, p.ldb, sb, &beta, C, kRocblasBF16, p.ldc, strideC,
                  C, kRocblasBF16, p.ldc, strideC, batches, kRocblasF32, 0, 0, 0 ));
}

void RocBlasBackend::gemm_ex_strided_batched_device(
    Layout layout, Transpose ta, Transpose tb,
    int m, int n, int k, float alpha,
    const void* A, BlasDataType typeA, int lda, long long strideA,
    const void* B, BlasDataType typeB, int ldb, long long strideB,
    float beta,
    void* C, BlasDataType typeC, int ldc, long long strideC,
    int batches, void* stream) {

  if (batches < 0 || strideA < 0 || strideB < 0 || strideC < 0)
    throw std::invalid_argument("Invalid mixed-precision batched GEMM count/stride");

  const bool int8_gemm =
      typeA == BlasDataType::I8 &&
      typeB == BlasDataType::I8 &&
      typeC == BlasDataType::I32;

  const bool floating_gemm =
      typeA == typeB &&
      (typeA == BlasDataType::Float32
           ? typeC == BlasDataType::Float32
           : ((typeA == BlasDataType::Float16 ||
               typeA == BlasDataType::BFloat16) &&
              (typeC == typeA ||
               typeC == BlasDataType::Float32)));

  if (!int8_gemm && !floating_gemm)
    throw std::invalid_argument(
        "Unsupported mixed-precision GEMM datatype combination");

  const int aType = rocblas_data_type(typeA);
  const int bType = rocblas_data_type(typeB);
  const int cType = rocblas_data_type(typeC);

  int32_t alpha_i32 = 0;
  int32_t beta_i32 = 0;

  const void* alpha_ptr = &alpha;
  const void* beta_ptr = &beta;
  const int compute_type = int8_gemm ? kRocblasI32 : kRocblasF32;

  if (int8_gemm) {
    alpha_i32 = checked_int32_scalar(alpha, "alpha");
    beta_i32 = checked_int32_scalar(beta, "beta");
    alpha_ptr = &alpha_i32;
    beta_ptr = &beta_i32;
  }

  validate(layout, ta, tb, m, n, k, lda, ldb, ldc, A, B, C);

  if (batches == 0 || m == 0 || n == 0)
    return;

  set_stream(handle, stream);
  auto p = params(layout, ta, tb, m, n, k, lda, ldb, ldc);

  using Fn = rocblas_status (*)(
      rocblas_handle, rocblas_operation, rocblas_operation,
      int, int, int, const void*,
      const void*, int, int, std::int64_t,
      const void*, int, int, std::int64_t,
      const void*, const void*, int, int, std::int64_t,
      void*, int, int, std::int64_t,
      int, int, int, int32_t, uint32_t);

  auto fn = vendor_symbol<Fn>("rocblas_gemm_strided_batched_ex");

  const void* a = (layout == Layout::RowMajor ? B : A);
  const void* b = (layout == Layout::RowMajor ? A : B);

  const int aDataType = (layout == Layout::RowMajor ? bType : aType);
  const int bDataType = (layout == Layout::RowMajor ? aType : bType);

  const auto sa = (layout == Layout::RowMajor ? strideB : strideA);
  const auto sb = (layout == Layout::RowMajor ? strideA : strideB);

  check((int)fn(handle, p.a, p.b, p.m, p.n, p.k,
                alpha_ptr,
                a, aDataType, p.lda, sa,
                b, bDataType, p.ldb, sb,
                beta_ptr,
                C, cType, p.ldc, strideC,
                C, cType, p.ldc, strideC,
                batches, compute_type, 0, 0, 0));
}

bool RocBlasBackend::gemm_scaled_device(
    Layout layout, Transpose ta, Transpose tb,
    int m, int n, int k, float alpha,
    const void* A, BlasDataType typeA, int lda,
    const float* scaleA,
    const void* B, BlasDataType typeB, int ldb,
    const float* scaleB,
    float beta,
    void* C, BlasDataType typeC, int ldc,
    void* stream) {

  const auto is_fp8 = [](BlasDataType type) {
    return type == BlasDataType::Float8E4M3 ||
           type == BlasDataType::Float8E5M2 ||
           type == BlasDataType::Float8E4M3Fnuz ||
           type == BlasDataType::Float8E5M2Fnuz;
  };

  const bool supported_output =
      typeC == BlasDataType::Float32 ||
      typeC == BlasDataType::Float16 ||
      typeC == BlasDataType::BFloat16;

  if (!is_fp8(typeA) || !is_fp8(typeB) || !supported_output) {
    return false;
  }

  validate(
      layout, ta, tb, m, n, k,
      lda, ldb, ldc, A, B, C);

  if (m == 0 || n == 0) {
    return true;
  }

  // A zero-K operation needs only beta*C. Leave it to the caller's
  // fallback path rather than requiring an FP8 hipBLASLt algorithm.
  if (k == 0 || !scaleA || !scaleB) {
    return false;
  }

  return clap::rocblas_gemm_select::gemm_scaled_fp8(
      layout, ta, tb,
      m, n, k, alpha,
      A, typeA, lda, scaleA,
      B, typeB, ldb, scaleB,
      beta,
      C, typeC, ldc,
      stream);
}

void RocBlasBackend::gemm_strided_batched_device(Layout layout,Transpose ta,Transpose tb,
  int m,int n,int k,float alpha,const void* A,int lda,long long strideA,
  const void* B,int ldb,long long strideB,float beta,void* C,int ldc,
  long long strideC,int batches,void* stream) {
  if(batches<0 || strideA<0 || strideB<0 || strideC<0)
    throw std::invalid_argument("Invalid batched GEMM count/stride");
  validate(layout,ta,tb,m,n,k,lda,ldb,ldc,A,B,C);
  if (batches==0 || m==0 || n==0) return;
  set_stream(handle,stream);

  clap::rocblas_gemm_select::gemm_fp32_strided_batched(
      handle, layout, ta, tb,
      m, n, k, alpha,
      static_cast<const float*>(A), lda, strideA,
      static_cast<const float*>(B), ldb, strideB,
      beta,
      static_cast<float*>(C), ldc, strideC,
      batches, stream);
}
}
