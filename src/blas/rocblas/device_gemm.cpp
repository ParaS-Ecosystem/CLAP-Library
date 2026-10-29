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
#include "clap/blas/rocblas/rocblas_utils.hpp"
#include "gemm_select.hpp"
#include <dlfcn.h>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cstdint>

namespace clap {
namespace {
template <typename T> T vendor_symbol(const char* symbol) {
  static void* library = [] {
    void* h = nullptr;
    for (const char* n : {"librocblas.so", "librocblas.so.5", "librocblas.so.4"})
      if (!h) h = dlopen(n, RTLD_LAZY | RTLD_LOCAL);
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
  using Fn = rocblas_status (*)(rocblas_handle,void*);
  auto fn = vendor_symbol<Fn>("rocblas_set_stream");
  if(fn(handle,stream) != rocblas_status_success)
    throw std::runtime_error("Failed to set device GEMM stream");
}

void check(int status) {if(status != 0) throw std::runtime_error("device BLAS GEMM status="+std::to_string(status));}

constexpr int kRocblasF32 = 151;
constexpr int kRocblasBF16 = 168;
constexpr int kRocblasF16 = 150;
constexpr int kRocblasI8 = 160;
constexpr int kRocblasI32 = 162;

int rocblas_type(GemmElem e) {
  switch (e) {
    case GemmElem::F16: return kRocblasF16;
    case GemmElem::BF16: return kRocblasBF16;
    case GemmElem::I8: return kRocblasI8;
    case GemmElem::I32: return kRocblasI32;
    default: return kRocblasF32;
  }
}

void check_ex_types(GemmElem ab, GemmElem c) {
  const bool fp = (ab == GemmElem::F32 || ab == GemmElem::F16 || ab == GemmElem::BF16) &&
                  (c == ab || (c == GemmElem::F32 && ab != GemmElem::F32));
  const bool i8 = ab == GemmElem::I8 && c == GemmElem::I32;
  if (!fp && !i8)
    throw std::invalid_argument("gemm_ex: C must have A/B's type or be F32 (I8 inputs: I32)");
}
}

void RocBlasBackend::gemm_device(Layout layout,Transpose transA,Transpose transB,
    int m,int n,int k,float alpha,const float* A,int lda,
    const float* B,int ldb,float beta,float* C,int ldc,void* stream) {
  validate(layout,transA,transB,m,n,k,lda,ldb,ldc,A,B,C);
  if(m==0 || n==0) return;
  set_stream(handle,stream);
  auto p=params(layout,transA,transB,m,n,k,lda,ldb,ldc);
  auto op=clap::dyn::p_rocblas_sgemm;
  if(!op) throw std::runtime_error("rocBLAS sgemm unavailable");
  const float* a=layout==Layout::RowMajor ? B:A;
  const float* b=layout==Layout::RowMajor ? A:B;
  auto roc=[&](void* c) {
    return (int)op(handle,p.a,p.b,p.m,p.n,p.k,&alpha,a,p.lda,b,p.ldb,&beta,
                   static_cast<float*>(c),p.ldc);
  };
  if(rocm_gemm::run(rocm_gemm::DType::F32,p.a,p.b,p.m,p.n,p.k,&alpha,a,p.lda,b,p.ldb,
                    &beta,C,p.ldc,1,0,0,0,stream,roc)) return;
  check(roc(C));
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
  auto p=params(layout,ta,tb,m,n,k,lda,ldb,ldc);
  using Fn=rocblas_status (*)(rocblas_handle,rocblas_operation,rocblas_operation,
    int,int,int,const void*,const void*,int,int,const void*,int,int,
    const void*,const void*,int,int,void*,int,int,int,int,int,unsigned);
  auto fn=vendor_symbol<Fn>("rocblas_gemm_ex");
  const void* a=(layout==Layout::RowMajor ? B:A);
  const void* b=(layout==Layout::RowMajor ? A:B);
  auto roc=[&](void* c) {
    return (int)fn(handle,p.a,p.b,p.m,p.n,p.k,&alpha,a,kRocblasBF16,p.lda,b,kRocblasBF16,p.ldb,
                   &beta,c,kRocblasBF16,p.ldc,c,kRocblasBF16,p.ldc,kRocblasF32,0,0,0);
  };
  if(rocm_gemm::run(rocm_gemm::DType::BF16,p.a,p.b,p.m,p.n,p.k,&alpha,a,p.lda,b,p.ldb,
                    &beta,C,p.ldc,1,0,0,0,stream,roc)) return;
  check(roc(C));
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
  auto p=params(layout,ta,tb,m,n,k,lda,ldb,ldc);
  using Fn=rocblas_status (*)(rocblas_handle,rocblas_operation,rocblas_operation,
    int,int,int,const float*,const float*,int,long long,
    const float*,int,long long,const float*,float*,int,long long,int);
  auto fn=vendor_symbol<Fn>("rocblas_sgemm_strided_batched");
  const float* a=static_cast<const float*>(layout==Layout::RowMajor ? B:A);
  const float* b=static_cast<const float*>(layout==Layout::RowMajor ? A:B);
  auto sa=layout==Layout::RowMajor ? strideB:strideA;
  auto sb=layout==Layout::RowMajor ? strideA:strideB;
  auto roc=[&](void* c) {
    return (int)fn(handle,p.a,p.b,p.m,p.n,p.k,&alpha,a,p.lda,sa,
                   b,p.ldb,sb,&beta,static_cast<float*>(c),p.ldc,strideC,batches);
  };
  if(rocm_gemm::run(rocm_gemm::DType::F32,p.a,p.b,p.m,p.n,p.k,&alpha,a,p.lda,b,p.ldb,
                    &beta,C,p.ldc,batches,sa,sb,strideC,stream,roc)) return;
  check(roc(C));
}

void RocBlasBackend::gemm_strided_batched_bf16_device(Layout layout,Transpose ta,Transpose tb,
  int m,int n,int k,float alpha,const void* A,int lda,long long strideA,
  const void* B,int ldb,long long strideB,float beta,void* C,int ldc,
  long long strideC,int batches,void* stream) {
  if(batches<0 || strideA<0 || strideB<0 || strideC<0)
    throw std::invalid_argument("Invalid batched GEMM count/stride");
  validate(layout,ta,tb,m,n,k,lda,ldb,ldc,A,B,C);
  if (batches==0 || m==0 || n==0) return;
  set_stream(handle,stream);
  auto p=params(layout,ta,tb,m,n,k,lda,ldb,ldc);
  using Fn=rocblas_status (*)(rocblas_handle,rocblas_operation,rocblas_operation,
    int,int,int,const void*,const void*,int,int,long long,const void*,int,int,long long,
    const void*,const void*,int,int,long long,void*,int,int,long long,
    int,int,int,int,unsigned);
  auto fn=vendor_symbol<Fn>("rocblas_gemm_strided_batched_ex");
  const void* a=(layout==Layout::RowMajor ? B:A);
  const void* b=(layout==Layout::RowMajor ? A:B);
  auto sa=layout==Layout::RowMajor ? strideB:strideA;
  auto sb=layout==Layout::RowMajor ? strideA:strideB;
  auto roc=[&](void* c) {
    return (int)fn(handle,p.a,p.b,p.m,p.n,p.k,&alpha,a,kRocblasBF16,p.lda,sa,
                   b,kRocblasBF16,p.ldb,sb,&beta,c,kRocblasBF16,p.ldc,strideC,
                   c,kRocblasBF16,p.ldc,strideC,batches,kRocblasF32,0,0,0);
  };
  if(rocm_gemm::run(rocm_gemm::DType::BF16,p.a,p.b,p.m,p.n,p.k,&alpha,a,p.lda,b,p.ldb,
                    &beta,C,p.ldc,batches,sa,sb,strideC,stream,roc)) return;
  check(roc(C));
}

void RocBlasBackend::gemm_ex_strided_batched_device(Layout layout,Transpose ta,Transpose tb,
  int m,int n,int k,float alpha,const void* A,int lda,long long strideA,
  const void* B,int ldb,long long strideB,float beta,void* C,int ldc,
  long long strideC,int batches,GemmElem ab,GemmElem c_type,void* stream) {
  check_ex_types(ab,c_type);
  if(batches<0 || strideA<0 || strideB<0 || strideC<0)
    throw std::invalid_argument("Invalid batched GEMM count/stride");
  validate(layout,ta,tb,m,n,k,lda,ldb,ldc,A,B,C);
  if (batches==0 || m==0 || n==0) return;
  set_stream(handle,stream);
  auto p=params(layout,ta,tb,m,n,k,lda,ldb,ldc);
  using Fn=rocblas_status (*)(rocblas_handle,rocblas_operation,rocblas_operation,
    int,int,int,const void*,const void*,int,int,long long,const void*,int,int,long long,
    const void*,const void*,int,int,long long,void*,int,int,long long,
    int,int,int,int,unsigned);
  auto fn=vendor_symbol<Fn>("rocblas_gemm_strided_batched_ex");
  const void* a=(layout==Layout::RowMajor ? B:A);
  const void* b=(layout==Layout::RowMajor ? A:B);
  auto sa=layout==Layout::RowMajor ? strideB:strideA;
  auto sb=layout==Layout::RowMajor ? strideA:strideB;
  const int t=rocblas_type(ab), tc=rocblas_type(c_type);
  if (ab == GemmElem::I8) {
    const int32_t ia = static_cast<int32_t>(alpha), ib = static_cast<int32_t>(beta);
    check((int)fn(handle,p.a,p.b,p.m,p.n,p.k,&ia,a,t,p.lda,sa,b,t,p.ldb,sb,
                  &ib,C,tc,p.ldc,strideC,C,tc,p.ldc,strideC,batches,kRocblasI32,0,0,0));
    return;
  }
  check((int)fn(handle,p.a,p.b,p.m,p.n,p.k,&alpha,a,t,p.lda,sa,b,t,p.ldb,sb,
                &beta,C,tc,p.ldc,strideC,C,tc,p.ldc,strideC,batches,kRocblasF32,0,0,0));
}

bool RocBlasBackend::gemm_scaled_device(Layout layout,Transpose ta,Transpose tb,int m,int n,int k,
  const void* A,int lda,const void* B,int ldb,void* C,int ldc,GemmElem a_type,GemmElem b_type,
  GemmElem c_type,const void* scale_a,const void* scale_b,void* stream) {
  auto lt_type=[](GemmElem e) {
    switch (e) {
      case GemmElem::F32: return 0;
      case GemmElem::F16: return 2;
      case GemmElem::BF16: return 14;
      case GemmElem::F8E4M3: return 28;
      case GemmElem::F8E5M2: return 29;
      case GemmElem::F8E4M3FNUZ: return 1000;
      case GemmElem::F8E5M2FNUZ: return 1001;
      default: return -1;
    }
  };
  const bool f8a=a_type>=GemmElem::F8E4M3, f8b=b_type>=GemmElem::F8E4M3;
  const bool out_ok=c_type==GemmElem::F32 || c_type==GemmElem::F16 || c_type==GemmElem::BF16;
  if (!f8a || !f8b || !out_ok) return false;
  validate(layout,ta,tb,m,n,k,lda,ldb,ldc,A,B,C);
  if (m==0 || n==0) return true;
  auto p=params(layout,ta,tb,m,n,k,lda,ldb,ldc);
  const bool rm=layout==Layout::RowMajor;
  return rocm_gemm::run_scaled(rm ? lt_type(b_type) : lt_type(a_type), rm ? lt_type(a_type) : lt_type(b_type),
                               lt_type(c_type), (int)p.a, (int)p.b, p.m, p.n, p.k, rm ? B : A, p.lda,
                               rm ? A : B, p.ldb, C, p.ldc, rm ? scale_b : scale_a, rm ? scale_a : scale_b,
                               stream);
}
}
