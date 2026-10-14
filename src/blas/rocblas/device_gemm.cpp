
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
#include <dlfcn.h>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cstdint>

namespace clap {
namespace {
template <typename T> T vendor_symbol(const char* symbol) {
  static void* library = [] {
    void* h = dlopen("librocblas.so", RTLD_LAZY | RTLD_LOCAL);
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
  if(!stream) return;
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
  auto p=params(layout,transA,transB,m,n,k,lda,ldb,ldc);
  auto op=clap::dyn::p_rocblas_sgemm;
  if(!op) throw std::runtime_error("rocBLAS sgemm unavailable");
  check((int)op(handle,p.a,p.b,p.m,p.n,p.k,&alpha,
      reinterpret_cast<const float*>(layout==Layout::RowMajor ? B:A),p.lda,
      reinterpret_cast<const float*>(layout==Layout::RowMajor ? A:B),p.ldb,
      &beta,reinterpret_cast<float*>(C),p.ldc));
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
  check((int)fn(handle,p.a,p.b,p.m,p.n,p.k,&alpha,a,14,p.lda,b,14,p.ldb,
    &beta,C,14,p.ldc,C,14,p.ldc,151,0,0,0));
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
  check((int)fn(handle,p.a,p.b,p.m,p.n,p.k,&alpha,a,p.lda,sa,
                b,p.ldb,sb,&beta,static_cast<float*>(C),p.ldc,strideC,batches));
}
}
