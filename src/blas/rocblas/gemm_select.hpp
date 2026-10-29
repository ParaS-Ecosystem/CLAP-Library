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
//
// Device GEMMs on ROCm through hipBLASLt where it is faster than rocBLAS.
//
// Neither library is fastest for every shape on MI300X: for x[m, 2048] @
// W[2048, 2048]^T in fp32, hipBLASLt takes 16 us at m = 64 where rocBLAS takes
// 37 us, while for W[4096, 2048] rocBLAS is ahead up to m = 256, and so is it
// for bf16 below m = 384. The first GEMM of a shape class therefore times
// both libraries on the caller's stream and remembers the faster one.
//
// CLAP_ROCM_GEMM=rocblas | hipblaslt | auto (default auto) overrides the
// choice; without a loadable hipBLASLt every GEMM uses rocBLAS. As in native
// PyTorch, hipBLASLt costs a one-time warm-up (library and kernel loading,
// ~5 s over the first Param2-17B prefill); rocblas avoids it for short jobs.
// CLAP_ROCM_GEMM_DEBUG=1 prints each choice.
#pragma once

#include <functional>

namespace clap {
namespace rocm_gemm {

enum class DType { F32, BF16 };

bool run(DType type, int op_a, int op_b, int m, int n, int k, const float* alpha,
         const void* A, int lda, const void* B, int ldb, const float* beta, void* C, int ldc,
         int batch, long long stride_a, long long stride_b, long long stride_c,
         void* stream, const std::function<int(void* c)>& rocblas);

bool run_scaled(int a_type, int b_type, int c_type, int op_a, int op_b, int m, int n, int k,
                const void* A, int lda, const void* B, int ldb, void* C, int ldc,
                const void* scale_a, const void* scale_b, void* stream);

}
}
