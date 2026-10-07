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

#include "clap/lapack_dyn.hpp"
#include <dlfcn.h>
#include <iostream>
#include <mutex>

namespace clap {
namespace dyn_lapack {

lapack_int (*p_LAPACKE_sgetrf)(int,lapack_int,lapack_int,float*,lapack_int,lapack_int*)=nullptr;
lapack_int (*p_LAPACKE_dgetrf)(int,lapack_int,lapack_int,double*,lapack_int,lapack_int*)=nullptr;
lapack_int (*p_LAPACKE_sgetrs)(int,char,lapack_int,lapack_int,const float*,lapack_int,const lapack_int*,float*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_dgetrs)(int,char,lapack_int,lapack_int,const double*,lapack_int,const lapack_int*,double*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_sgesv)(int,lapack_int,lapack_int,float*,lapack_int,lapack_int*,float*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_dgesv)(int,lapack_int,lapack_int,double*,lapack_int,lapack_int*,double*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_sgetri)(int,lapack_int,float*,lapack_int,const lapack_int*)=nullptr;
lapack_int (*p_LAPACKE_dgetri)(int,lapack_int,double*,lapack_int,const lapack_int*)=nullptr;
lapack_int (*p_LAPACKE_spotrf)(int,char,lapack_int,float*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_dpotrf)(int,char,lapack_int,double*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_spotri)(int, char, lapack_int, float*, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_dpotri)(int, char, lapack_int, double*, lapack_int) = nullptr;
lapack_int (*p_LAPACKE_spotrs)(int,char,lapack_int,lapack_int,const float*,lapack_int,float*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_dpotrs)(int,char,lapack_int,lapack_int,const double*,lapack_int,double*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_sposv)(int,char,lapack_int,lapack_int,float*,lapack_int,float*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_dposv)(int,char,lapack_int,lapack_int,double*,lapack_int,double*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_sgeqrf)(int,lapack_int,lapack_int,float*,lapack_int,float*)=nullptr;
lapack_int (*p_LAPACKE_dgeqrf)(int,lapack_int,lapack_int,double*,lapack_int,double*)=nullptr;
lapack_int (*p_LAPACKE_sorgqr)(int,lapack_int,lapack_int,lapack_int,float*,lapack_int,const float*)=nullptr;
lapack_int (*p_LAPACKE_dorgqr)(int,lapack_int,lapack_int,lapack_int,double*,lapack_int,const double*)=nullptr;
lapack_int (*p_LAPACKE_sgels)(int,char,lapack_int,lapack_int,lapack_int,float*,lapack_int,float*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_dgels)(int,char,lapack_int,lapack_int,lapack_int,double*,lapack_int,double*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_ssyev)(int,char,char,lapack_int,float*,lapack_int,float*)=nullptr;
lapack_int (*p_LAPACKE_dsyev)(int,char,char,lapack_int,double*,lapack_int,double*)=nullptr;
lapack_int (*p_LAPACKE_sgeev)(int,char,char,lapack_int,float*,lapack_int,float*,float*,float*,lapack_int,float*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_dgeev)(int,char,char,lapack_int,double*,lapack_int,double*,double*,double*,lapack_int,double*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_sgesvd)(int,char,char,lapack_int,lapack_int,float*,lapack_int,float*,float*,lapack_int,float*,lapack_int,float*)=nullptr;
lapack_int (*p_LAPACKE_dgesvd)(int,char,char,lapack_int,lapack_int,double*,lapack_int,double*,double*,lapack_int,double*,lapack_int,double*)=nullptr;
lapack_int (*p_LAPACKE_strtrs)(int,char,char,char,lapack_int,lapack_int,const float*,lapack_int,float*,lapack_int)=nullptr;
lapack_int (*p_LAPACKE_dtrtrs)(int,char,char,char,lapack_int,lapack_int,const double*,lapack_int,double*,lapack_int)=nullptr;

cusolverStatus_t (*p_cusolverDnCreate)(cusolverDnHandle_t*)=nullptr;
cusolverStatus_t (*p_cusolverDnDestroy)(cusolverDnHandle_t)=nullptr;
cusolverStatus_t (*p_cusolverDnSgetrf_bufferSize)(cusolverDnHandle_t,int,int,float*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnDgetrf_bufferSize)(cusolverDnHandle_t,int,int,double*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnSgetrf)(cusolverDnHandle_t,int,int,float*,int,float*,int*,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnDgetrf)(cusolverDnHandle_t,int,int,double*,int,double*,int*,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnSgetrs)(cusolverDnHandle_t,int,int,int,const float*,int,const int*,float*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnDgetrs)(cusolverDnHandle_t,int,int,int,const double*,int,const int*,double*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnSpotrf_bufferSize)(cusolverDnHandle_t,int,int,float*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnDpotrf_bufferSize)(cusolverDnHandle_t,int,int,double*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnSpotrf)(cusolverDnHandle_t,int,int,float*,int,float*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnDpotrf)(cusolverDnHandle_t,int,int,double*,int,double*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnSpotri_bufferSize)(cusolverDnHandle_t,int,int,float*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnDpotri_bufferSize)(cusolverDnHandle_t,int,int,double*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnSpotri)(cusolverDnHandle_t,int,int,float*,int,float*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnDpotri)(cusolverDnHandle_t,int,int,double*,int,double*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnSpotrs)(cusolverDnHandle_t,int,int,int,const float*,int,float*,int,int*) = nullptr;
cusolverStatus_t (*p_cusolverDnDpotrs)(cusolverDnHandle_t,int,int,int,const double*,int,double*,int,int*) = nullptr;
cusolverStatus_t (*p_cusolverDnSgesvd_bufferSize)(cusolverDnHandle_t,int,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnDgesvd_bufferSize)(cusolverDnHandle_t,int,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnSgesvd)(cusolverDnHandle_t,char,char,int,int,float*,int,float*,float*,int,float*,int,float*,int,float*,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnDgesvd)(cusolverDnHandle_t,char,char,int,int,double*,int,double*,double*,int,double*,int,double*,int,double*,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnSsyevd_bufferSize)(cusolverDnHandle_t,int,int,int,const float*,int,const float*,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnDsyevd_bufferSize)(cusolverDnHandle_t,int,int,int,const double*,int,const double*,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnSsyevd)(cusolverDnHandle_t,int,int,int,float*,int,float*,float*,int,int*)=nullptr;
cusolverStatus_t (*p_cusolverDnDsyevd)(cusolverDnHandle_t,int,int,int,double*,int,double*,double*,int,int*)=nullptr;

int (*p_rocsolver_sgetrf)(rocsolver_handle,rocblas_int_solver,rocblas_int_solver,float*,rocblas_int_solver,rocblas_int_solver*,rocblas_int_solver*)=nullptr;
int (*p_rocsolver_dgetrf)(rocsolver_handle,rocblas_int_solver,rocblas_int_solver,double*,rocblas_int_solver,rocblas_int_solver*,rocblas_int_solver*)=nullptr;
int (*p_rocsolver_sgetrs)(rocsolver_handle,int,rocblas_int_solver,rocblas_int_solver,const float*,rocblas_int_solver,const rocblas_int_solver*,float*,rocblas_int_solver)=nullptr;
int (*p_rocsolver_dgetrs)(rocsolver_handle,int,rocblas_int_solver,rocblas_int_solver,const double*,rocblas_int_solver,const rocblas_int_solver*,double*,rocblas_int_solver)=nullptr;
int (*p_rocsolver_spotrf)(rocsolver_handle,int,rocblas_int_solver,float*,rocblas_int_solver,rocblas_int_solver*)=nullptr;
int (*p_rocsolver_dpotrf)(rocsolver_handle,int,rocblas_int_solver,double*,rocblas_int_solver,rocblas_int_solver*)=nullptr;
int (*p_rocsolver_spotri)(rocsolver_handle,int,rocblas_int_solver,float*,rocblas_int_solver,rocblas_int_solver*)=nullptr;
int (*p_rocsolver_dpotri)(rocsolver_handle,int,rocblas_int_solver,double*,rocblas_int_solver,rocblas_int_solver*)=nullptr;
int (*p_rocsolver_spotrs)(rocsolver_handle,int,rocblas_int_solver,rocblas_int_solver,const float*,rocblas_int_solver,float*,rocblas_int_solver) = nullptr;
int (*p_rocsolver_dpotrs)(rocsolver_handle,int,rocblas_int_solver,rocblas_int_solver,const double*,rocblas_int_solver,double*,rocblas_int_solver) = nullptr;
int (*p_rocsolver_sgesvd)(rocsolver_handle,int,int,rocblas_int_solver,rocblas_int_solver,float*,rocblas_int_solver,float*,float*,rocblas_int_solver,float*,rocblas_int_solver,float*,int,rocblas_int_solver*)=nullptr;
int (*p_rocsolver_dgesvd)(rocsolver_handle,int,int,rocblas_int_solver,rocblas_int_solver,double*,rocblas_int_solver,double*,double*,rocblas_int_solver,double*,rocblas_int_solver,double*,int,rocblas_int_solver*)=nullptr;
int (*p_rocsolver_ssyev)(rocsolver_handle,int,int,rocblas_int_solver,float*,rocblas_int_solver,float*,float*,rocblas_int_solver*)=nullptr;
int (*p_rocsolver_dsyev)(rocsolver_handle,int,int,rocblas_int_solver,double*,rocblas_int_solver,double*,double*,rocblas_int_solver*)=nullptr;

#define LOAD_SYM(handle, ptr, name)                                    \
    do {                                                               \
        ptr = reinterpret_cast<decltype(ptr)>(dlsym(handle, #name));  \
        if (!ptr) {                                                    \
            std::cerr << "[CLAP LAPACK] Warning: symbol " #name       \
                      << " not found\n";                              \
        }                                                              \
    } while(0)

bool loadOpenBLASLapack() {
    static bool loaded = false;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    if (loaded) return true;

    void *handle = nullptr;
    const char *libs[] = {
        "libopenblas.so",
        "libopenblas.so.0",
        "libopenblas64.so",
        "liblapacke.so",
        "liblapacke.so.3",
        nullptr
    };

    for (int i = 0; libs[i]; ++i) {
        handle = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            std::cout << "[CLAP LAPACK] Loaded " << libs[i] << "\n";
            break;
        }
    }

    if (!handle) {
        std::cerr << "[CLAP LAPACK] Failed to load OpenBLAS LAPACK: "
                  << dlerror() << "\n";
        return false;
    }

    LOAD_SYM(handle, p_LAPACKE_sgetrf, LAPACKE_sgetrf);
    LOAD_SYM(handle, p_LAPACKE_dgetrf, LAPACKE_dgetrf);
    LOAD_SYM(handle, p_LAPACKE_sgetrs, LAPACKE_sgetrs);
    LOAD_SYM(handle, p_LAPACKE_dgetrs, LAPACKE_dgetrs);
    LOAD_SYM(handle, p_LAPACKE_sgesv,  LAPACKE_sgesv);
    LOAD_SYM(handle, p_LAPACKE_dgesv,  LAPACKE_dgesv);
    LOAD_SYM(handle, p_LAPACKE_sgetri, LAPACKE_sgetri);
    LOAD_SYM(handle, p_LAPACKE_dgetri, LAPACKE_dgetri);
    LOAD_SYM(handle, p_LAPACKE_spotrf, LAPACKE_spotrf);
    LOAD_SYM(handle, p_LAPACKE_dpotrf, LAPACKE_dpotrf);
    LOAD_SYM(handle, p_LAPACKE_spotri, LAPACKE_spotri);
    LOAD_SYM(handle, p_LAPACKE_dpotri, LAPACKE_dpotri);
    LOAD_SYM(handle, p_LAPACKE_spotrs, LAPACKE_spotrs);
    LOAD_SYM(handle, p_LAPACKE_dpotrs, LAPACKE_dpotrs);
    LOAD_SYM(handle, p_LAPACKE_sposv,  LAPACKE_sposv);
    LOAD_SYM(handle, p_LAPACKE_dposv,  LAPACKE_dposv);
    LOAD_SYM(handle, p_LAPACKE_sgeqrf, LAPACKE_sgeqrf);
    LOAD_SYM(handle, p_LAPACKE_dgeqrf, LAPACKE_dgeqrf);
    LOAD_SYM(handle, p_LAPACKE_sorgqr, LAPACKE_sorgqr);
    LOAD_SYM(handle, p_LAPACKE_dorgqr, LAPACKE_dorgqr);
    LOAD_SYM(handle, p_LAPACKE_sgels,  LAPACKE_sgels);
    LOAD_SYM(handle, p_LAPACKE_dgels,  LAPACKE_dgels);
    LOAD_SYM(handle, p_LAPACKE_ssyev,  LAPACKE_ssyev);
    LOAD_SYM(handle, p_LAPACKE_dsyev,  LAPACKE_dsyev);
    LOAD_SYM(handle, p_LAPACKE_sgeev,  LAPACKE_sgeev);
    LOAD_SYM(handle, p_LAPACKE_dgeev,  LAPACKE_dgeev);
    LOAD_SYM(handle, p_LAPACKE_sgesvd, LAPACKE_sgesvd);
    LOAD_SYM(handle, p_LAPACKE_dgesvd, LAPACKE_dgesvd);
    LOAD_SYM(handle, p_LAPACKE_strtrs, LAPACKE_strtrs);
    LOAD_SYM(handle, p_LAPACKE_dtrtrs, LAPACKE_dtrtrs);

    loaded = true;
    return true;
}

bool loadCuSolver() {
    static bool loaded = false;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    if (loaded) return true;

    void *handle = nullptr;
    const char *libs[] = {
        "libcusolver.so",
        "libcusolver.so.11",
        "libcusolver.so.12",
        nullptr
    };

    for (int i = 0; libs[i]; ++i) {
        handle = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            std::cout << "[CLAP LAPACK] Loaded " << libs[i] << "\n";
            break;
        }
    }

    if (!handle) {
        std::cerr << "[CLAP LAPACK] Failed to load cuSOLVER: "
                  << dlerror() << "\n";
        return false;
    }

    LOAD_SYM(handle, p_cusolverDnCreate,  cusolverDnCreate);
    LOAD_SYM(handle, p_cusolverDnDestroy, cusolverDnDestroy);
    LOAD_SYM(handle, p_cusolverDnSgetrf_bufferSize, cusolverDnSgetrf_bufferSize);
    LOAD_SYM(handle, p_cusolverDnDgetrf_bufferSize, cusolverDnDgetrf_bufferSize);
    LOAD_SYM(handle, p_cusolverDnSgetrf,  cusolverDnSgetrf);
    LOAD_SYM(handle, p_cusolverDnDgetrf,  cusolverDnDgetrf);
    LOAD_SYM(handle, p_cusolverDnSgetrs,  cusolverDnSgetrs);
    LOAD_SYM(handle, p_cusolverDnDgetrs,  cusolverDnDgetrs);
    LOAD_SYM(handle, p_cusolverDnSpotrf_bufferSize, cusolverDnSpotrf_bufferSize);
    LOAD_SYM(handle, p_cusolverDnDpotrf_bufferSize, cusolverDnDpotrf_bufferSize);
    LOAD_SYM(handle, p_cusolverDnSpotrf,  cusolverDnSpotrf);
    LOAD_SYM(handle, p_cusolverDnDpotrf,  cusolverDnDpotrf);
    LOAD_SYM(handle, p_cusolverDnSpotri_bufferSize, cusolverDnSpotri_bufferSize);
    LOAD_SYM(handle, p_cusolverDnDpotri_bufferSize, cusolverDnDpotri_bufferSize);
    LOAD_SYM(handle, p_cusolverDnSpotri,  cusolverDnSpotri);
    LOAD_SYM(handle, p_cusolverDnDpotri,  cusolverDnDpotri);
	LOAD_SYM(handle, p_cusolverDnSpotrs, cusolverDnSpotrs);
	LOAD_SYM(handle, p_cusolverDnDpotrs, cusolverDnDpotrs);
    LOAD_SYM(handle, p_cusolverDnSgesvd_bufferSize, cusolverDnSgesvd_bufferSize);
    LOAD_SYM(handle, p_cusolverDnDgesvd_bufferSize, cusolverDnDgesvd_bufferSize);
    LOAD_SYM(handle, p_cusolverDnSgesvd,  cusolverDnSgesvd);
    LOAD_SYM(handle, p_cusolverDnDgesvd,  cusolverDnDgesvd);
    LOAD_SYM(handle, p_cusolverDnSsyevd_bufferSize, cusolverDnSsyevd_bufferSize);
    LOAD_SYM(handle, p_cusolverDnDsyevd_bufferSize, cusolverDnDsyevd_bufferSize);
    LOAD_SYM(handle, p_cusolverDnSsyevd,  cusolverDnSsyevd);
    LOAD_SYM(handle, p_cusolverDnDsyevd,  cusolverDnDsyevd);

    loaded = true;
    return true;
}

bool loadRocSolver() {
    static bool loaded = false;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    if (loaded) return true;

    void *handle = nullptr;
    const char *libs[] = {
        "librocsolver.so",
        "librocsolver.so.0",
        nullptr
    };

    for (int i = 0; libs[i]; ++i) {
        handle = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            std::cout << "[CLAP LAPACK] Loaded " << libs[i] << "\n";
            break;
        }
    }

    if (!handle) {
        std::cerr << "[CLAP LAPACK] Failed to load rocSOLVER: "
                  << dlerror() << "\n";
        return false;
    }

    LOAD_SYM(handle, p_rocsolver_sgetrf, rocsolver_sgetrf);
    LOAD_SYM(handle, p_rocsolver_dgetrf, rocsolver_dgetrf);
    LOAD_SYM(handle, p_rocsolver_sgetrs, rocsolver_sgetrs);
    LOAD_SYM(handle, p_rocsolver_dgetrs, rocsolver_dgetrs);
    LOAD_SYM(handle, p_rocsolver_spotrf, rocsolver_spotrf);
    LOAD_SYM(handle, p_rocsolver_dpotrf, rocsolver_dpotrf);
    LOAD_SYM(handle, p_rocsolver_spotri, rocsolver_spotri);
    LOAD_SYM(handle, p_rocsolver_dpotri, rocsolver_dpotri);
	LOAD_SYM(handle, p_rocsolver_spotrs, rocsolver_spotrs);
	LOAD_SYM(handle, p_rocsolver_dpotrs, rocsolver_dpotrs);
    LOAD_SYM(handle, p_rocsolver_sgesvd, rocsolver_sgesvd);
    LOAD_SYM(handle, p_rocsolver_dgesvd, rocsolver_dgesvd);
    LOAD_SYM(handle, p_rocsolver_ssyev,  rocsolver_ssyev);
    LOAD_SYM(handle, p_rocsolver_dsyev,  rocsolver_dsyev);

    loaded = true;
    return true;
}

}
}
