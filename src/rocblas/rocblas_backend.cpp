// -----------------------------------------------------------------------------
// This file is part of the CLAP library — an implementation of
// high-performance linear algebra routines based on the BLAS standard.
//
// CLAP provides a unified math interface across architectures,
// enabling portable and optimized performance on CPUs.
//
// Copyright (C) The CLAP Contributors
//
// CLAP is released under the GNU Lesser General Public License (LGPL) v3.
// See the file LICENSE in the project root for full license details.
// -----------------------------------------------------------------------------


#include "clap/rocblas_backend.hpp"
#include "clap/dyn_backends.hpp"

namespace clap {

RocBlasBackend::RocBlasBackend()
    {
        if(!dyn::loadHipAndRocblas())
        {std::cout << "Using ROCBLAS backend\n";
            throw std::runtime_error(
                "Failed to load HIP/rocBLAS dynamically."
            );
        }

        if(clap_rocblas_create_handle(&handle)
            != rocblas_status_success)
        {
            throw std::runtime_error(
                "Failed to create rocBLAS handle."
            );
        }
    }
RocBlasBackend::~RocBlasBackend()
    {
        clap_rocblas_destroy_handle(handle);
    }

}