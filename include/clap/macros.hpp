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

//==========*Cublas*=========
#define clap_cudaMalloc clap::dyn::p_cudaMalloc
#define clap_cudaFree clap::dyn::p_cudaFree
#define clap_cudaMemcpy clap::dyn::p_cudaMemcpy
#define clap_cublasSetPointerMode clap::dyn::p_cublasSetPointerMode_v2
#define clap_cudaDeviceSynchronize clap::dyn::p_cudaDeviceSynchronize
#define clap_cublasCreate clap::dyn::p_cublasCreate_v2
#define clap_cublasDestroy clap::dyn::p_cublasDestroy_v2
#define clap_cudaMemset clap::dyn::p_cudaMemset

//Level 1
#define clap_cublasScopy clap::dyn::p_cublasScopy_v2
#define clap_cublasDcopy clap::dyn::p_cublasDcopy_v2
#define clap_cublasSscal clap::dyn::p_cublasSscal_v2
#define clap_cublasDscal clap::dyn::p_cublasDscal_v2
#define clap_cublasSaxpy clap::dyn::p_cublasSaxpy_v2
#define clap_cublasDaxpy clap::dyn::p_cublasDaxpy_v2
#define clap_cublasIsamax clap::dyn::p_cublasIsamax_v2
#define clap_cublasIdamax clap::dyn::p_cublasIdamax_v2
#define clap_cublasSswap clap::dyn::p_cublasSswap_v2
#define clap_cublasDswap clap::dyn::p_cublasDswap_v2
#define clap_cublasSdot clap::dyn::p_cublasSdot_v2
#define clap_cublasDdot clap::dyn::p_cublasDdot_v2
#define clap_cublasSnrm2 clap::dyn::p_cublasSnrm2_v2
#define clap_cublasDnrm2 clap::dyn::p_cublasDnrm2_v2
#define clap_cublasSasum clap::dyn::p_cublasSasum_v2
#define clap_cublasDasum clap::dyn::p_cublasDasum_v2
#define clap_cublasSrot clap::dyn::p_cublasSrot_v2
#define clap_cublasDrot clap::dyn::p_cublasDrot_v2
#define clap_cublasSrotg clap::dyn::p_cublasSrotg_v2
#define clap_cublasDrotg clap::dyn::p_cublasDrotg_v2
#define clap_cublasSrotm clap::dyn::p_cublasSrotm_v2
#define clap_cublasDrotm clap::dyn::p_cublasDrotm_v2
#define clap_cublasSrotmg clap::dyn::p_cublasSrotmg_v2
#define clap_cublasDrotmg clap::dyn::p_cublasDrotmg_v2

#define clap_cublasIcamax clap::dyn::p_cublasIcamax_v2
#define clap_cublasIzamax clap::dyn::p_cublasIzamax_v2
#define clap_cublasScasum clap::dyn::p_cublasScasum_v2
#define clap_cublasDzasum clap::dyn::p_cublasDzasum_v2

#define clap_cublasCaxpy clap::dyn::p_cublasCaxpy_v2
#define clap_cublasZaxpy clap::dyn::p_cublasZaxpy_v2

#define clap_cublasCcopy clap::dyn::p_cublasCcopy_v2
#define clap_cublasZcopy clap::dyn::p_cublasZcopy_v2

#define clap_cublasCdotu clap::dyn::p_cublasCdotu_v2
#define clap_cublasCdotc clap::dyn::p_cublasCdotc_v2
#define clap_cublasZdotu clap::dyn::p_cublasZdotu_v2
#define clap_cublasZdotc clap::dyn::p_cublasZdotc_v2

#define clap_cublasScnrm2 clap::dyn::p_cublasScnrm2_v2
#define clap_cublasDznrm2 clap::dyn::p_cublasDznrm2_v2

#define clap_cublasCscal clap::dyn::p_cublasCscal_v2
#define clap_cublasZscal clap::dyn::p_cublasZscal_v2

#define clap_cublasCsscal clap::dyn::p_cublasCsscal_v2
#define clap_cublasZdscal clap::dyn::p_cublasZdscal_v2

#define clap_cublasCswap clap::dyn::p_cublasCswap_v2
#define clap_cublasZswap clap::dyn::p_cublasZswap_v2

//Level 2
#define clap_cublasSgemv clap::dyn::p_cublasSgemv_v2
#define clap_cublasDgemv clap::dyn::p_cublasDgemv_v2
#define clap_cublasCgemv clap::dyn::p_cublasCgemv_v2
#define clap_cublasZgemv clap::dyn::p_cublasZgemv_v2
#define clap_cublasSsymv clap::dyn::p_cublasSsymv_v2
#define clap_cublasDsymv clap::dyn::p_cublasDsymv_v2
#define clap_cublasChemv clap::dyn::p_cublasChemv_v2
#define clap_cublasZhemv clap::dyn::p_cublasZhemv_v2
#define clap_cublasStrmv clap::dyn::p_cublasStrmv_v2
#define clap_cublasDtrmv clap::dyn::p_cublasDtrmv_v2
#define clap_cublasCtrmv clap::dyn::p_cublasCtrmv_v2
#define clap_cublasZtrmv clap::dyn::p_cublasZtrmv_v2
#define clap_cublasStrsv clap::dyn::p_cublasStrsv_v2
#define clap_cublasDtrsv clap::dyn::p_cublasDtrsv_v2
#define clap_cublasCtrsv clap::dyn::p_cublasCtrsv_v2
#define clap_cublasZtrsv clap::dyn::p_cublasZtrsv_v2
#define clap_cublasCgerc clap::dyn::p_cublasCgerc_v2
#define clap_cublasZgerc clap::dyn::p_cublasZgerc_v2
#define clap_cublasSger clap::dyn::p_cublasSger_v2
#define clap_cublasDger clap::dyn::p_cublasDger_v2
#define clap_cublasCgeru clap::dyn::p_cublasCgeru_v2
#define clap_cublasZgeru clap::dyn::p_cublasZgeru_v2
#define clap_cublasSsyr clap::dyn::p_cublasSsyr_v2
#define clap_cublasDsyr clap::dyn::p_cublasDsyr_v2
#define clap_cublasSsyr2 clap::dyn::p_cublasSsyr2_v2
#define clap_cublasDsyr2 clap::dyn::p_cublasDsyr2_v2
#define clap_cublasCher clap::dyn::p_cublasCher_v2
#define clap_cublasZher clap::dyn::p_cublasZher_v2
#define clap_cublasCher2 clap::dyn::p_cublasCher2_v2
#define clap_cublasZher2 clap::dyn::p_cublasZher2_v2
#define clap_cublasChpr clap::dyn::p_cublasChpr_v2
#define clap_cublasZhpr clap::dyn::p_cublasZhpr_v2
#define clap_cublasChpr2 clap::dyn::p_cublasChpr2_v2
#define clap_cublasZhpr2 clap::dyn::p_cublasZhpr2_v2
#define clap_cublasSgbmv clap::dyn::p_cublasSgbmv_v2
#define clap_cublasDgbmv clap::dyn::p_cublasDgbmv_v2
#define clap_cublasCgbmv clap::dyn::p_cublasCgbmv_v2
#define clap_cublasZgbmv clap::dyn::p_cublasZgbmv_v2
#define clap_cublasChbmv clap::dyn::p_cublasChbmv_v2
#define clap_cublasZhbmv clap::dyn::p_cublasZhbmv_v2
#define clap_cublasSsbmv clap::dyn::p_cublasSsbmv_v2
#define clap_cublasDsbmv clap::dyn::p_cublasDsbmv_v2
#define clap_cublasStbmv clap::dyn::p_cublasStbmv_v2
#define clap_cublasDtbmv clap::dyn::p_cublasDtbmv_v2
#define clap_cublasCtbmv clap::dyn::p_cublasCtbmv_v2
#define clap_cublasZtbmv clap::dyn::p_cublasZtbmv_v2
#define clap_cublasStbsv clap::dyn::p_cublasStbsv_v2
#define clap_cublasDtbsv clap::dyn::p_cublasDtbsv_v2
#define clap_cublasCtbsv clap::dyn::p_cublasCtbsv_v2
#define clap_cublasZtbsv clap::dyn::p_cublasZtbsv_v2
#define clap_cublasChpmv clap::dyn::p_cublasChpmv_v2
#define clap_cublasZhpmv clap::dyn::p_cublasZhpmv_v2
#define clap_cublasSspmv clap::dyn::p_cublasSspmv_v2
#define clap_cublasDspmv clap::dyn::p_cublasDspmv_v2
#define clap_cublasStpmv clap::dyn::p_cublasStpmv_v2
#define clap_cublasDtpmv clap::dyn::p_cublasDtpmv_v2
#define clap_cublasCtpmv clap::dyn::p_cublasCtpmv_v2
#define clap_cublasZtpmv clap::dyn::p_cublasZtpmv_v2
#define clap_cublasStpsv clap::dyn::p_cublasStpsv_v2
#define clap_cublasDtpsv clap::dyn::p_cublasDtpsv_v2
#define clap_cublasCtpsv clap::dyn::p_cublasCtpsv_v2
#define clap_cublasZtpsv clap::dyn::p_cublasZtpsv_v2
#define clap_cublasSspr clap::dyn::p_cublasSspr_v2
#define clap_cublasDspr clap::dyn::p_cublasDspr_v2
#define clap_cublasSspr2 clap::dyn::p_cublasSspr2_v2
#define clap_cublasDspr2 clap::dyn::p_cublasDspr2_v2

//Level 3
#define clap_cublasSgemm clap::dyn::p_cublasSgemm_v2
#define clap_cublasDgemm clap::dyn::p_cublasDgemm_v2
#define clap_cublasCgemm clap::dyn::p_cublasCgemm_v2
#define clap_cublasZgemm clap::dyn::p_cublasZgemm_v2
#define clap_cublasSsymm clap::dyn::p_cublasSsymm_v2
#define clap_cublasDsymm clap::dyn::p_cublasDsymm_v2
#define clap_cublasCsymm clap::dyn::p_cublasCsymm_v2
#define clap_cublasZsymm clap::dyn::p_cublasZsymm_v2
#define clap_cublasSsyrk clap::dyn::p_cublasSsyrk_v2
#define clap_cublasDsyrk clap::dyn::p_cublasDsyrk_v2
#define clap_cublasCsyrk clap::dyn::p_cublasCsyrk_v2
#define clap_cublasZsyrk clap::dyn::p_cublasZsyrk_v2
#define clap_cublasSsyr2k clap::dyn::p_cublasSsyr2k_v2
#define clap_cublasDsyr2k clap::dyn::p_cublasDsyr2k_v2
#define clap_cublasCsyr2k clap::dyn::p_cublasCsyr2k_v2
#define clap_cublasZsyr2k clap::dyn::p_cublasZsyr2k_v2
#define clap_cublasStrmm clap::dyn::p_cublasStrmm_v2
#define clap_cublasDtrmm clap::dyn::p_cublasDtrmm_v2
#define clap_cublasCtrmm clap::dyn::p_cublasCtrmm_v2
#define clap_cublasZtrmm clap::dyn::p_cublasZtrmm_v2
#define clap_cublasStrsm clap::dyn::p_cublasStrsm_v2
#define clap_cublasDtrsm clap::dyn::p_cublasDtrsm_v2
#define clap_cublasCtrsm clap::dyn::p_cublasCtrsm_v2
#define clap_cublasZtrsm clap::dyn::p_cublasZtrsm_v2

#define clap_cublasChemm clap::dyn::p_cublasChemm_v2
#define clap_cublasZhemm clap::dyn::p_cublasZhemm_v2
#define clap_cublasCherk clap::dyn::p_cublasCherk_v2
#define clap_cublasZherk clap::dyn::p_cublasZherk_v2

#define clap_cublasCher2k clap::dyn::p_cublasCher2k_v2
#define clap_cublasZher2k clap::dyn::p_cublasZher2k_v2

//==========*Openblas*=========
//Level 1
#define clap_cblas_scopy clap::dyn::p_cblas_scopy
#define clap_cblas_dcopy clap::dyn::p_cblas_dcopy
#define clap_cblas_sscal clap::dyn::p_cblas_sscal
#define clap_cblas_dscal clap::dyn::p_cblas_dscal
#define clap_cblas_saxpy clap::dyn::p_cblas_saxpy
#define clap_cblas_daxpy clap::dyn::p_cblas_daxpy
#define clap_cblas_isamax clap::dyn::p_cblas_isamax
#define clap_cblas_idamax clap::dyn::p_cblas_idamax
#define clap_cblas_sswap clap::dyn::p_cblas_sswap
#define clap_cblas_dswap clap::dyn::p_cblas_dswap
#define clap_cblas_sdot clap::dyn::p_cblas_sdot
#define clap_cblas_ddot clap::dyn::p_cblas_ddot
#define clap_cblas_snrm2 clap::dyn::p_cblas_snrm2
#define clap_cblas_dnrm2 clap::dyn::p_cblas_dnrm2
#define clap_cblas_sasum clap::dyn::p_cblas_sasum
#define clap_cblas_dasum clap::dyn::p_cblas_dasum
#define clap_cblas_srot clap::dyn::p_cblas_srot
#define clap_cblas_drot clap::dyn::p_cblas_drot
#define clap_cblas_srotg clap::dyn::p_cblas_srotg
#define clap_cblas_drotg clap::dyn::p_cblas_drotg
#define clap_cblas_srotm clap::dyn::p_cblas_srotm
#define clap_cblas_drotm clap::dyn::p_cblas_drotm
#define clap_cblas_srotmg clap::dyn::p_cblas_srotmg
#define clap_cblas_drotmg clap::dyn::p_cblas_drotmg

#define clap_cblas_cdotu_sub clap::dyn::p_cblas_cdotu_sub
#define clap_cblas_zdotu_sub clap::dyn::p_cblas_zdotu_sub
#define clap_cblas_cdotc_sub clap::dyn::p_cblas_cdotc_sub
#define clap_cblas_zdotc_sub clap::dyn::p_cblas_zdotc_sub
#define clap_cblas_cswap clap::dyn::p_cblas_cswap
#define clap_cblas_zswap clap::dyn::p_cblas_zswap
#define clap_cblas_ccopy clap::dyn::p_cblas_ccopy
#define clap_cblas_zcopy clap::dyn::p_cblas_zcopy
#define clap_cblas_cscal clap::dyn::p_cblas_cscal
#define clap_cblas_zscal clap::dyn::p_cblas_zscal
#define clap_cblas_csscal clap::dyn::p_cblas_csscal
#define clap_cblas_zdscal clap::dyn::p_cblas_zdscal
#define clap_cblas_caxpy clap::dyn::p_cblas_caxpy
#define clap_cblas_zaxpy clap::dyn::p_cblas_zaxpy
#define clap_cblas_scnrm2 clap::dyn::p_cblas_scnrm2
#define clap_cblas_dznrm2 clap::dyn::p_cblas_dznrm2
#define clap_cblas_scasum clap::dyn::p_cblas_scasum
#define clap_cblas_dzasum clap::dyn::p_cblas_dzasum
#define clap_cblas_icamax clap::dyn::p_cblas_icamax
#define clap_cblas_izamax clap::dyn::p_cblas_izamax

//Level 2
#define clap_cblas_sgemv clap::dyn::p_cblas_sgemv
#define clap_cblas_dgemv clap::dyn::p_cblas_dgemv
#define clap_cblas_cgemv clap::dyn::p_cblas_cgemv
#define clap_cblas_zgemv clap::dyn::p_cblas_zgemv
#define clap_cblas_ssymv clap::dyn::p_cblas_ssymv
#define clap_cblas_dsymv clap::dyn::p_cblas_dsymv
#define clap_cblas_chemv clap::dyn::p_cblas_chemv
#define clap_cblas_zhemv clap::dyn::p_cblas_zhemv
#define clap_cblas_strmv clap::dyn::p_cblas_strmv
#define clap_cblas_dtrmv clap::dyn::p_cblas_dtrmv
#define clap_cblas_ctrmv clap::dyn::p_cblas_ctrmv
#define clap_cblas_ztrmv clap::dyn::p_cblas_ztrmv
#define clap_cblas_strsv clap::dyn::p_cblas_strsv
#define clap_cblas_dtrsv clap::dyn::p_cblas_dtrsv
#define clap_cblas_ctrsv clap::dyn::p_cblas_ctrsv
#define clap_cblas_ztrsv clap::dyn::p_cblas_ztrsv
#define clap_cblas_cgerc clap::dyn::p_cblas_cgerc
#define clap_cblas_zgerc clap::dyn::p_cblas_zgerc
#define clap_cblas_sger clap::dyn::p_cblas_sger
#define clap_cblas_dger clap::dyn::p_cblas_dger
#define clap_cblas_cgeru clap::dyn::p_cblas_cgeru
#define clap_cblas_zgeru clap::dyn::p_cblas_zgeru
#define clap_cblas_ssyr clap::dyn::p_cblas_ssyr
#define clap_cblas_dsyr clap::dyn::p_cblas_dsyr
#define clap_cblas_ssyr2 clap::dyn::p_cblas_ssyr2
#define clap_cblas_dsyr2 clap::dyn::p_cblas_dsyr2
#define clap_cblas_cher clap::dyn::p_cblas_cher
#define clap_cblas_zher clap::dyn::p_cblas_zher
#define clap_cblas_cher2 clap::dyn::p_cblas_cher2
#define clap_cblas_zher2 clap::dyn::p_cblas_zher2
#define clap_cblas_chpr clap::dyn::p_cblas_chpr
#define clap_cblas_zhpr clap::dyn::p_cblas_zhpr
#define clap_cblas_chpr2 clap::dyn::p_cblas_chpr2
#define clap_cblas_zhpr2 clap::dyn::p_cblas_zhpr2
#define clap_cblas_sgbmv clap::dyn::p_cblas_sgbmv
#define clap_cblas_dgbmv clap::dyn::p_cblas_dgbmv
#define clap_cblas_cgbmv clap::dyn::p_cblas_cgbmv
#define clap_cblas_zgbmv clap::dyn::p_cblas_zgbmv
#define clap_cblas_chbmv clap::dyn::p_cblas_chbmv
#define clap_cblas_zhbmv clap::dyn::p_cblas_zhbmv
#define clap_cblas_ssbmv clap::dyn::p_cblas_ssbmv
#define clap_cblas_dsbmv clap::dyn::p_cblas_dsbmv
#define clap_cblas_stbmv clap::dyn::p_cblas_stbmv
#define clap_cblas_dtbmv clap::dyn::p_cblas_dtbmv
#define clap_cblas_ctbmv clap::dyn::p_cblas_ctbmv
#define clap_cblas_ztbmv clap::dyn::p_cblas_ztbmv
#define clap_cblas_stbsv clap::dyn::p_cblas_stbsv
#define clap_cblas_dtbsv clap::dyn::p_cblas_dtbsv
#define clap_cblas_ctbsv clap::dyn::p_cblas_ctbsv
#define clap_cblas_ztbsv clap::dyn::p_cblas_ztbsv
#define clap_cblas_chpmv clap::dyn::p_cblas_chpmv
#define clap_cblas_zhpmv clap::dyn::p_cblas_zhpmv
#define clap_cblas_sspmv clap::dyn::p_cblas_sspmv
#define clap_cblas_dspmv clap::dyn::p_cblas_dspmv
#define clap_cblas_stpmv clap::dyn::p_cblas_stpmv
#define clap_cblas_dtpmv clap::dyn::p_cblas_dtpmv
#define clap_cblas_ctpmv clap::dyn::p_cblas_ctpmv
#define clap_cblas_ztpmv clap::dyn::p_cblas_ztpmv
#define clap_cblas_stpsv clap::dyn::p_cblas_stpsv
#define clap_cblas_dtpsv clap::dyn::p_cblas_dtpsv
#define clap_cblas_ctpsv clap::dyn::p_cblas_ctpsv
#define clap_cblas_ztpsv clap::dyn::p_cblas_ztpsv
#define clap_cblas_sspr clap::dyn::p_cblas_sspr
#define clap_cblas_dspr clap::dyn::p_cblas_dspr
#define clap_cblas_sspr2 clap::dyn::p_cblas_sspr2
#define clap_cblas_dspr2 clap::dyn::p_cblas_dspr2

//Level 3
#define clap_cblas_sgemm clap::dyn::p_cblas_sgemm
#define clap_cblas_dgemm clap::dyn::p_cblas_dgemm
#define clap_cblas_cgemm clap::dyn::p_cblas_cgemm
#define clap_cblas_zgemm clap::dyn::p_cblas_zgemm
#define clap_cblas_ssymm clap::dyn::p_cblas_ssymm
#define clap_cblas_dsymm clap::dyn::p_cblas_dsymm
#define clap_cblas_csymm clap::dyn::p_cblas_csymm
#define clap_cblas_zsymm clap::dyn::p_cblas_zsymm
#define clap_cblas_ssyrk clap::dyn::p_cblas_ssyrk
#define clap_cblas_dsyrk clap::dyn::p_cblas_dsyrk
#define clap_cblas_csyrk clap::dyn::p_cblas_csyrk
#define clap_cblas_zsyrk clap::dyn::p_cblas_zsyrk
#define clap_cblas_ssyr2k clap::dyn::p_cblas_ssyr2k
#define clap_cblas_dsyr2k clap::dyn::p_cblas_dsyr2k
#define clap_cblas_csyr2k clap::dyn::p_cblas_csyr2k
#define clap_cblas_zsyr2k clap::dyn::p_cblas_zsyr2k
#define clap_cblas_strmm clap::dyn::p_cblas_strmm
#define clap_cblas_dtrmm clap::dyn::p_cblas_dtrmm
#define clap_cblas_ctrmm clap::dyn::p_cblas_ctrmm
#define clap_cblas_ztrmm clap::dyn::p_cblas_ztrmm
#define clap_cblas_strsm clap::dyn::p_cblas_strsm
#define clap_cblas_dtrsm clap::dyn::p_cblas_dtrsm
#define clap_cblas_ctrsm clap::dyn::p_cblas_ctrsm
#define clap_cblas_ztrsm clap::dyn::p_cblas_ztrsm
#define clap_cblas_chemm clap::dyn::p_cblas_chemm
#define clap_cblas_zhemm clap::dyn::p_cblas_zhemm
#define clap_cblas_cherk clap::dyn::p_cblas_cherk
#define clap_cblas_zherk clap::dyn::p_cblas_zherk
#define clap_cblas_cher2k clap::dyn::p_cblas_cher2k
#define clap_cblas_zher2k clap::dyn::p_cblas_zher2k

//==========*Rocblas*=========
#define clap_hipMalloc clap::dyn::p_hipMalloc
#define clap_hipFree clap::dyn::p_hipFree
#define clap_hipMemcpy clap::dyn::p_hipMemcpy

//Level 1
#define clap_rocblas_create_handle clap::dyn::p_rocblas_create_handle
#define clap_rocblas_destroy_handle clap::dyn::p_rocblas_destroy_handle
#define clap_rocblas_isamax clap::dyn::p_rocblas_isamax
#define clap_rocblas_idamax clap::dyn::p_rocblas_idamax
#define clap_rocblas_sasum clap::dyn::p_rocblas_sasum
#define clap_rocblas_dasum clap::dyn::p_rocblas_dasum
#define clap_rocblas_saxpy clap::dyn::p_rocblas_saxpy
#define clap_rocblas_daxpy clap::dyn::p_rocblas_daxpy
#define clap_rocblas_scopy clap::dyn::p_rocblas_scopy
#define clap_rocblas_dcopy clap::dyn::p_rocblas_dcopy
#define clap_rocblas_sdot clap::dyn::p_rocblas_sdot
#define clap_rocblas_ddot clap::dyn::p_rocblas_ddot
#define clap_rocblas_snrm2 clap::dyn::p_rocblas_snrm2
#define clap_rocblas_dnrm2 clap::dyn::p_rocblas_dnrm2
#define clap_rocblas_sscal clap::dyn::p_rocblas_sscal
#define clap_rocblas_dscal clap::dyn::p_rocblas_dscal
#define clap_rocblas_sswap clap::dyn::p_rocblas_sswap
#define clap_rocblas_dswap clap::dyn::p_rocblas_dswap
#define clap_rocblas_srot clap::dyn::p_rocblas_srot
#define clap_rocblas_drot clap::dyn::p_rocblas_drot
#define clap_rocblas_srotg clap::dyn::p_rocblas_srotg
#define clap_rocblas_drotg clap::dyn::p_rocblas_drotg
#define clap_rocblas_srotm clap::dyn::p_rocblas_srotm
#define clap_rocblas_drotm clap::dyn::p_rocblas_drotm
#define clap_rocblas_srotmg clap::dyn::p_rocblas_srotmg
#define clap_rocblas_drotmg clap::dyn::p_rocblas_drotmg

#define clap_rocblas_icamax clap::dyn::p_rocblas_icamax
#define clap_rocblas_izamax clap::dyn::p_rocblas_izamax
#define clap_rocblas_scasum clap::dyn::p_rocblas_scasum
#define clap_rocblas_dzasum clap::dyn::p_rocblas_dzasum
#define clap_rocblas_caxpy clap::dyn::p_rocblas_caxpy
#define clap_rocblas_zaxpy clap::dyn::p_rocblas_zaxpy
#define clap_rocblas_ccopy clap::dyn::p_rocblas_ccopy
#define clap_rocblas_zcopy clap::dyn::p_rocblas_zcopy
#define clap_rocblas_cdotu clap::dyn::p_rocblas_cdotu
#define clap_rocblas_cdotc clap::dyn::p_rocblas_cdotc
#define clap_rocblas_zdotu clap::dyn::p_rocblas_zdotu
#define clap_rocblas_zdotc clap::dyn::p_rocblas_zdotc
#define clap_rocblas_scnrm2 clap::dyn::p_rocblas_scnrm2
#define clap_rocblas_dznrm2 clap::dyn::p_rocblas_dznrm2
#define clap_rocblas_cscal clap::dyn::p_rocblas_cscal
#define clap_rocblas_zscal clap::dyn::p_rocblas_zscal
#define clap_rocblas_csscal clap::dyn::p_rocblas_csscal
#define clap_rocblas_zdscal clap::dyn::p_rocblas_zdscal
#define clap_rocblas_cswap clap::dyn::p_rocblas_cswap
#define clap_rocblas_zswap clap::dyn::p_rocblas_zswap
//Level 2
#define clap_rocblasSgemv clap::dyn::p_rocblas_sgemv
#define clap_rocblasDgemv clap::dyn::p_rocblas_dgemv
#define clap_rocblasCgemv clap::dyn::p_rocblas_cgemv
#define clap_rocblasZgemv clap::dyn::p_rocblas_zgemv
#define clap_rocblasSsymv clap::dyn::p_rocblas_ssymv
#define clap_rocblasDsymv clap::dyn::p_rocblas_dsymv
#define clap_rocblasChemv clap::dyn::p_rocblas_chemv
#define clap_rocblasZhemv clap::dyn::p_rocblas_zhemv
#define clap_rocblasStrmv clap::dyn::p_rocblas_strmv
#define clap_rocblasDtrmv clap::dyn::p_rocblas_dtrmv
#define clap_rocblasCtrmv clap::dyn::p_rocblas_ctrmv
#define clap_rocblasZtrmv clap::dyn::p_rocblas_ztrmv
#define clap_rocblasStrsv clap::dyn::p_rocblas_strsv
#define clap_rocblasDtrsv clap::dyn::p_rocblas_dtrsv
#define clap_rocblasCtrsv clap::dyn::p_rocblas_ctrsv
#define clap_rocblasZtrsv clap::dyn::p_rocblas_ztrsv
#define clap_rocblasChpmv clap::dyn::p_rocblas_chpmv
#define clap_rocblasZhpmv clap::dyn::p_rocblas_zhpmv
#define clap_rocblasSsbmv clap::dyn::p_rocblas_ssbmv
#define clap_rocblasDsbmv clap::dyn::p_rocblas_dsbmv
#define clap_rocblasStbmv clap::dyn::p_rocblas_stbmv
#define clap_rocblasDtbmv clap::dyn::p_rocblas_dtbmv
#define clap_rocblasCtbmv clap::dyn::p_rocblas_ctbmv
#define clap_rocblasZtbmv clap::dyn::p_rocblas_ztbmv
#define clap_rocblasStbsv clap::dyn::p_rocblas_stbsv
#define clap_rocblasDtbsv clap::dyn::p_rocblas_dtbsv
#define clap_rocblasCtbsv clap::dyn::p_rocblas_ctbsv
#define clap_rocblasZtbsv clap::dyn::p_rocblas_ztbsv
#define clap_rocblasSger clap::dyn::p_rocblas_sger
#define clap_rocblasDger clap::dyn::p_rocblas_dger
#define clap_rocblasCgerc clap::dyn::p_rocblas_cgerc
#define clap_rocblasZgerc clap::dyn::p_rocblas_zgerc
#define clap_rocblasCgeru clap::dyn::p_rocblas_cgeru
#define clap_rocblasZgeru clap::dyn::p_rocblas_zgeru
#define clap_rocblasSsyr clap::dyn::p_rocblas_ssyr
#define clap_rocblasDsyr clap::dyn::p_rocblas_dsyr
#define clap_rocblasSsyr2 clap::dyn::p_rocblas_ssyr2
#define clap_rocblasDsyr2 clap::dyn::p_rocblas_dsyr2
#define clap_rocblasCher clap::dyn::p_rocblas_cher
#define clap_rocblasZher clap::dyn::p_rocblas_zher
#define clap_rocblasCher2 clap::dyn::p_rocblas_cher2
#define clap_rocblasZher2 clap::dyn::p_rocblas_zher2
#define clap_rocblasChpr clap::dyn::p_rocblas_chpr
#define clap_rocblasZhpr clap::dyn::p_rocblas_zhpr
#define clap_rocblasChpr2 clap::dyn::p_rocblas_chpr2
#define clap_rocblasZhpr2 clap::dyn::p_rocblas_zhpr2
#define clap_rocblasSgbmv clap::dyn::p_rocblas_sgbmv
#define clap_rocblasDgbmv clap::dyn::p_rocblas_dgbmv
#define clap_rocblasCgbmv clap::dyn::p_rocblas_cgbmv
#define clap_rocblasZgbmv clap::dyn::p_rocblas_zgbmv
#define clap_rocblasChbmv clap::dyn::p_rocblas_chbmv
#define clap_rocblasZhbmv clap::dyn::p_rocblas_zhbmv
#define clap_rocblasSspmv clap::dyn::p_rocblas_sspmv
#define clap_rocblasDspmv clap::dyn::p_rocblas_dspmv
#define clap_rocblasStpmv clap::dyn::p_rocblas_stpmv
#define clap_rocblasDtpmv clap::dyn::p_rocblas_dtpmv
#define clap_rocblasCtpmv clap::dyn::p_rocblas_ctpmv
#define clap_rocblasZtpmv clap::dyn::p_rocblas_ztpmv
#define clap_rocblasStpsv clap::dyn::p_rocblas_stpsv
#define clap_rocblasDtpsv clap::dyn::p_rocblas_dtpsv
#define clap_rocblasCtpsv clap::dyn::p_rocblas_ctpsv
#define clap_rocblasZtpsv clap::dyn::p_rocblas_ztpsv
#define clap_rocblasSspr clap::dyn::p_rocblas_sspr
#define clap_rocblasDspr clap::dyn::p_rocblas_dspr
#define clap_rocblasSspr2 clap::dyn::p_rocblas_sspr2
#define clap_rocblasDspr2 clap::dyn::p_rocblas_dspr2
//Level 3
#define clap_rocblas_sgemm clap::dyn::p_rocblas_sgemm
#define clap_rocblas_dgemm clap::dyn::p_rocblas_dgemm
#define clap_rocblas_cgemm clap::dyn::p_rocblas_cgemm
#define clap_rocblas_zgemm clap::dyn::p_rocblas_zgemm
#define clap_rocblas_ssymm clap::dyn::p_rocblas_ssymm
#define clap_rocblas_dsymm clap::dyn::p_rocblas_dsymm
#define clap_rocblas_csymm clap::dyn::p_rocblas_csymm
#define clap_rocblas_zsymm clap::dyn::p_rocblas_zsymm
#define clap_rocblas_ssyrk clap::dyn::p_rocblas_ssyrk
#define clap_rocblas_dsyrk clap::dyn::p_rocblas_dsyrk
#define clap_rocblas_csyrk clap::dyn::p_rocblas_csyrk
#define clap_rocblas_zsyrk clap::dyn::p_rocblas_zsyrk
#define clap_rocblas_ssyr2k clap::dyn::p_rocblas_ssyr2k
#define clap_rocblas_dsyr2k clap::dyn::p_rocblas_dsyr2k
#define clap_rocblas_csyr2k clap::dyn::p_rocblas_csyr2k
#define clap_rocblas_zsyr2k clap::dyn::p_rocblas_zsyr2k
#define clap_rocblas_strmm clap::dyn::p_rocblas_strmm
#define clap_rocblas_dtrmm clap::dyn::p_rocblas_dtrmm
#define clap_rocblas_ctrmm clap::dyn::p_rocblas_ctrmm
#define clap_rocblas_ztrmm clap::dyn::p_rocblas_ztrmm
#define clap_rocblas_strsm clap::dyn::p_rocblas_strsm
#define clap_rocblas_dtrsm clap::dyn::p_rocblas_dtrsm
#define clap_rocblas_ctrsm clap::dyn::p_rocblas_ctrsm
#define clap_rocblas_ztrsm clap::dyn::p_rocblas_ztrsm
#define clap_rocblas_chemm clap::dyn::p_rocblas_chemm
#define clap_rocblas_zhemm clap::dyn::p_rocblas_zhemm
#define clap_rocblas_cherk clap::dyn::p_rocblas_cherk
#define clap_rocblas_zherk clap::dyn::p_rocblas_zherk
#define clap_rocblas_cher2k clap::dyn::p_rocblas_cher2k
#define clap_rocblas_zher2k clap::dyn::p_rocblas_zher2k
