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

#pragma once

#include "vendor_abi.hpp"
#include <cstddef>
#include <cstdint>

namespace clap::dnn_dyn {

extern abi::dnnl_status_t (*p_dnnl_engine_create)(abi::dnnl_engine_t*, abi::dnnl_engine_kind_t, std::size_t);
extern abi::dnnl_status_t (*p_dnnl_engine_destroy)(abi::dnnl_engine_t);
extern abi::dnnl_status_t (*p_dnnl_stream_create)(abi::dnnl_stream_t*, abi::dnnl_engine_t, unsigned);
extern abi::dnnl_status_t (*p_dnnl_stream_wait)(abi::dnnl_stream_t);
extern abi::dnnl_status_t (*p_dnnl_stream_destroy)(abi::dnnl_stream_t);
extern abi::dnnl_status_t (*p_dnnl_memory_desc_create_with_strides)(abi::dnnl_memory_desc_t*, int, const abi::dnnl_dim_t*, abi::dnnl_data_type_t, const abi::dnnl_dim_t*);
extern abi::dnnl_status_t (*p_dnnl_memory_desc_destroy)(abi::dnnl_memory_desc_t);
extern abi::dnnl_status_t (*p_dnnl_memory_create)(abi::dnnl_memory_t*, abi::const_dnnl_memory_desc_t, abi::dnnl_engine_t, void*);
extern abi::dnnl_status_t (*p_dnnl_memory_destroy)(abi::dnnl_memory_t);
extern abi::dnnl_status_t (*p_dnnl_primitive_create)(abi::dnnl_primitive_t*, abi::const_dnnl_primitive_desc_t);
extern abi::dnnl_status_t (*p_dnnl_primitive_execute)(abi::const_dnnl_primitive_t, abi::dnnl_stream_t, int, const abi::dnnl_exec_arg_t*);
extern abi::dnnl_status_t (*p_dnnl_primitive_destroy)(abi::dnnl_primitive_t);
extern abi::dnnl_status_t (*p_dnnl_primitive_desc_destroy)(abi::dnnl_primitive_desc_t);
extern abi::dnnl_status_t (*p_dnnl_convolution_forward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_prop_kind_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const void*);
extern abi::dnnl_status_t (*p_dnnl_convolution_backward_data_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, abi::const_dnnl_primitive_desc_t, const void*);
extern abi::dnnl_status_t (*p_dnnl_convolution_backward_weights_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, abi::const_dnnl_primitive_desc_t, const void*);
extern abi::dnnl_status_t (*p_dnnl_eltwise_forward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_prop_kind_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, float, float, const void*);
extern abi::dnnl_status_t (*p_dnnl_eltwise_backward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, float, float, abi::const_dnnl_primitive_desc_t, const void*);
extern abi::dnnl_status_t (*p_dnnl_pooling_forward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_prop_kind_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const void*);
extern abi::dnnl_status_t (*p_dnnl_pooling_backward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, abi::const_dnnl_primitive_desc_t, const void*);
extern abi::dnnl_status_t (*p_dnnl_softmax_forward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_prop_kind_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, int, const void*);
extern abi::dnnl_status_t (*p_dnnl_softmax_backward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, int, abi::const_dnnl_primitive_desc_t, const void*);
extern abi::dnnl_status_t (*p_dnnl_batch_normalization_forward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_prop_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, float, unsigned, const void*);
extern abi::dnnl_status_t (*p_dnnl_batch_normalization_backward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_prop_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, float, unsigned, abi::const_dnnl_primitive_desc_t, const void*);

extern abi::dnnl_status_t (*p_dnnl_deconvolution_forward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_prop_kind_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, const void*);
extern abi::dnnl_status_t (*p_dnnl_layer_normalization_forward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_prop_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, float, unsigned, const void*);
extern abi::dnnl_status_t (*p_dnnl_layer_normalization_backward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_prop_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, float, unsigned, abi::const_dnnl_primitive_desc_t, const void*);
extern abi::dnnl_status_t (*p_dnnl_primitive_attr_create)(abi::dnnl_primitive_attr_t*);
extern abi::dnnl_status_t (*p_dnnl_primitive_attr_destroy)(abi::dnnl_primitive_attr_t);
extern abi::dnnl_status_t (*p_dnnl_primitive_attr_set_dropout)(abi::dnnl_primitive_attr_t, abi::const_dnnl_memory_desc_t);
extern abi::dnnl_status_t (*p_dnnl_post_ops_create)(abi::dnnl_post_ops_t*);
extern abi::dnnl_status_t (*p_dnnl_post_ops_destroy)(abi::dnnl_post_ops_t);
extern abi::dnnl_status_t (*p_dnnl_post_ops_append_eltwise)(abi::dnnl_post_ops_t, abi::dnnl_alg_kind_t, float, float);
extern abi::dnnl_status_t (*p_dnnl_primitive_attr_set_post_ops)(abi::dnnl_primitive_attr_t, const abi::dnnl_post_ops_t);

extern abi::dnnl_status_t (*p_dnnl_lrn_forward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_prop_kind_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::dnnl_dim_t, float, float, float, const void*);
extern abi::dnnl_status_t (*p_dnnl_lrn_backward_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::dnnl_dim_t, float, float, float, abi::const_dnnl_primitive_desc_t, const void*);
extern abi::dnnl_status_t (*p_dnnl_binary_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, const void*);
extern abi::dnnl_status_t (*p_dnnl_reduction_primitive_desc_create)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_alg_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, float, float, const void*);
extern abi::dnnl_status_t (*p_dnnl_stream_get_engine)(const void*, abi::dnnl_engine_t*);
extern abi::const_dnnl_memory_desc_t (*p_dnnl_primitive_desc_query_md)(abi::const_dnnl_primitive_desc_t, int, int);
extern std::size_t (*p_dnnl_memory_desc_get_size)(abi::const_dnnl_memory_desc_t);
extern abi::dnnl_status_t (*p_dnnl_layer_normalization_forward_primitive_desc_create_v2)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_prop_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::dnnl_data_type_t, float, unsigned, const void*);
extern abi::dnnl_status_t (*p_dnnl_layer_normalization_backward_primitive_desc_create_v2)(abi::dnnl_primitive_desc_t*, abi::dnnl_engine_t, abi::dnnl_prop_kind_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::const_dnnl_memory_desc_t, abi::dnnl_data_type_t, abi::dnnl_data_type_t, float, unsigned, abi::const_dnnl_primitive_desc_t, const void*);

extern abi::dnnl_status_t (*p_dnnl_graph_logical_tensor_init_with_strides)(abi::dnnl_graph_logical_tensor_t*, std::size_t, abi::dnnl_data_type_t, std::int32_t, const abi::dnnl_dim_t*, const abi::dnnl_dim_t*, abi::dnnl_graph_tensor_property_t);
extern abi::dnnl_status_t (*p_dnnl_graph_op_create)(abi::dnnl_graph_op_t*, std::size_t, abi::dnnl_graph_op_kind_t, const char*);
extern abi::dnnl_status_t (*p_dnnl_graph_op_destroy)(abi::dnnl_graph_op_t);
extern abi::dnnl_status_t (*p_dnnl_graph_op_add_input)(abi::dnnl_graph_op_t, const abi::dnnl_graph_logical_tensor_t*);
extern abi::dnnl_status_t (*p_dnnl_graph_op_add_output)(abi::dnnl_graph_op_t, const abi::dnnl_graph_logical_tensor_t*);
extern abi::dnnl_status_t (*p_dnnl_graph_op_set_attr_bool)(abi::dnnl_graph_op_t, abi::dnnl_graph_op_attr_t, const std::uint8_t*, std::size_t);
extern abi::dnnl_status_t (*p_dnnl_graph_op_set_attr_s64)(abi::dnnl_graph_op_t, abi::dnnl_graph_op_attr_t, const std::int64_t*, std::size_t);
extern abi::dnnl_status_t (*p_dnnl_graph_op_set_attr_str)(abi::dnnl_graph_op_t, abi::dnnl_graph_op_attr_t, const char*, std::size_t);
extern abi::dnnl_status_t (*p_dnnl_graph_graph_create)(abi::dnnl_graph_graph_t*, abi::dnnl_engine_kind_t);
extern abi::dnnl_status_t (*p_dnnl_graph_graph_destroy)(abi::dnnl_graph_graph_t);
extern abi::dnnl_status_t (*p_dnnl_graph_add_op)(abi::dnnl_graph_graph_t, abi::dnnl_graph_op_t);
extern abi::dnnl_status_t (*p_dnnl_graph_graph_finalize)(abi::dnnl_graph_graph_t);
extern abi::dnnl_status_t (*p_dnnl_graph_graph_filter)(abi::dnnl_graph_graph_t, abi::dnnl_graph_partition_policy_t);
extern abi::dnnl_status_t (*p_dnnl_graph_graph_get_partition_num)(abi::const_dnnl_graph_graph_t, std::size_t*);
extern abi::dnnl_status_t (*p_dnnl_graph_graph_get_partitions)(abi::dnnl_graph_graph_t, std::size_t, abi::dnnl_graph_partition_t*);
extern abi::dnnl_status_t (*p_dnnl_graph_partition_destroy)(abi::dnnl_graph_partition_t);
extern abi::dnnl_status_t (*p_dnnl_graph_partition_is_supported)(abi::const_dnnl_graph_partition_t, std::uint8_t*);
extern abi::dnnl_status_t (*p_dnnl_graph_partition_compile)(abi::dnnl_graph_partition_t, abi::dnnl_graph_compiled_partition_t, std::size_t, const abi::dnnl_graph_logical_tensor_t**, std::size_t, const abi::dnnl_graph_logical_tensor_t**, abi::dnnl_engine_t);
extern abi::dnnl_status_t (*p_dnnl_graph_compiled_partition_create)(abi::dnnl_graph_compiled_partition_t*, abi::dnnl_graph_partition_t);
extern abi::dnnl_status_t (*p_dnnl_graph_compiled_partition_destroy)(abi::dnnl_graph_compiled_partition_t);
extern abi::dnnl_status_t (*p_dnnl_graph_compiled_partition_execute)(abi::const_dnnl_graph_compiled_partition_t, abi::dnnl_stream_t, std::size_t, abi::const_dnnl_graph_tensor_t*, std::size_t, abi::const_dnnl_graph_tensor_t*);
extern abi::dnnl_status_t (*p_dnnl_graph_tensor_create)(abi::dnnl_graph_tensor_t*, const abi::dnnl_graph_logical_tensor_t*, abi::dnnl_engine_t, void*);
extern abi::dnnl_status_t (*p_dnnl_graph_tensor_destroy)(abi::dnnl_graph_tensor_t);

extern abi::cudaError_t (*p_cudaGetDeviceCount)(int*);
extern abi::cudaError_t (*p_cudaMalloc)(void**, std::size_t);
extern abi::cudaError_t (*p_cudaFree)(void*);
extern abi::cudaError_t (*p_cudaMemcpy)(void*, const void*, std::size_t, abi::cudaMemcpyKind);
extern abi::cudaError_t (*p_cudaDeviceSynchronize)();
extern abi::cudaError_t (*p_cudaMemset)(void*, int, std::size_t);

extern abi::cudaError_t (*p_cudaGetDevice)(int*);
extern abi::cudaError_t (*p_cudaStreamCreate)(abi::cudaStream_t*);
extern abi::cudaError_t (*p_cudaStreamDestroy)(abi::cudaStream_t);
extern abi::cudaError_t (*p_cudaStreamSynchronize)(abi::cudaStream_t);
extern abi::cudaError_t (*p_cudaMemcpyAsync)(void*, const void*, std::size_t, abi::cudaMemcpyKind, abi::cudaStream_t);
extern abi::cudaError_t (*p_cudaMemsetAsync)(void*, int, std::size_t, abi::cudaStream_t);
extern abi::cudaError_t (*p_cudaMallocAsync)(void**, std::size_t, abi::cudaStream_t);
extern abi::cudaError_t (*p_cudaFreeAsync)(void*, abi::cudaStream_t);

extern abi::CUresult (*p_cuInit)(unsigned);
extern abi::CUresult (*p_cuModuleLoadData)(abi::CUmodule*, const void*);
extern abi::CUresult (*p_cuModuleGetFunction)(abi::CUfunction*, abi::CUmodule, const char*);
extern abi::CUresult (*p_cuLaunchKernel)(abi::CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, abi::CUstream, void**, void**);
extern abi::CUresult (*p_cuCtxSynchronize)();
extern abi::CUresult (*p_cuModuleUnload)(abi::CUmodule);

extern abi::nvrtcResult (*p_nvrtcCreateProgram)(abi::nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
extern abi::nvrtcResult (*p_nvrtcCompileProgram)(abi::nvrtcProgram, int, const char* const*);
extern abi::nvrtcResult (*p_nvrtcGetPTXSize)(abi::nvrtcProgram, std::size_t*);
extern abi::nvrtcResult (*p_nvrtcGetPTX)(abi::nvrtcProgram, char*);
extern abi::nvrtcResult (*p_nvrtcDestroyProgram)(abi::nvrtcProgram*);
extern abi::nvrtcResult (*p_nvrtcGetProgramLogSize)(abi::nvrtcProgram, std::size_t*);
extern abi::nvrtcResult (*p_nvrtcGetProgramLog)(abi::nvrtcProgram, char*);

bool loadCudaJit();
extern abi::cudnnStatus_t (*p_cudnnCreateLRNDescriptor)(abi::cudnnLRNDescriptor_t*);
extern abi::cudnnStatus_t (*p_cudnnSetLRNDescriptor)(abi::cudnnLRNDescriptor_t, unsigned, double, double, double);
extern abi::cudnnStatus_t (*p_cudnnDestroyLRNDescriptor)(abi::cudnnLRNDescriptor_t);
extern abi::cudnnStatus_t (*p_cudnnLRNCrossChannelForward)(abi::cudnnHandle_t, abi::cudnnLRNDescriptor_t, abi::cudnnLRNMode_t, const void*, abi::cudnnTensorDescriptor_t, const void*, const void*, abi::cudnnTensorDescriptor_t, void*);
extern abi::cudnnStatus_t (*p_cudnnLRNCrossChannelBackward)(abi::cudnnHandle_t, abi::cudnnLRNDescriptor_t, abi::cudnnLRNMode_t, const void*, abi::cudnnTensorDescriptor_t, const void*, abi::cudnnTensorDescriptor_t, const void*, abi::cudnnTensorDescriptor_t, const void*, const void*, abi::cudnnTensorDescriptor_t, void*);
extern abi::cudnnStatus_t (*p_cudnnCreateOpTensorDescriptor)(abi::cudnnOpTensorDescriptor_t*);
extern abi::cudnnStatus_t (*p_cudnnSetOpTensorDescriptor)(abi::cudnnOpTensorDescriptor_t, abi::cudnnOpTensorOp_t, abi::cudnnDataType_t, abi::cudnnNanPropagation_t);
extern abi::cudnnStatus_t (*p_cudnnDestroyOpTensorDescriptor)(abi::cudnnOpTensorDescriptor_t);
extern abi::cudnnStatus_t (*p_cudnnOpTensor)(abi::cudnnHandle_t, abi::cudnnOpTensorDescriptor_t, const void*, abi::cudnnTensorDescriptor_t, const void*, const void*, abi::cudnnTensorDescriptor_t, const void*, const void*, abi::cudnnTensorDescriptor_t, void*);
extern abi::cudnnStatus_t (*p_cudnnCreateReduceTensorDescriptor)(abi::cudnnReduceTensorDescriptor_t*);
extern abi::cudnnStatus_t (*p_cudnnSetReduceTensorDescriptor)(abi::cudnnReduceTensorDescriptor_t, abi::cudnnReduceTensorOp_t, abi::cudnnDataType_t, abi::cudnnNanPropagation_t, abi::cudnnReduceTensorIndices_t, abi::cudnnIndicesType_t);
extern abi::cudnnStatus_t (*p_cudnnDestroyReduceTensorDescriptor)(abi::cudnnReduceTensorDescriptor_t);
extern abi::cudnnStatus_t (*p_cudnnGetReductionWorkspaceSize)(abi::cudnnHandle_t, abi::cudnnReduceTensorDescriptor_t, abi::cudnnTensorDescriptor_t, abi::cudnnTensorDescriptor_t, std::size_t*);
extern abi::cudnnStatus_t (*p_cudnnReduceTensor)(abi::cudnnHandle_t, abi::cudnnReduceTensorDescriptor_t, void*, std::size_t, void*, std::size_t, const void*, abi::cudnnTensorDescriptor_t, const void*, const void*, abi::cudnnTensorDescriptor_t, void*);
extern abi::cudnnStatus_t (*p_cudnnCreate)(abi::cudnnHandle_t*);
extern abi::cudnnStatus_t (*p_cudnnDestroy)(abi::cudnnHandle_t);
extern abi::cudnnStatus_t (*p_cudnnCreateTensorDescriptor)(abi::cudnnTensorDescriptor_t*);
extern abi::cudnnStatus_t (*p_cudnnSetTensor4dDescriptor)(abi::cudnnTensorDescriptor_t, abi::cudnnTensorFormat_t, abi::cudnnDataType_t, int, int, int, int);
extern abi::cudnnStatus_t (*p_cudnnDestroyTensorDescriptor)(abi::cudnnTensorDescriptor_t);
extern abi::cudnnStatus_t (*p_cudnnCreateFilterDescriptor)(abi::cudnnFilterDescriptor_t*);
extern abi::cudnnStatus_t (*p_cudnnSetFilter4dDescriptor)(abi::cudnnFilterDescriptor_t, abi::cudnnDataType_t, abi::cudnnTensorFormat_t, int, int, int, int);
extern abi::cudnnStatus_t (*p_cudnnDestroyFilterDescriptor)(abi::cudnnFilterDescriptor_t);
extern abi::cudnnStatus_t (*p_cudnnCreateConvolutionDescriptor)(abi::cudnnConvolutionDescriptor_t*);
extern abi::cudnnStatus_t (*p_cudnnSetConvolution2dDescriptor)(abi::cudnnConvolutionDescriptor_t, int, int, int, int, int, int, abi::cudnnConvolutionMode_t, abi::cudnnDataType_t);
extern abi::cudnnStatus_t (*p_cudnnDestroyConvolutionDescriptor)(abi::cudnnConvolutionDescriptor_t);
extern abi::cudnnStatus_t (*p_cudnnConvolutionForward)(abi::cudnnHandle_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnFilterDescriptor_t,const void*,abi::cudnnConvolutionDescriptor_t,abi::cudnnConvolutionFwdAlgo_t,void*,std::size_t,const void*,abi::cudnnTensorDescriptor_t,void*);
extern abi::cudnnStatus_t (*p_cudnnConvolutionBackwardData)(abi::cudnnHandle_t,const void*,abi::cudnnFilterDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnConvolutionDescriptor_t,abi::cudnnConvolutionBwdDataAlgo_t,void*,std::size_t,const void*,abi::cudnnTensorDescriptor_t,void*);
extern abi::cudnnStatus_t (*p_cudnnConvolutionBackwardFilter)(abi::cudnnHandle_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnConvolutionDescriptor_t,abi::cudnnConvolutionBwdFilterAlgo_t,void*,std::size_t,const void*,abi::cudnnFilterDescriptor_t,void*);
extern abi::cudnnStatus_t (*p_cudnnCreateActivationDescriptor)(abi::cudnnActivationDescriptor_t*);
extern abi::cudnnStatus_t (*p_cudnnSetActivationDescriptor)(abi::cudnnActivationDescriptor_t,abi::cudnnActivationMode_t,abi::cudnnNanPropagation_t,double);
extern abi::cudnnStatus_t (*p_cudnnDestroyActivationDescriptor)(abi::cudnnActivationDescriptor_t);
extern abi::cudnnStatus_t (*p_cudnnActivationForward)(abi::cudnnHandle_t,abi::cudnnActivationDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,const void*,abi::cudnnTensorDescriptor_t,void*);
extern abi::cudnnStatus_t (*p_cudnnActivationBackward)(abi::cudnnHandle_t,abi::cudnnActivationDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,const void*,abi::cudnnTensorDescriptor_t,void*);
extern abi::cudnnStatus_t (*p_cudnnCreatePoolingDescriptor)(abi::cudnnPoolingDescriptor_t*);
extern abi::cudnnStatus_t (*p_cudnnSetPooling2dDescriptor)(abi::cudnnPoolingDescriptor_t,abi::cudnnPoolingMode_t,abi::cudnnNanPropagation_t,int,int,int,int,int,int);
extern abi::cudnnStatus_t (*p_cudnnDestroyPoolingDescriptor)(abi::cudnnPoolingDescriptor_t);
extern abi::cudnnStatus_t (*p_cudnnPoolingForward)(abi::cudnnHandle_t,abi::cudnnPoolingDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,const void*,abi::cudnnTensorDescriptor_t,void*);
extern abi::cudnnStatus_t (*p_cudnnPoolingBackward)(abi::cudnnHandle_t,abi::cudnnPoolingDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,const void*,abi::cudnnTensorDescriptor_t,void*);
extern abi::cudnnStatus_t (*p_cudnnSoftmaxForward)(abi::cudnnHandle_t,abi::cudnnSoftmaxAlgorithm_t,abi::cudnnSoftmaxMode_t,const void*,abi::cudnnTensorDescriptor_t,const void*,const void*,abi::cudnnTensorDescriptor_t,void*);
extern abi::cudnnStatus_t (*p_cudnnSoftmaxBackward)(abi::cudnnHandle_t,abi::cudnnSoftmaxAlgorithm_t,abi::cudnnSoftmaxMode_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,const void*,abi::cudnnTensorDescriptor_t,void*);
extern abi::cudnnStatus_t (*p_cudnnDeriveBNTensorDescriptor)(abi::cudnnTensorDescriptor_t, abi::cudnnTensorDescriptor_t, abi::cudnnBatchNormMode_t);
extern abi::cudnnStatus_t (*p_cudnnBatchNormalizationForwardTraining)(abi::cudnnHandle_t,abi::cudnnBatchNormMode_t,const void*,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,void*,abi::cudnnTensorDescriptor_t,const void*,const void*,double,void*,void*,double,void*,void*);
extern abi::cudnnStatus_t (*p_cudnnBatchNormalizationForwardInference)(abi::cudnnHandle_t,abi::cudnnBatchNormMode_t,const void*,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,void*,abi::cudnnTensorDescriptor_t,const void*,const void*,const void*,const void*,double);
extern abi::cudnnStatus_t (*p_cudnnBatchNormalizationBackward)(abi::cudnnHandle_t,abi::cudnnBatchNormMode_t,const void*,const void*,const void*,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,void*,abi::cudnnTensorDescriptor_t,const void*,void*,void*,double,const void*,const void*);

extern abi::cudnnStatus_t (*p_cudnnConvolutionBackwardBias)(abi::cudnnHandle_t,const void*,abi::cudnnTensorDescriptor_t,const void*,const void*,abi::cudnnTensorDescriptor_t,void*);
extern abi::cudnnStatus_t (*p_cudnnConvolutionBiasActivationForward)(abi::cudnnHandle_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnFilterDescriptor_t,const void*,abi::cudnnConvolutionDescriptor_t,abi::cudnnConvolutionFwdAlgo_t,void*,std::size_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnActivationDescriptor_t,abi::cudnnTensorDescriptor_t,void*);
extern abi::cudnnStatus_t (*p_cudnnCreateDropoutDescriptor)(abi::cudnnDropoutDescriptor_t*);
extern abi::cudnnStatus_t (*p_cudnnDestroyDropoutDescriptor)(abi::cudnnDropoutDescriptor_t);
extern abi::cudnnStatus_t (*p_cudnnDropoutGetStatesSize)(abi::cudnnHandle_t,std::size_t*);
extern abi::cudnnStatus_t (*p_cudnnSetDropoutDescriptor)(abi::cudnnDropoutDescriptor_t,abi::cudnnHandle_t,float,void*,std::size_t,unsigned long long);
extern abi::cudnnStatus_t (*p_cudnnDropoutGetReserveSpaceSize)(abi::cudnnTensorDescriptor_t,std::size_t*);
extern abi::cudnnStatus_t (*p_cudnnDropoutForward)(abi::cudnnHandle_t,abi::cudnnDropoutDescriptor_t,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,void*,void*,std::size_t);
extern abi::cudnnStatus_t (*p_cudnnDropoutBackward)(abi::cudnnHandle_t,abi::cudnnDropoutDescriptor_t,abi::cudnnTensorDescriptor_t,const void*,abi::cudnnTensorDescriptor_t,void*,void*,std::size_t);
extern abi::cudnnStatus_t (*p_cudnnBackendCreateDescriptor)(int,abi::cudnnBackendDescriptor_t*);
extern abi::cudnnStatus_t (*p_cudnnBackendDestroyDescriptor)(abi::cudnnBackendDescriptor_t);
extern abi::cudnnStatus_t (*p_cudnnBackendSetAttribute)(abi::cudnnBackendDescriptor_t,int,int,std::int64_t,const void*);
extern abi::cudnnStatus_t (*p_cudnnBackendFinalize)(abi::cudnnBackendDescriptor_t);
extern abi::cudnnStatus_t (*p_cudnnBackendExecute)(abi::cudnnHandle_t,abi::cudnnBackendDescriptor_t,abi::cudnnBackendDescriptor_t);
extern abi::cudnnStatus_t (*p_cudnnBackendGetAttribute)(abi::cudnnBackendDescriptor_t,int,int,std::int64_t,std::int64_t*,void*);
extern std::size_t (*p_cudnnGetVersion)();
extern abi::cudnnStatus_t (*p_cudnnSetStream)(abi::cudnnHandle_t,abi::cudaStream_t);
extern abi::cudnnStatus_t (*p_cudnnSetTensorNdDescriptor)(abi::cudnnTensorDescriptor_t,abi::cudnnDataType_t,int,const int*,const int*);
extern abi::cudnnStatus_t (*p_cudnnSetTensor4dDescriptorEx)(abi::cudnnTensorDescriptor_t,abi::cudnnDataType_t,int,int,int,int,int,int,int,int);
extern abi::cudnnStatus_t (*p_cudnnSetActivationDescriptorSwishBeta)(abi::cudnnActivationDescriptor_t,double);

extern abi::hipError_t (*p_hipGetDeviceCount)(int*);
extern abi::hipError_t (*p_hipMalloc)(void**, std::size_t);
extern abi::hipError_t (*p_hipFree)(void*);
extern abi::hipError_t (*p_hipMemcpy)(void*, const void*, std::size_t, abi::hipMemcpyKind);
extern abi::hipError_t (*p_hipDeviceSynchronize)();

extern abi::hipError_t (*p_hipGetDevice)(int*);
extern abi::hipError_t (*p_hipStreamCreate)(abi::hipStream_t*);
extern abi::hipError_t (*p_hipStreamDestroy)(abi::hipStream_t);
extern abi::hipError_t (*p_hipStreamSynchronize)(abi::hipStream_t);
extern abi::hipError_t (*p_hipMemcpyAsync)(void*, const void*, std::size_t, abi::hipMemcpyKind, abi::hipStream_t);
extern abi::hipError_t (*p_hipMemsetAsync)(void*, int, std::size_t, abi::hipStream_t);
extern abi::hipError_t (*p_hipMallocAsync)(void**, std::size_t, abi::hipStream_t);
extern abi::hipError_t (*p_hipFreeAsync)(void*, abi::hipStream_t);
extern abi::miopenStatus_t (*p_miopenSetStream)(abi::miopenHandle_t, abi::hipStream_t);
extern abi::miopenStatus_t (*p_miopenSetTensorDescriptor)(abi::miopenTensorDescriptor_t, abi::miopenDataType_t, int, const int*, const int*);
extern abi::miopenStatus_t (*p_miopenCreateLRNDescriptor)(abi::miopenLRNDescriptor_t*);
extern abi::miopenStatus_t (*p_miopenSetLRNDescriptor)(abi::miopenLRNDescriptor_t, abi::miopenLRNMode_t, unsigned, double, double, double);
extern abi::miopenStatus_t (*p_miopenDestroyLRNDescriptor)(abi::miopenLRNDescriptor_t);
extern abi::miopenStatus_t (*p_miopenLRNForward)(abi::miopenHandle_t, abi::miopenLRNDescriptor_t, const void*, abi::miopenTensorDescriptor_t, const void*, const void*, abi::miopenTensorDescriptor_t, void*, bool, void*);
extern abi::miopenStatus_t (*p_miopenLRNBackward)(abi::miopenHandle_t, abi::miopenLRNDescriptor_t, const void*, abi::miopenTensorDescriptor_t, const void*, abi::miopenTensorDescriptor_t, const void*, abi::miopenTensorDescriptor_t, const void*, const void*, abi::miopenTensorDescriptor_t, void*, const void*);
extern abi::miopenStatus_t (*p_miopenLRNGetWorkSpaceSize)(abi::miopenTensorDescriptor_t, std::size_t*);
extern abi::miopenStatus_t (*p_miopenOpTensor)(abi::miopenHandle_t, abi::miopenTensorOp_t, const void*, abi::miopenTensorDescriptor_t, const void*, const void*, abi::miopenTensorDescriptor_t, const void*, const void*, abi::miopenTensorDescriptor_t, void*);
extern abi::miopenStatus_t (*p_miopenCreateReduceTensorDescriptor)(abi::miopenReduceTensorDescriptor_t*);
extern abi::miopenStatus_t (*p_miopenDestroyReduceTensorDescriptor)(abi::miopenReduceTensorDescriptor_t);
extern abi::miopenStatus_t (*p_miopenSetReduceTensorDescriptor)(abi::miopenReduceTensorDescriptor_t, abi::miopenReduceTensorOp_t, abi::miopenDataType_t, abi::miopenNanPropagation_t, abi::miopenReduceTensorIndices_t, abi::miopenIndicesType_t);
extern abi::miopenStatus_t (*p_miopenGetReductionWorkspaceSize)(abi::miopenHandle_t, abi::miopenReduceTensorDescriptor_t, abi::miopenTensorDescriptor_t, abi::miopenTensorDescriptor_t, std::size_t*);
extern abi::miopenStatus_t (*p_miopenReduceTensor)(abi::miopenHandle_t, abi::miopenReduceTensorDescriptor_t, void*, std::size_t, void*, std::size_t, const void*, abi::miopenTensorDescriptor_t, const void*, const void*, abi::miopenTensorDescriptor_t, void*);
extern abi::miopenStatus_t (*p_miopenCreate)(abi::miopenHandle_t*);
extern abi::miopenStatus_t (*p_miopenDestroy)(abi::miopenHandle_t);
extern abi::miopenStatus_t (*p_miopenCreateTensorDescriptor)(abi::miopenTensorDescriptor_t*);
extern abi::miopenStatus_t (*p_miopenSet4dTensorDescriptor)(abi::miopenTensorDescriptor_t,abi::miopenDataType_t,int,int,int,int);
extern abi::miopenStatus_t (*p_miopenDestroyTensorDescriptor)(abi::miopenTensorDescriptor_t);
extern abi::miopenStatus_t (*p_miopenCreateConvolutionDescriptor)(abi::miopenConvolutionDescriptor_t*);
extern abi::miopenStatus_t (*p_miopenInitConvolutionDescriptor)(abi::miopenConvolutionDescriptor_t,abi::miopenConvolutionMode_t,int,int,int,int,int,int);
extern abi::miopenStatus_t (*p_miopenDestroyConvolutionDescriptor)(abi::miopenConvolutionDescriptor_t);
extern abi::miopenStatus_t (*p_miopenConvolutionForward)(abi::miopenHandle_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenConvolutionDescriptor_t,abi::miopenConvFwdAlgorithm_t,const void*,abi::miopenTensorDescriptor_t,void*,void*,std::size_t);
extern abi::miopenStatus_t (*p_miopenConvolutionBackwardData)(abi::miopenHandle_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenConvolutionDescriptor_t,abi::miopenConvBwdDataAlgorithm_t,const void*,abi::miopenTensorDescriptor_t,void*,void*,std::size_t);
extern abi::miopenStatus_t (*p_miopenConvolutionBackwardWeights)(abi::miopenHandle_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenConvolutionDescriptor_t,abi::miopenConvBwdWeightsAlgorithm_t,const void*,abi::miopenTensorDescriptor_t,void*,void*,std::size_t);
extern abi::miopenStatus_t (*p_miopenCreateActivationDescriptor)(abi::miopenActivationDescriptor_t*);
extern abi::miopenStatus_t (*p_miopenSetActivationDescriptor)(abi::miopenActivationDescriptor_t,abi::miopenActivationMode_t,double,double,double);
extern abi::miopenStatus_t (*p_miopenDestroyActivationDescriptor)(abi::miopenActivationDescriptor_t);
extern abi::miopenStatus_t (*p_miopenActivationForward)(abi::miopenHandle_t,abi::miopenActivationDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,const void*,abi::miopenTensorDescriptor_t,void*);
extern abi::miopenStatus_t (*p_miopenActivationBackward)(abi::miopenHandle_t,abi::miopenActivationDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,const void*,abi::miopenTensorDescriptor_t,void*);
extern abi::miopenStatus_t (*p_miopenCreatePoolingDescriptor)(abi::miopenPoolingDescriptor_t*);
extern abi::miopenStatus_t (*p_miopenSet2dPoolingDescriptor)(abi::miopenPoolingDescriptor_t,abi::miopenPoolingMode_t,int,int,int,int,int,int);
extern abi::miopenStatus_t (*p_miopenDestroyPoolingDescriptor)(abi::miopenPoolingDescriptor_t);
extern abi::miopenStatus_t (*p_miopenPoolingForward)(abi::miopenHandle_t,abi::miopenPoolingDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,const void*,abi::miopenTensorDescriptor_t,void*,bool,void*,std::size_t);
extern abi::miopenStatus_t (*p_miopenPoolingBackward)(abi::miopenHandle_t,abi::miopenPoolingDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,const void*,abi::miopenTensorDescriptor_t,void*,void*);
extern abi::miopenStatus_t (*p_miopenSoftmaxForward_V2)(abi::miopenHandle_t,const void*,abi::miopenTensorDescriptor_t,const void*,const void*,abi::miopenTensorDescriptor_t,void*,abi::miopenSoftmaxAlgorithm_t,abi::miopenSoftmaxMode_t);
extern abi::miopenStatus_t (*p_miopenSoftmaxBackward_V2)(abi::miopenHandle_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,const void*,abi::miopenTensorDescriptor_t,void*,abi::miopenSoftmaxAlgorithm_t,abi::miopenSoftmaxMode_t);
extern abi::miopenStatus_t (*p_miopenBatchNormalizationForwardTraining)(abi::miopenHandle_t,abi::miopenBatchNormMode_t,const void*,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,void*,abi::miopenTensorDescriptor_t,const void*,const void*,double,void*,void*,double,void*,void*);
extern abi::miopenStatus_t (*p_miopenBatchNormalizationForwardInference)(abi::miopenHandle_t,abi::miopenBatchNormMode_t,const void*,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,void*,abi::miopenTensorDescriptor_t,const void*,const void*,const void*,const void*,double);
extern abi::miopenStatus_t (*p_miopenBatchNormalizationBackward)(abi::miopenHandle_t,abi::miopenBatchNormMode_t,const void*,const void*,const void*,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,void*,abi::miopenTensorDescriptor_t,const void*,void*,void*,double,const void*,const void*);

extern abi::miopenStatus_t (*p_miopenConvolutionBackwardBias)(abi::miopenHandle_t,const void*,abi::miopenTensorDescriptor_t,const void*,const void*,abi::miopenTensorDescriptor_t,void*);
extern abi::miopenStatus_t (*p_miopenConvolutionForwardBias)(abi::miopenHandle_t,const void*,abi::miopenTensorDescriptor_t,const void*,const void*,abi::miopenTensorDescriptor_t,void*);
extern abi::miopenStatus_t (*p_miopenLayerNormForward)(abi::miopenHandle_t,abi::miopenNormMode_t,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,float,std::int32_t,abi::miopenTensorDescriptor_t,void*,abi::miopenTensorDescriptor_t,void*,abi::miopenTensorDescriptor_t,void*);
extern abi::miopenStatus_t (*p_miopenGetLayerNormBackwardWorkspaceSize)(abi::miopenHandle_t,abi::miopenNormMode_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,std::int32_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,std::size_t*);
extern abi::miopenStatus_t (*p_miopenLayerNormBackward)(abi::miopenHandle_t,abi::miopenNormMode_t,void*,std::size_t,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,std::int32_t,abi::miopenTensorDescriptor_t,void*,abi::miopenTensorDescriptor_t,void*,abi::miopenTensorDescriptor_t,void*);
extern abi::miopenStatus_t (*p_miopenCreateDropoutDescriptor)(abi::miopenDropoutDescriptor_t*);
extern abi::miopenStatus_t (*p_miopenDestroyDropoutDescriptor)(abi::miopenDropoutDescriptor_t);
extern abi::miopenStatus_t (*p_miopenDropoutGetStatesSize)(abi::miopenHandle_t,std::size_t*);
extern abi::miopenStatus_t (*p_miopenSetDropoutDescriptor)(abi::miopenDropoutDescriptor_t,abi::miopenHandle_t,float,void*,std::size_t,unsigned long long,bool,bool,int);
extern abi::miopenStatus_t (*p_miopenDropoutGetReserveSpaceSize)(abi::miopenTensorDescriptor_t,std::size_t*);
extern abi::miopenStatus_t (*p_miopenDropoutForward)(abi::miopenHandle_t,abi::miopenDropoutDescriptor_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,void*,void*,std::size_t);
extern abi::miopenStatus_t (*p_miopenDropoutBackward)(abi::miopenHandle_t,abi::miopenDropoutDescriptor_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,void*,void*,std::size_t);

extern abi::miopenStatus_t (*p_miopenT5LayerNormForward)(abi::miopenHandle_t,abi::miopenNormMode_t,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,float,abi::miopenTensorDescriptor_t,void*,abi::miopenTensorDescriptor_t,void*);
extern abi::miopenStatus_t (*p_miopenGetT5LayerNormBackwardWorkspaceSize)(abi::miopenHandle_t,abi::miopenNormMode_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,abi::miopenTensorDescriptor_t,std::size_t*);
extern abi::miopenStatus_t (*p_miopenT5LayerNormBackward)(abi::miopenHandle_t,abi::miopenNormMode_t,void*,std::size_t,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,const void*,abi::miopenTensorDescriptor_t,void*,abi::miopenTensorDescriptor_t,void*);
extern abi::miopenStatus_t (*p_miopenCreateMhaDescriptor)(abi::miopenMhaDescriptor_t*);
extern abi::miopenStatus_t (*p_miopenSetMhaDescriptor)(abi::miopenMhaDescriptor_t,float);
extern abi::miopenStatus_t (*p_miopenCreateMhaProblem)(abi::miopenProblem_t*,abi::miopenMhaDescriptor_t,abi::miopenProblemDirection_t);
extern abi::miopenStatus_t (*p_miopenSetProblemTensorDescriptor)(abi::miopenProblem_t,abi::miopenTensorArgumentId_t,abi::miopenTensorDescriptor_t);
extern abi::miopenStatus_t (*p_miopenFindSolutions)(abi::miopenHandle_t,abi::miopenProblem_t,abi::miopenFindOptions_t,abi::miopenSolution_t*,std::size_t*,std::size_t);
extern abi::miopenStatus_t (*p_miopenGetSolutionWorkspaceSize)(abi::miopenSolution_t,std::size_t*);
extern abi::miopenStatus_t (*p_miopenRunSolution)(abi::miopenHandle_t,abi::miopenSolution_t,std::size_t,const abi::miopenTensorArgument_t*,void*,std::size_t);
extern abi::miopenStatus_t (*p_miopenDestroySolution)(abi::miopenSolution_t);
extern abi::miopenStatus_t (*p_miopenDestroyProblem)(abi::miopenProblem_t);

bool loadOneDnn();
bool loadCudaAndCudnn();
bool loadHipAndMiopen();

bool hasOneDnnGraph();
bool hasMiopenT5LayerNorm();
bool hasMiopenMha();
bool hasNvidiaGpu();
bool hasAmdGpu();
const char* lastError();

}
