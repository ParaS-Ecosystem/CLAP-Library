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

#include <cstddef>
#include <cstdint>

namespace clap::abi {

using cudaError_t = int;
using cudaStream_t = void*;
enum cudaMemcpyKind : int { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };

using CUresult = int;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using nvrtcResult = int;
using nvrtcProgram = void*;
constexpr int CUDA_SUCCESS = 0;
constexpr int NVRTC_SUCCESS = 0;

using hipError_t = int;
using hipStream_t = void*;
enum hipMemcpyKind : int { hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2 };

using cudnnStatus_t = int;
using cudnnHandle_t = void*;
using cudnnTensorDescriptor_t = void*;
using cudnnFilterDescriptor_t = void*;
using cudnnConvolutionDescriptor_t = void*;
using cudnnActivationDescriptor_t = void*;
using cudnnPoolingDescriptor_t = void*;
using cudnnLRNDescriptor_t = void*;
using cudnnLRNMode_t = int;
using cudnnOpTensorDescriptor_t = void*;
using cudnnReduceTensorDescriptor_t = void*;
using cudnnDropoutDescriptor_t = void*;
using cudnnBackendDescriptor_t = void*;
using cudnnDataType_t = int;
using cudnnTensorFormat_t = int;
using cudnnConvolutionMode_t = int;
using cudnnNanPropagation_t = int;
using cudnnActivationMode_t = int;
using cudnnPoolingMode_t = int;
using cudnnSoftmaxAlgorithm_t = int;
using cudnnSoftmaxMode_t = int;
using cudnnBatchNormMode_t = int;
using cudnnOpTensorOp_t = int;
using cudnnReduceTensorOp_t = int;
using cudnnReduceTensorIndices_t = int;
using cudnnIndicesType_t = int;
using cudnnConvolutionFwdAlgo_t = int;
using cudnnConvolutionBwdDataAlgo_t = int;
using cudnnConvolutionBwdFilterAlgo_t = int;

constexpr int CUDNN_STATUS_SUCCESS = 0;
constexpr int CUDNN_STATUS_BAD_PARAM = 2000;
constexpr int CUDNN_STATUS_NOT_SUPPORTED = 3000;
constexpr int CUDNN_DATA_FLOAT = 0;
constexpr int CUDNN_DATA_HALF = 2;
constexpr int CUDNN_DATA_INT32 = 4;
constexpr int CUDNN_DATA_BFLOAT16 = 9;
constexpr int CUDNN_DATA_BOOLEAN = 11;
constexpr int CUDNN_TENSOR_NCHW = 0;
constexpr int CUDNN_CROSS_CORRELATION = 1;
constexpr int CUDNN_PROPAGATE_NAN = 1;
constexpr int CUDNN_ACTIVATION_SIGMOID = 0;
constexpr int CUDNN_ACTIVATION_RELU = 1;
constexpr int CUDNN_ACTIVATION_TANH = 2;
constexpr int CUDNN_ACTIVATION_SWISH = 6;
constexpr int CUDNN_POOLING_MAX = 0;
constexpr int CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1;
constexpr int CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0;
constexpr int CUDNN_SOFTMAX_ACCURATE = 1;
constexpr int CUDNN_SOFTMAX_LOG = 2;
constexpr int CUDNN_SOFTMAX_MODE_CHANNEL = 1;
constexpr int CUDNN_BATCHNORM_SPATIAL = 1;
constexpr int CUDNN_OP_TENSOR_ADD = 0;
constexpr int CUDNN_OP_TENSOR_MUL = 1;
constexpr int CUDNN_OP_TENSOR_MIN = 2;
constexpr int CUDNN_OP_TENSOR_MAX = 3;
constexpr int CUDNN_REDUCE_TENSOR_ADD = 0;
constexpr int CUDNN_REDUCE_TENSOR_MUL = 1;
constexpr int CUDNN_REDUCE_TENSOR_MAX = 3;
constexpr int CUDNN_PROPAGATE_NAN_REDUCE = 1;
constexpr int CUDNN_REDUCE_TENSOR_NO_INDICES = 0;
constexpr int CUDNN_32BIT_INDICES = 0;
constexpr int CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0;
constexpr int CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0;
constexpr int CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0;

using cudnnBackendAttributeName_t = int;
using cudnnBackendAttributeType_t = int;
using cudnnBackendDescriptorType_t = int;

constexpr int CUDNN_BACKEND_POINTWISE_DESCRIPTOR = 0;
constexpr int CUDNN_BACKEND_ENGINECFG_DESCRIPTOR = 3;
constexpr int CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR = 4;
constexpr int CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR = 5;
constexpr int CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR = 13;
constexpr int CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR = 15;
constexpr int CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR = 16;
constexpr int CUDNN_BACKEND_TENSOR_DESCRIPTOR = 17;
constexpr int CUDNN_BACKEND_MATMUL_DESCRIPTOR = 18;
constexpr int CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR = 19;
constexpr int CUDNN_BACKEND_REDUCTION_DESCRIPTOR = 21;
constexpr int CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR = 22;
constexpr int CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR = 29;
constexpr int CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR = 30;

constexpr int CUDNN_TYPE_HANDLE = 0;
constexpr int CUDNN_TYPE_DATA_TYPE = 1;
constexpr int CUDNN_TYPE_BOOLEAN = 2;
constexpr int CUDNN_TYPE_INT64 = 3;
constexpr int CUDNN_TYPE_FLOAT = 4;
constexpr int CUDNN_TYPE_DOUBLE = 5;
constexpr int CUDNN_TYPE_VOID_PTR = 6;
constexpr int CUDNN_TYPE_HEUR_MODE = 8;
constexpr int CUDNN_TYPE_POINTWISE_MODE = 14;
constexpr int CUDNN_TYPE_BACKEND_DESCRIPTOR = 15;
constexpr int CUDNN_TYPE_REDUCTION_OPERATOR_TYPE = 18;
constexpr int CUDNN_TYPE_NORM_MODE = 27;
constexpr int CUDNN_TYPE_NORM_FWD_PHASE = 28;

constexpr int CUDNN_ATTR_POINTWISE_MODE = 0;
constexpr int CUDNN_ATTR_POINTWISE_MATH_PREC = 1;
constexpr int CUDNN_ATTR_POINTWISE_AXIS = 9;
constexpr int CUDNN_ATTR_ENGINEHEUR_MODE = 200;
constexpr int CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = 201;
constexpr int CUDNN_ATTR_ENGINEHEUR_RESULTS = 202;
constexpr int CUDNN_ATTR_EXECUTION_PLAN_HANDLE = 400;
constexpr int CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG = 401;
constexpr int CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = 402;
constexpr int CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = 750;
constexpr int CUDNN_ATTR_OPERATION_POINTWISE_XDESC = 751;
constexpr int CUDNN_ATTR_OPERATION_POINTWISE_BDESC = 752;
constexpr int CUDNN_ATTR_OPERATION_POINTWISE_YDESC = 753;
constexpr int CUDNN_ATTR_OPERATION_POINTWISE_TDESC = 758;
constexpr int CUDNN_ATTR_OPERATIONGRAPH_HANDLE = 800;
constexpr int CUDNN_ATTR_OPERATIONGRAPH_OPS = 801;
constexpr int CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT = 900;
constexpr int CUDNN_ATTR_TENSOR_DATA_TYPE = 901;
constexpr int CUDNN_ATTR_TENSOR_DIMENSIONS = 902;
constexpr int CUDNN_ATTR_TENSOR_STRIDES = 903;
constexpr int CUDNN_ATTR_TENSOR_UNIQUE_ID = 906;
constexpr int CUDNN_ATTR_TENSOR_IS_VIRTUAL = 907;
constexpr int CUDNN_ATTR_TENSOR_IS_BY_VALUE = 908;
constexpr int CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS = 1000;
constexpr int CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = 1001;
constexpr int CUDNN_ATTR_VARIANT_PACK_WORKSPACE = 1003;
constexpr int CUDNN_ATTR_MATMUL_COMP_TYPE = 1500;
constexpr int CUDNN_ATTR_OPERATION_MATMUL_ADESC = 1520;
constexpr int CUDNN_ATTR_OPERATION_MATMUL_BDESC = 1521;
constexpr int CUDNN_ATTR_OPERATION_MATMUL_CDESC = 1522;
constexpr int CUDNN_ATTR_OPERATION_MATMUL_DESC = 1523;
constexpr int CUDNN_ATTR_REDUCTION_OPERATOR = 1600;
constexpr int CUDNN_ATTR_REDUCTION_COMP_TYPE = 1601;
constexpr int CUDNN_ATTR_OPERATION_REDUCTION_XDESC = 1610;
constexpr int CUDNN_ATTR_OPERATION_REDUCTION_YDESC = 1611;
constexpr int CUDNN_ATTR_OPERATION_REDUCTION_DESC = 1612;
constexpr int CUDNN_ATTR_OPERATION_NORM_FWD_MODE = 2000;
constexpr int CUDNN_ATTR_OPERATION_NORM_FWD_PHASE = 2001;
constexpr int CUDNN_ATTR_OPERATION_NORM_FWD_XDESC = 2002;
constexpr int CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC = 2004;
constexpr int CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC = 2005;
constexpr int CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC = 2007;
constexpr int CUDNN_ATTR_OPERATION_NORM_FWD_YDESC = 2013;
constexpr int CUDNN_ATTR_OPERATION_NORM_BWD_MODE = 2100;
constexpr int CUDNN_ATTR_OPERATION_NORM_BWD_XDESC = 2101;
constexpr int CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC = 2103;
constexpr int CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC = 2104;
constexpr int CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC = 2105;
constexpr int CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC = 2107;
constexpr int CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC = 2109;

constexpr int CUDNN_POINTWISE_ADD = 0;
constexpr int CUDNN_POINTWISE_MUL = 1;
constexpr int CUDNN_POINTWISE_DIV = 6;
constexpr int CUDNN_POINTWISE_SUB = 9;
constexpr int CUDNN_POINTWISE_EXP = 13;
constexpr int CUDNN_POINTWISE_LOG = 15;
constexpr int CUDNN_POINTWISE_CMP_GE = 303;
constexpr int CUDNN_POINTWISE_GEN_INDEX = 501;
constexpr int CUDNN_POINTWISE_BINARY_SELECT = 601;

constexpr int CUDNN_RMS_NORM = 4;
constexpr int CUDNN_NORM_FWD_INFERENCE = 0;
constexpr int CUDNN_NORM_FWD_TRAINING = 1;
constexpr int CUDNN_HEUR_MODE_FALLBACK = 2;
constexpr int CUDNN_HEUR_MODE_A = 3;

using miopenStatus_t = int;
using miopenHandle_t = void*;
using miopenTensorDescriptor_t = void*;
using miopenConvolutionDescriptor_t = void*;
using miopenActivationDescriptor_t = void*;
using miopenPoolingDescriptor_t = void*;
using miopenDropoutDescriptor_t = void*;
using miopenLRNDescriptor_t = void*;
using miopenReduceTensorDescriptor_t = void*;
using miopenNormMode_t = int;
using miopenDataType_t = int;
using miopenConvolutionMode_t = int;
using miopenActivationMode_t = int;
using miopenPoolingMode_t = int;
using miopenTensorOp_t = int;
using miopenReduceTensorOp_t = int;
using miopenNanPropagation_t = int;
using miopenReduceTensorIndices_t = int;
using miopenIndicesType_t = int;
using miopenLRNMode_t = int;
using miopenSoftmaxAlgorithm_t = int;
using miopenSoftmaxMode_t = int;
using miopenBatchNormMode_t = int;
using miopenConvFwdAlgorithm_t = int;
using miopenConvBwdDataAlgorithm_t = int;
using miopenConvBwdWeightsAlgorithm_t = int;
using miopenMhaDescriptor_t = void*;
using miopenProblem_t = void*;
using miopenFindOptions_t = void*;
using miopenSolution_t = void*;
using miopenProblemDirection_t = int;
using miopenTensorArgumentId_t = int;

struct miopenTensorArgument_t {
    miopenTensorArgumentId_t id;
    miopenTensorDescriptor_t* descriptor;
    void* buffer;
};

constexpr int miopenStatusSuccess = 0;
constexpr int miopenStatusNotImplemented = 6;
constexpr int miopenStatusUnsupportedOp = 8;
constexpr int miopenHalf = 0;
constexpr int miopenFloat = 1;
constexpr int miopenBFloat16 = 5;
constexpr int miopenInt64 = 9;
constexpr int miopenConvolution = 0;
constexpr int miopenTranspose = 1;
constexpr int MIOPEN_WEIGHT_BIAS = 1;
constexpr int MIOPEN_WEIGHT_BIAS_T5 = 5;
constexpr int miopenActivationPASTHRU = 0;
constexpr int miopenActivationLOGISTIC = 1;
constexpr int miopenActivationTANH = 2;
constexpr int miopenActivationRELU = 3;
constexpr int miopenPoolingMax = 0;
constexpr int miopenPoolingAverage = 1;
constexpr int MIOPEN_SOFTMAX_ACCURATE = 1;
constexpr int MIOPEN_SOFTMAX_LOG = 2;
constexpr int MIOPEN_SOFTMAX_MODE_CHANNEL = 1;
constexpr int miopenBNSpatial = 1;
constexpr int miopenConvolutionFwdAlgoGEMM = 0;
constexpr int miopenConvolutionBwdDataAlgoGEMM = 0;
constexpr int miopenConvolutionBwdWeightsAlgoGEMM = 0;
constexpr int miopenTensorOpAdd = 0;
constexpr int miopenTensorOpMul = 1;
constexpr int miopenTensorOpMin = 2;
constexpr int miopenTensorOpMax = 3;
constexpr int miopenReduceTensorAdd = 0;
constexpr int miopenReduceTensorMul = 1;
constexpr int miopenNotPropagateNaN = 0;
constexpr int miopenReduceTensorNoIndices = 0;
constexpr int miopen32BitIndices = 0;
constexpr int miopenLRNCrossChannel = 1;

constexpr int miopenProblemDirectionForward = 0;
constexpr int miopenTensorMhaK = 4;
constexpr int miopenTensorMhaQ = 5;
constexpr int miopenTensorMhaV = 6;
constexpr int miopenTensorMhaDescaleK = 7;
constexpr int miopenTensorMhaDescaleQ = 8;
constexpr int miopenTensorMhaDescaleV = 9;
constexpr int miopenTensorMhaDescaleS = 10;
constexpr int miopenTensorMhaScaleS = 11;
constexpr int miopenTensorMhaScaleO = 12;
constexpr int miopenTensorMhaDropoutProbability = 13;
constexpr int miopenTensorMhaDropoutSeed = 14;
constexpr int miopenTensorMhaDropoutOffset = 15;
constexpr int miopenTensorMhaO = 16;
constexpr int miopenTensorMhaAmaxO = 17;
constexpr int miopenTensorMhaAmaxS = 18;
constexpr int miopenTensorMhaM = 19;
constexpr int miopenTensorMhaZInv = 20;

using dnnl_status_t = int;
using dnnl_dim_t = std::int64_t;
using dnnl_engine_t = void*;
using dnnl_stream_t = void*;
using dnnl_memory_desc_t = void*;
using const_dnnl_memory_desc_t = const void*;
using dnnl_memory_t = void*;
using dnnl_primitive_desc_t = void*;
using const_dnnl_primitive_desc_t = const void*;
using dnnl_primitive_t = void*;
using const_dnnl_primitive_t = const void*;
using dnnl_engine_kind_t = int;
using dnnl_data_type_t = int;
using dnnl_format_tag_t = int;
using dnnl_prop_kind_t = int;
using dnnl_alg_kind_t = int;
using dnnl_primitive_attr_t = void*;
using const_dnnl_primitive_attr_t = const void*;
using dnnl_post_ops_t = void*;

struct dnnl_exec_arg_t { int arg; dnnl_memory_t memory; };

constexpr int dnnl_success = 0;
constexpr int dnnl_unimplemented = 3;
constexpr int dnnl_cpu = 1;
constexpr unsigned dnnl_stream_in_order = 0x1U;
constexpr int dnnl_f16 = 1;
constexpr int dnnl_bf16 = 2;
constexpr int dnnl_f32 = 3;
constexpr int dnnl_s32 = 4;
constexpr int dnnl_boolean = 8;
constexpr int dnnl_abcd = 4;
constexpr int dnnl_forward_training = 64;
constexpr int dnnl_forward_inference = 96;
constexpr int dnnl_backward = 128;
constexpr int dnnl_backward_data = 160;
constexpr int dnnl_backward_weights = 192;
constexpr int dnnl_convolution_direct = 0x1;
constexpr int dnnl_deconvolution_direct = 0xA;
constexpr int dnnl_eltwise_relu = 0x20;
constexpr int dnnl_eltwise_tanh = 0x21;
constexpr int dnnl_eltwise_linear = 0x26;
constexpr int dnnl_eltwise_logistic = 0x29;
constexpr int dnnl_eltwise_swish = 0x2c;
constexpr int dnnl_eltwise_pow = 0x30;
constexpr int dnnl_lrn_across_channels = 0xaff;
constexpr int dnnl_binary_add = 0x1fff0;
constexpr int dnnl_binary_mul = 0x1fff1;
constexpr int dnnl_binary_max = 0x1fff2;
constexpr int dnnl_binary_min = 0x1fff3;
constexpr int dnnl_reduction_sum = 0x2fff4;
constexpr int dnnl_reduction_mul = 0x2fff5;
constexpr int dnnl_pooling_max = 0x1ff;
constexpr int dnnl_pooling_avg_include_padding = 0x2ff;
constexpr int dnnl_softmax_accurate = 0x30000;
constexpr int dnnl_softmax_log = 0x30001;
constexpr unsigned dnnl_use_global_stats = 0x1U;
constexpr unsigned dnnl_use_scale = 0x2U;
constexpr unsigned dnnl_use_shift = 0x4U;
constexpr unsigned dnnl_normalization_use_scale = 0x2U;
constexpr unsigned dnnl_normalization_use_shift = 0x4U;
constexpr unsigned dnnl_rms_norm = 0x20U;

constexpr int DNNL_ARG_SRC = 1;
constexpr int DNNL_ARG_DST = 17;
constexpr int DNNL_ARG_WEIGHTS = 33;
constexpr int DNNL_ARG_MEAN = 49;
constexpr int DNNL_ARG_VARIANCE = 50;
constexpr int DNNL_ARG_SCALE = 51;
constexpr int DNNL_ARG_SHIFT = 52;
constexpr int DNNL_ARG_SRC_0 = 1;
constexpr int DNNL_ARG_SRC_1 = 2;
constexpr int DNNL_ARG_WORKSPACE = 64;
constexpr int DNNL_ARG_DIFF_SRC = 129;
constexpr int DNNL_ARG_DIFF_DST = 145;
constexpr int DNNL_ARG_DIFF_WEIGHTS = 161;
constexpr int DNNL_ARG_DIFF_SCALE = 255;
constexpr int DNNL_ARG_DIFF_SHIFT = 256;
constexpr int DNNL_ARG_BIAS = 41;
constexpr int dnnl_query_workspace_md = 135;

using dnnl_graph_op_t = void*;
using dnnl_graph_graph_t = void*;
using const_dnnl_graph_graph_t = const void*;
using dnnl_graph_partition_t = void*;
using const_dnnl_graph_partition_t = const void*;
using dnnl_graph_compiled_partition_t = void*;
using const_dnnl_graph_compiled_partition_t = const void*;
using dnnl_graph_tensor_t = void*;
using const_dnnl_graph_tensor_t = const void*;
using dnnl_graph_op_kind_t = int;
using dnnl_graph_op_attr_t = int;
using dnnl_graph_layout_type_t = int;
using dnnl_graph_tensor_property_t = int;
using dnnl_graph_partition_policy_t = int;

constexpr int DNNL_MAX_NDIMS = 12;

struct dnnl_graph_logical_tensor_t {
    std::size_t id;
    int ndims;
    dnnl_dim_t dims[DNNL_MAX_NDIMS];
    dnnl_data_type_t data_type;
    dnnl_graph_tensor_property_t property;
    dnnl_graph_layout_type_t layout_type;
    union {
        dnnl_dim_t strides[DNNL_MAX_NDIMS];
        std::size_t layout_id;
    } layout;
};

constexpr int dnnl_graph_op_add = 0x2;
constexpr int dnnl_graph_op_matmul = 0x27;
constexpr int dnnl_graph_op_multiply = 0x2e;
constexpr int dnnl_graph_op_softmax = 0x40;
constexpr int dnnl_graph_op_select = 0x51;
constexpr int dnnl_graph_op_gen_index = 0x54;
constexpr int dnnl_graph_op_greater_equal = 0x55;
constexpr int dnnl_graph_op_attr_axis = 0x30;
constexpr int dnnl_graph_op_attr_transpose_b = 0x66;
constexpr int dnnl_graph_op_attr_mode = 0x84;
constexpr int dnnl_graph_layout_type_strided = 2;
constexpr int dnnl_graph_tensor_property_variable = 1;
constexpr int dnnl_graph_partition_policy_fusion = 1;

}
