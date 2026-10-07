#include "clap/dnn_dyn_backends.hpp"

#include <dlfcn.h>
#include <initializer_list>
#include <mutex>
#include <string>

namespace clap::dnn_dyn {

#define DEF(name) decltype(name) name = nullptr

DEF(p_dnnl_engine_create); DEF(p_dnnl_engine_destroy); DEF(p_dnnl_stream_create); DEF(p_dnnl_stream_wait); DEF(p_dnnl_stream_destroy);
DEF(p_dnnl_memory_desc_create_with_strides); DEF(p_dnnl_memory_desc_destroy); DEF(p_dnnl_memory_create); DEF(p_dnnl_memory_destroy);
DEF(p_dnnl_primitive_create); DEF(p_dnnl_primitive_execute); DEF(p_dnnl_primitive_destroy); DEF(p_dnnl_primitive_desc_destroy);
DEF(p_dnnl_convolution_forward_primitive_desc_create); DEF(p_dnnl_convolution_backward_data_primitive_desc_create); DEF(p_dnnl_convolution_backward_weights_primitive_desc_create);
DEF(p_dnnl_eltwise_forward_primitive_desc_create); DEF(p_dnnl_eltwise_backward_primitive_desc_create);
DEF(p_dnnl_pooling_forward_primitive_desc_create); DEF(p_dnnl_pooling_backward_primitive_desc_create);
DEF(p_dnnl_softmax_forward_primitive_desc_create); DEF(p_dnnl_softmax_backward_primitive_desc_create);
DEF(p_dnnl_batch_normalization_forward_primitive_desc_create); DEF(p_dnnl_batch_normalization_backward_primitive_desc_create);
DEF(p_dnnl_deconvolution_forward_primitive_desc_create);
DEF(p_dnnl_layer_normalization_forward_primitive_desc_create); DEF(p_dnnl_layer_normalization_backward_primitive_desc_create);
DEF(p_dnnl_primitive_attr_create); DEF(p_dnnl_primitive_attr_destroy); DEF(p_dnnl_primitive_attr_set_dropout);
DEF(p_dnnl_post_ops_create); DEF(p_dnnl_post_ops_destroy); DEF(p_dnnl_post_ops_append_eltwise); DEF(p_dnnl_primitive_attr_set_post_ops); DEF(p_dnnl_lrn_forward_primitive_desc_create); DEF(p_dnnl_lrn_backward_primitive_desc_create); DEF(p_dnnl_binary_primitive_desc_create); DEF(p_dnnl_reduction_primitive_desc_create);
DEF(p_dnnl_stream_get_engine); DEF(p_dnnl_primitive_desc_query_md); DEF(p_dnnl_memory_desc_get_size);
DEF(p_dnnl_layer_normalization_forward_primitive_desc_create_v2); DEF(p_dnnl_layer_normalization_backward_primitive_desc_create_v2);
DEF(p_dnnl_graph_logical_tensor_init_with_strides); DEF(p_dnnl_graph_op_create); DEF(p_dnnl_graph_op_destroy); DEF(p_dnnl_graph_op_add_input); DEF(p_dnnl_graph_op_add_output);
DEF(p_dnnl_graph_op_set_attr_bool); DEF(p_dnnl_graph_op_set_attr_s64); DEF(p_dnnl_graph_op_set_attr_str);
DEF(p_dnnl_graph_graph_create); DEF(p_dnnl_graph_graph_destroy); DEF(p_dnnl_graph_add_op); DEF(p_dnnl_graph_graph_finalize); DEF(p_dnnl_graph_graph_filter);
DEF(p_dnnl_graph_graph_get_partition_num); DEF(p_dnnl_graph_graph_get_partitions); DEF(p_dnnl_graph_partition_destroy); DEF(p_dnnl_graph_partition_is_supported);
DEF(p_dnnl_graph_partition_compile); DEF(p_dnnl_graph_compiled_partition_create); DEF(p_dnnl_graph_compiled_partition_destroy); DEF(p_dnnl_graph_compiled_partition_execute);
DEF(p_dnnl_graph_tensor_create); DEF(p_dnnl_graph_tensor_destroy);

DEF(p_cudaGetDeviceCount); DEF(p_cudaMalloc); DEF(p_cudaFree); DEF(p_cudaMemcpy); DEF(p_cudaDeviceSynchronize); DEF(p_cudaMemset);
DEF(p_cudaGetDevice); DEF(p_cudaStreamCreate); DEF(p_cudaStreamDestroy); DEF(p_cudaStreamSynchronize); DEF(p_cudaMemcpyAsync); DEF(p_cudaMemsetAsync); DEF(p_cudaMallocAsync); DEF(p_cudaFreeAsync);
DEF(p_cuInit); DEF(p_cuModuleLoadData); DEF(p_cuModuleGetFunction); DEF(p_cuLaunchKernel); DEF(p_cuCtxSynchronize); DEF(p_cuModuleUnload);
DEF(p_nvrtcCreateProgram); DEF(p_nvrtcCompileProgram); DEF(p_nvrtcGetPTXSize); DEF(p_nvrtcGetPTX); DEF(p_nvrtcDestroyProgram); DEF(p_nvrtcGetProgramLogSize); DEF(p_nvrtcGetProgramLog);
DEF(p_cudnnCreateLRNDescriptor); DEF(p_cudnnSetLRNDescriptor); DEF(p_cudnnDestroyLRNDescriptor); DEF(p_cudnnLRNCrossChannelForward); DEF(p_cudnnLRNCrossChannelBackward); DEF(p_cudnnCreateOpTensorDescriptor); DEF(p_cudnnSetOpTensorDescriptor); DEF(p_cudnnDestroyOpTensorDescriptor); DEF(p_cudnnOpTensor); DEF(p_cudnnCreateReduceTensorDescriptor); DEF(p_cudnnSetReduceTensorDescriptor); DEF(p_cudnnDestroyReduceTensorDescriptor); DEF(p_cudnnGetReductionWorkspaceSize); DEF(p_cudnnReduceTensor);
DEF(p_cudnnCreate); DEF(p_cudnnDestroy); DEF(p_cudnnCreateTensorDescriptor); DEF(p_cudnnSetTensor4dDescriptor); DEF(p_cudnnDestroyTensorDescriptor);
DEF(p_cudnnCreateFilterDescriptor); DEF(p_cudnnSetFilter4dDescriptor); DEF(p_cudnnDestroyFilterDescriptor);
DEF(p_cudnnCreateConvolutionDescriptor); DEF(p_cudnnSetConvolution2dDescriptor); DEF(p_cudnnDestroyConvolutionDescriptor);
DEF(p_cudnnConvolutionForward); DEF(p_cudnnConvolutionBackwardData); DEF(p_cudnnConvolutionBackwardFilter);
DEF(p_cudnnCreateActivationDescriptor); DEF(p_cudnnSetActivationDescriptor); DEF(p_cudnnDestroyActivationDescriptor); DEF(p_cudnnActivationForward); DEF(p_cudnnActivationBackward);
DEF(p_cudnnCreatePoolingDescriptor); DEF(p_cudnnSetPooling2dDescriptor); DEF(p_cudnnDestroyPoolingDescriptor); DEF(p_cudnnPoolingForward); DEF(p_cudnnPoolingBackward);
DEF(p_cudnnSoftmaxForward); DEF(p_cudnnSoftmaxBackward);
DEF(p_cudnnDeriveBNTensorDescriptor); DEF(p_cudnnBatchNormalizationForwardTraining); DEF(p_cudnnBatchNormalizationForwardInference); DEF(p_cudnnBatchNormalizationBackward);
DEF(p_cudnnConvolutionBackwardBias); DEF(p_cudnnConvolutionBiasActivationForward);
DEF(p_cudnnCreateDropoutDescriptor); DEF(p_cudnnDestroyDropoutDescriptor); DEF(p_cudnnDropoutGetStatesSize); DEF(p_cudnnSetDropoutDescriptor); DEF(p_cudnnDropoutGetReserveSpaceSize); DEF(p_cudnnDropoutForward); DEF(p_cudnnDropoutBackward);
DEF(p_cudnnBackendCreateDescriptor); DEF(p_cudnnBackendDestroyDescriptor); DEF(p_cudnnBackendSetAttribute); DEF(p_cudnnBackendFinalize); DEF(p_cudnnBackendExecute);
DEF(p_cudnnBackendGetAttribute); DEF(p_cudnnGetVersion); DEF(p_cudnnSetStream); DEF(p_cudnnSetTensorNdDescriptor); DEF(p_cudnnSetTensor4dDescriptorEx); DEF(p_cudnnSetActivationDescriptorSwishBeta);

DEF(p_hipGetDeviceCount); DEF(p_hipMalloc); DEF(p_hipFree); DEF(p_hipMemcpy); DEF(p_hipDeviceSynchronize);
DEF(p_hipGetDevice); DEF(p_hipStreamCreate); DEF(p_hipStreamDestroy); DEF(p_hipStreamSynchronize); DEF(p_hipMemcpyAsync); DEF(p_hipMemsetAsync); DEF(p_hipMallocAsync); DEF(p_hipFreeAsync);
DEF(p_miopenSetStream); DEF(p_miopenSetTensorDescriptor);
DEF(p_miopenCreate); DEF(p_miopenDestroy); DEF(p_miopenCreateTensorDescriptor); DEF(p_miopenSet4dTensorDescriptor); DEF(p_miopenDestroyTensorDescriptor);
DEF(p_miopenCreateConvolutionDescriptor); DEF(p_miopenInitConvolutionDescriptor); DEF(p_miopenDestroyConvolutionDescriptor);
DEF(p_miopenConvolutionForward); DEF(p_miopenConvolutionBackwardData); DEF(p_miopenConvolutionBackwardWeights);
DEF(p_miopenCreateLRNDescriptor); DEF(p_miopenSetLRNDescriptor); DEF(p_miopenDestroyLRNDescriptor); DEF(p_miopenLRNForward); DEF(p_miopenLRNBackward); DEF(p_miopenLRNGetWorkSpaceSize); DEF(p_miopenOpTensor); DEF(p_miopenCreateReduceTensorDescriptor); DEF(p_miopenDestroyReduceTensorDescriptor); DEF(p_miopenSetReduceTensorDescriptor); DEF(p_miopenGetReductionWorkspaceSize); DEF(p_miopenReduceTensor); DEF(p_miopenCreateActivationDescriptor); DEF(p_miopenSetActivationDescriptor); DEF(p_miopenDestroyActivationDescriptor); DEF(p_miopenActivationForward); DEF(p_miopenActivationBackward);
DEF(p_miopenCreatePoolingDescriptor); DEF(p_miopenSet2dPoolingDescriptor); DEF(p_miopenDestroyPoolingDescriptor); DEF(p_miopenPoolingForward); DEF(p_miopenPoolingBackward);
DEF(p_miopenSoftmaxForward_V2); DEF(p_miopenSoftmaxBackward_V2);
DEF(p_miopenBatchNormalizationForwardTraining); DEF(p_miopenBatchNormalizationForwardInference); DEF(p_miopenBatchNormalizationBackward);
DEF(p_miopenConvolutionBackwardBias); DEF(p_miopenConvolutionForwardBias);
DEF(p_miopenLayerNormForward); DEF(p_miopenGetLayerNormBackwardWorkspaceSize); DEF(p_miopenLayerNormBackward);
DEF(p_miopenCreateDropoutDescriptor); DEF(p_miopenDestroyDropoutDescriptor); DEF(p_miopenDropoutGetStatesSize); DEF(p_miopenSetDropoutDescriptor); DEF(p_miopenDropoutGetReserveSpaceSize); DEF(p_miopenDropoutForward); DEF(p_miopenDropoutBackward);
DEF(p_miopenT5LayerNormForward); DEF(p_miopenGetT5LayerNormBackwardWorkspaceSize); DEF(p_miopenT5LayerNormBackward);
DEF(p_miopenCreateMhaDescriptor); DEF(p_miopenSetMhaDescriptor); DEF(p_miopenCreateMhaProblem); DEF(p_miopenSetProblemTensorDescriptor);
DEF(p_miopenFindSolutions); DEF(p_miopenGetSolutionWorkspaceSize); DEF(p_miopenRunSolution); DEF(p_miopenDestroySolution); DEF(p_miopenDestroyProblem);

#undef DEF

namespace {
std::mutex one_mutex, cuda_mutex, hip_mutex;
void* one_lib = nullptr; void* cuda_lib = nullptr; void* cudnn_lib = nullptr; void* hip_lib = nullptr; void* miopen_lib = nullptr;
void* cuda_driver_lib = nullptr; void* nvrtc_lib = nullptr;
bool one_done=false, cuda_done=false, hip_done=false, cuda_jit_done=false;
bool one_graph=false, miopen_t5=false, miopen_mha=false;
std::mutex jit_mutex;
std::string error_text;

void* open_any(std::initializer_list<const char*> names) {
    for (auto n : names) {
        if (void* h = dlopen(n, RTLD_LAZY | RTLD_LOCAL)) return h; //dlopen("onednn.so",RTLD_LAZY)
    }
    return nullptr;
}

template<class T>
bool sym(void* h, const char* n, T& out) {
	//h = cublas_lib ,n= name of rotine , out = function pointer for routine
	//main dlsym handling 
    out = reinterpret_cast<T>(dlsym(h, n));
    if (!out) { error_text = std::string("missing symbol ") + n; return false; }
    return true;
}

// Optional symbol: a missing entry point leaves the pointer null and only
// disables the feature group that needs it.
template<class T>
bool symOptional(void* h, const char* n, T& out) {
    out = reinterpret_cast<T>(dlsym(h, n));
    return out != nullptr;
}
}

const char* lastError() { return error_text.c_str(); }

bool loadOneDnn() {
    std::lock_guard<std::mutex> lock(one_mutex);
    if (one_done) return one_lib != nullptr;
    one_done=true;
    one_lib=open_any({"libdnnl.so", "libdnnl.so.3", "libdnnl.so.2"});//dlopen 
    if (!one_lib) { error_text="cannot dlopen libdnnl.so; set LD_LIBRARY_PATH to oneDNN lib directory"; return false; }

#define S(x) if(!sym(one_lib,#x,p_##x)) return false   //dlsym   

    S(dnnl_engine_create); S(dnnl_engine_destroy); S(dnnl_stream_create); S(dnnl_stream_wait); S(dnnl_stream_destroy); //SYM(cublas_lib, cublas_gemm_v)
    S(dnnl_memory_desc_create_with_strides); S(dnnl_memory_desc_destroy); S(dnnl_memory_create); S(dnnl_memory_destroy);
    S(dnnl_primitive_create); S(dnnl_primitive_execute); S(dnnl_primitive_destroy); S(dnnl_primitive_desc_destroy);
    S(dnnl_convolution_forward_primitive_desc_create); S(dnnl_convolution_backward_data_primitive_desc_create); S(dnnl_convolution_backward_weights_primitive_desc_create);
    S(dnnl_eltwise_forward_primitive_desc_create); S(dnnl_eltwise_backward_primitive_desc_create);
    S(dnnl_pooling_forward_primitive_desc_create); S(dnnl_pooling_backward_primitive_desc_create);
    S(dnnl_softmax_forward_primitive_desc_create); S(dnnl_softmax_backward_primitive_desc_create);
    S(dnnl_batch_normalization_forward_primitive_desc_create); S(dnnl_batch_normalization_backward_primitive_desc_create);
    S(dnnl_deconvolution_forward_primitive_desc_create);
    S(dnnl_layer_normalization_forward_primitive_desc_create); S(dnnl_layer_normalization_backward_primitive_desc_create);
    S(dnnl_primitive_attr_create); S(dnnl_primitive_attr_destroy); S(dnnl_primitive_attr_set_dropout); S(dnnl_lrn_forward_primitive_desc_create); S(dnnl_lrn_backward_primitive_desc_create); S(dnnl_binary_primitive_desc_create); S(dnnl_reduction_primitive_desc_create);
    S(dnnl_post_ops_create); S(dnnl_post_ops_destroy); S(dnnl_post_ops_append_eltwise); S(dnnl_primitive_attr_set_post_ops);
    S(dnnl_stream_get_engine); S(dnnl_primitive_desc_query_md); S(dnnl_memory_desc_get_size);
#undef S

#define SO(x) ok = symOptional(one_lib,#x,p_##x) && ok
    bool ok = true;
    SO(dnnl_layer_normalization_forward_primitive_desc_create_v2); SO(dnnl_layer_normalization_backward_primitive_desc_create_v2);
    ok = true;
    SO(dnnl_graph_logical_tensor_init_with_strides); SO(dnnl_graph_op_create); SO(dnnl_graph_op_destroy); SO(dnnl_graph_op_add_input); SO(dnnl_graph_op_add_output);
    SO(dnnl_graph_op_set_attr_bool); SO(dnnl_graph_op_set_attr_s64); SO(dnnl_graph_op_set_attr_str);
    SO(dnnl_graph_graph_create); SO(dnnl_graph_graph_destroy); SO(dnnl_graph_add_op); SO(dnnl_graph_graph_finalize); SO(dnnl_graph_graph_filter);
    SO(dnnl_graph_graph_get_partition_num); SO(dnnl_graph_graph_get_partitions); SO(dnnl_graph_partition_destroy); SO(dnnl_graph_partition_is_supported);
    SO(dnnl_graph_partition_compile); SO(dnnl_graph_compiled_partition_create); SO(dnnl_graph_compiled_partition_destroy); SO(dnnl_graph_compiled_partition_execute);
    SO(dnnl_graph_tensor_create); SO(dnnl_graph_tensor_destroy);
#undef SO
    one_graph = ok;
    return true;
}

bool loadCudaAndCudnn() {
    std::lock_guard<std::mutex> lock(cuda_mutex);
    if (cuda_done) return cuda_lib && cudnn_lib;
    cuda_done=true;
    cuda_lib=open_any({"libcudart.so", "libcudart.so.12", "libcudart.so.13"});
    if (!cuda_lib) { error_text="cannot dlopen libcudart.so"; return false; }
#define S0(x) if(!sym(cuda_lib,#x,p_##x)) return false
    S0(cudaGetDeviceCount); S0(cudaMalloc); S0(cudaFree); S0(cudaMemcpy); S0(cudaDeviceSynchronize); S0(cudaMemset);
    S0(cudaGetDevice); S0(cudaStreamCreate); S0(cudaStreamDestroy); S0(cudaStreamSynchronize); S0(cudaMemcpyAsync); S0(cudaMemsetAsync); S0(cudaMallocAsync); S0(cudaFreeAsync);
#undef S0
    int count=0; if(p_cudaGetDeviceCount(&count)!=0 || count<=0) { error_text="CUDA runtime loaded but no NVIDIA GPU is available"; return false; }
    cudnn_lib=open_any({"libcudnn.so", "libcudnn.so.9"});
    if (!cudnn_lib) { error_text="cannot dlopen libcudnn.so; set LD_LIBRARY_PATH to cuDNN lib directory"; return false; }
#define S(x) if(!sym(cudnn_lib,#x,p_##x)) return false
    S(cudnnCreate); S(cudnnDestroy); S(cudnnCreateTensorDescriptor); S(cudnnSetTensor4dDescriptor); S(cudnnDestroyTensorDescriptor);
    S(cudnnCreateFilterDescriptor); S(cudnnSetFilter4dDescriptor); S(cudnnDestroyFilterDescriptor);
    S(cudnnCreateConvolutionDescriptor); S(cudnnSetConvolution2dDescriptor); S(cudnnDestroyConvolutionDescriptor);
    S(cudnnConvolutionForward); S(cudnnConvolutionBackwardData); S(cudnnConvolutionBackwardFilter);
    S(cudnnCreateActivationDescriptor); S(cudnnSetActivationDescriptor); S(cudnnDestroyActivationDescriptor); S(cudnnActivationForward); S(cudnnActivationBackward);
    S(cudnnCreatePoolingDescriptor); S(cudnnSetPooling2dDescriptor); S(cudnnDestroyPoolingDescriptor); S(cudnnPoolingForward); S(cudnnPoolingBackward);
    S(cudnnSoftmaxForward); S(cudnnSoftmaxBackward); S(cudnnCreateLRNDescriptor); S(cudnnSetLRNDescriptor); S(cudnnDestroyLRNDescriptor); S(cudnnLRNCrossChannelForward); S(cudnnLRNCrossChannelBackward); S(cudnnCreateOpTensorDescriptor); S(cudnnSetOpTensorDescriptor); S(cudnnDestroyOpTensorDescriptor); S(cudnnOpTensor); S(cudnnCreateReduceTensorDescriptor); S(cudnnSetReduceTensorDescriptor); S(cudnnDestroyReduceTensorDescriptor); S(cudnnGetReductionWorkspaceSize); S(cudnnReduceTensor);
    S(cudnnDeriveBNTensorDescriptor); S(cudnnBatchNormalizationForwardTraining); S(cudnnBatchNormalizationForwardInference); S(cudnnBatchNormalizationBackward);
    S(cudnnConvolutionBackwardBias); S(cudnnConvolutionBiasActivationForward);
    S(cudnnCreateDropoutDescriptor); S(cudnnDestroyDropoutDescriptor); S(cudnnDropoutGetStatesSize); S(cudnnSetDropoutDescriptor); S(cudnnDropoutGetReserveSpaceSize); S(cudnnDropoutForward); S(cudnnDropoutBackward);
    S(cudnnBackendCreateDescriptor); S(cudnnBackendDestroyDescriptor); S(cudnnBackendSetAttribute); S(cudnnBackendFinalize); S(cudnnBackendExecute);
    S(cudnnBackendGetAttribute); S(cudnnGetVersion); S(cudnnSetStream); S(cudnnSetTensorNdDescriptor); S(cudnnSetTensor4dDescriptorEx); S(cudnnSetActivationDescriptorSwishBeta);
#undef S
    return true;
}

bool loadCudaJit() {
    std::lock_guard<std::mutex> lock(jit_mutex);
    if (cuda_jit_done) return cuda_driver_lib && nvrtc_lib;
    cuda_jit_done = true;

    if (!loadCudaAndCudnn()) return false;

    cuda_driver_lib = open_any({"libcuda.so.1", "libcuda.so"});
    if (!cuda_driver_lib) {
        error_text = "cannot dlopen libcuda.so.1 required for CUDA JIT LayerNorm";
        return false;
    }

    nvrtc_lib = open_any({"libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.13"});
    if (!nvrtc_lib) {
        error_text = "cannot dlopen libnvrtc.so required for CUDA JIT LayerNorm";
        return false;
    }

#define SD(x) if(!sym(cuda_driver_lib,#x,p_##x)) return false
    SD(cuInit); SD(cuModuleLoadData); SD(cuModuleGetFunction); SD(cuLaunchKernel); SD(cuCtxSynchronize); SD(cuModuleUnload);
#undef SD

#define SN(x) if(!sym(nvrtc_lib,#x,p_##x)) return false
    SN(nvrtcCreateProgram); SN(nvrtcCompileProgram); SN(nvrtcGetPTXSize); SN(nvrtcGetPTX); SN(nvrtcDestroyProgram); SN(nvrtcGetProgramLogSize); SN(nvrtcGetProgramLog);
#undef SN

    if (p_cuInit(0) != abi::CUDA_SUCCESS) {
        error_text = "cuInit failed for CUDA JIT LayerNorm";
        return false;
    }

    return true;
}

bool loadHipAndMiopen() {
    std::lock_guard<std::mutex> lock(hip_mutex);
    if (hip_done) return hip_lib && miopen_lib;
    hip_done=true;
    hip_lib=open_any({"libamdhip64.so", "libamdhip64.so.6"});
    if (!hip_lib) { error_text="cannot dlopen libamdhip64.so"; return false; }
#define S0(x) if(!sym(hip_lib,#x,p_##x)) return false
    S0(hipGetDeviceCount); S0(hipMalloc); S0(hipFree); S0(hipMemcpy); S0(hipDeviceSynchronize);
    S0(hipGetDevice); S0(hipStreamCreate); S0(hipStreamDestroy); S0(hipStreamSynchronize); S0(hipMemcpyAsync); S0(hipMemsetAsync); S0(hipMallocAsync); S0(hipFreeAsync);
#undef S0
    int count=0; if(p_hipGetDeviceCount(&count)!=0 || count<=0) { error_text="HIP runtime loaded but no AMD GPU is available"; return false; }
    miopen_lib=open_any({"libMIOpen.so", "libMIOpen.so.1"});
    if (!miopen_lib) { error_text="cannot dlopen libMIOpen.so; set LD_LIBRARY_PATH to ROCm/MIOpen lib directory"; return false; }
#define S(x) if(!sym(miopen_lib,#x,p_##x)) return false
    S(miopenCreate); S(miopenDestroy); S(miopenCreateTensorDescriptor); S(miopenSet4dTensorDescriptor); S(miopenDestroyTensorDescriptor);
    S(miopenCreateConvolutionDescriptor); S(miopenInitConvolutionDescriptor); S(miopenDestroyConvolutionDescriptor);
    S(miopenConvolutionForward); S(miopenConvolutionBackwardData); S(miopenConvolutionBackwardWeights);
    S(miopenCreateActivationDescriptor); S(miopenSetActivationDescriptor); S(miopenDestroyActivationDescriptor); S(miopenActivationForward); S(miopenActivationBackward);
    S(miopenCreatePoolingDescriptor); S(miopenSet2dPoolingDescriptor); S(miopenDestroyPoolingDescriptor); S(miopenPoolingForward); S(miopenPoolingBackward);
    S(miopenSoftmaxForward_V2); S(miopenSoftmaxBackward_V2); S(miopenCreateLRNDescriptor); S(miopenSetLRNDescriptor); S(miopenDestroyLRNDescriptor); S(miopenLRNForward); S(miopenLRNBackward); S(miopenLRNGetWorkSpaceSize); S(miopenOpTensor); S(miopenCreateReduceTensorDescriptor); S(miopenDestroyReduceTensorDescriptor); S(miopenSetReduceTensorDescriptor); S(miopenGetReductionWorkspaceSize); S(miopenReduceTensor);
    S(miopenBatchNormalizationForwardTraining); S(miopenBatchNormalizationForwardInference); S(miopenBatchNormalizationBackward);
    S(miopenConvolutionBackwardBias); S(miopenConvolutionForwardBias);
    S(miopenLayerNormForward); S(miopenGetLayerNormBackwardWorkspaceSize); S(miopenLayerNormBackward);
    S(miopenCreateDropoutDescriptor); S(miopenDestroyDropoutDescriptor); S(miopenDropoutGetStatesSize); S(miopenSetDropoutDescriptor); S(miopenDropoutGetReserveSpaceSize); S(miopenDropoutForward); S(miopenDropoutBackward);
    S(miopenSetStream); S(miopenSetTensorDescriptor);
#undef S

#define SO(x) ok = symOptional(miopen_lib,#x,p_##x) && ok
    bool ok = true;
    SO(miopenT5LayerNormForward); SO(miopenGetT5LayerNormBackwardWorkspaceSize); SO(miopenT5LayerNormBackward);
    miopen_t5 = ok;

    ok = true;
    SO(miopenCreateMhaDescriptor); SO(miopenSetMhaDescriptor); SO(miopenCreateMhaProblem); SO(miopenSetProblemTensorDescriptor);
    SO(miopenFindSolutions); SO(miopenGetSolutionWorkspaceSize); SO(miopenRunSolution); SO(miopenDestroySolution); SO(miopenDestroyProblem);
    miopen_mha = ok;
#undef SO
    return true;
}

bool hasNvidiaGpu() { return loadCudaAndCudnn(); }
bool hasAmdGpu() { return loadHipAndMiopen(); }

bool hasOneDnnGraph() { return loadOneDnn() && one_graph; }
bool hasMiopenT5LayerNorm() { return loadHipAndMiopen() && miopen_t5; }
bool hasMiopenMha() { return loadHipAndMiopen() && miopen_mha; }

} // namespace clap::dnn_dyn
