#pragma once

#include "dnn_types.hpp"

#include <cstddef>
#include <cstdint>

namespace clap {

class IDnnBackend {
public:
    virtual ~IDnnBackend() = default;

    virtual const char* name() const noexcept = 0;

    virtual void convolutionForward(
        const TensorDesc& x_desc, const float* x,
        const TensorDesc& w_desc, const float* w,
        const ConvolutionDesc& conv_desc,
        const TensorDesc& y_desc, float* y) = 0;

    virtual void convolutionBackwardData(
        const TensorDesc& dy_desc, const float* dy,
        const TensorDesc& w_desc, const float* w,
        const ConvolutionDesc& conv_desc,
        const TensorDesc& dx_desc, float* dx) = 0;

    virtual void convolutionBackwardWeights(
        const TensorDesc& x_desc, const float* x,
        const TensorDesc& dy_desc, const float* dy,
        const ConvolutionDesc& conv_desc,
        const TensorDesc& dw_desc, float* dw) = 0;

    virtual void convolutionBackwardBias(
        const TensorDesc& dy_desc, const float* dy,
        const TensorDesc& db_desc, float* db) = 0;

    virtual void convolutionTransposeForward(
        const TensorDesc& x_desc, const float* x,
        const TensorDesc& w_desc, const float* w,
        const ConvolutionDesc& conv_desc,
        const TensorDesc& y_desc, float* y) = 0;

    virtual void fusedConvolutionBiasActivation(
        const TensorDesc& x_desc, const float* x,
        const TensorDesc& w_desc, const float* w,
        const float* bias,
        const ConvolutionDesc& conv_desc,
        const ActivationDesc& activation,
        const TensorDesc& y_desc, float* y) = 0;

    virtual void activationForward(
        const ActivationDesc& activation,
        const TensorDesc& x_desc, const float* x,
        const TensorDesc& y_desc, float* y) = 0;

    virtual void activationBackward(
        const ActivationDesc& activation,
        const TensorDesc& x_desc, const float* x,
        const TensorDesc& dy_desc, const float* dy,
        const TensorDesc& dx_desc, float* dx) = 0;

    virtual void poolingForward(
        const PoolingDesc& pooling,
        const TensorDesc& x_desc, const float* x,
        const TensorDesc& y_desc, float* y) = 0;

    virtual void poolingBackward(
        const PoolingDesc& pooling,
        const TensorDesc& x_desc, const float* x,
        const TensorDesc& y_desc, const float* y,
        const TensorDesc& dy_desc, const float* dy,
        const TensorDesc& dx_desc, float* dx) = 0;

    virtual void softmaxForward(
        const TensorDesc& x_desc, const float* x,
        const TensorDesc& y_desc, float* y) = 0;

    virtual void softmaxBackward(
        const TensorDesc& y_desc, const float* y,
        const TensorDesc& dy_desc, const float* dy,
        const TensorDesc& dx_desc, float* dx) = 0;

    virtual void batchNormForwardTraining(
        const BatchNormDesc& bn,
        const TensorDesc& x_desc, const float* x,
        const float* scale, const float* bias,
        float* running_mean, float* running_variance,
        float* saved_mean, float* saved_variance,
        const TensorDesc& y_desc, float* y) = 0;

    virtual void batchNormForwardInference(
        const BatchNormDesc& bn,
        const TensorDesc& x_desc, const float* x,
        const float* scale, const float* bias,
        const float* mean, const float* variance,
        const TensorDesc& y_desc, float* y) = 0;

    virtual void batchNormBackward(
        const BatchNormDesc& bn,
        const TensorDesc& x_desc, const float* x,
        const TensorDesc& dy_desc, const float* dy,
        const float* scale,
        const float* saved_mean, const float* saved_variance,
        const TensorDesc& dx_desc, float* dx,
        float* dscale, float* dbias) = 0;

    virtual void layerNormForward(
        const LayerNormDesc& ln,
        const TensorDesc& x_desc, const float* x,
        const float* scale, const float* bias,
        const TensorDesc& y_desc, float* y,
        float* saved_mean, float* saved_rstd) = 0;

    virtual void layerNormBackward(
        const LayerNormDesc& ln,
        const TensorDesc& x_desc, const float* x,
        const TensorDesc& dy_desc, const float* dy,
        const float* scale,
        const float* saved_mean, const float* saved_rstd,
        const TensorDesc& dx_desc, float* dx,
        float* dscale, float* dbias) = 0;

    virtual void dropoutForward(
        const DropoutDesc& dropout,
        const TensorDesc& x_desc, const float* x,
        const TensorDesc& y_desc, float* y,
        std::uint8_t* mask) = 0;

    virtual void dropoutBackward(
        const DropoutDesc& dropout,
        const TensorDesc& dy_desc, const float* dy,
        const std::uint8_t* mask,
        const TensorDesc& dx_desc, float* dx) = 0;

    virtual void lrnForward(const LrnDesc&, const TensorDesc&, const float*, const TensorDesc&, float*) = 0;
    virtual void lrnBackward(const LrnDesc&, const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) = 0;
    virtual void tensorAdd(const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) = 0;
    virtual void tensorMultiply(const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) = 0;
    virtual void tensorMin(const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) = 0;
    virtual void tensorMax(const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) = 0;
    virtual void tensorReduceSum(const TensorDesc&, const float*, const TensorDesc&, float*) = 0;
    virtual void tensorReduceProduct(const TensorDesc&, const float*, const TensorDesc&, float*) = 0;

    // ======================================================================
    // Framework execution contract
    // ======================================================================
    //
    // The overloads below take untyped tensor pointers plus an
    // ExecutionContext.  The TensorDesc carries the datatype (Float32,
    // Float16, BFloat16), the dimensions (any rank where the operation
    // allows it) and optional element strides.
    //
    // For ExecutionBackend::CUDA / ROCM every tensor pointer is a device
    // pointer owned by the caller.  The backend binds its vendor handle to
    // ctx.stream, enqueues the vendor primitive on that stream and returns.
    // It does not copy tensors through host memory, does not allocate
    // duplicate input/output buffers and does not synchronize the device.
    //
    // For ExecutionBackend::CPU the pointers are host pointers and the
    // oneDNN backend executes on ctx.stream (a dnnl_stream_t) when given,
    // otherwise on its own stream.
    //
    // The float* overloads above keep their original host-pointer behavior.
    // On GPU backends they stage through device memory and then call the
    // corresponding overload below.
    //
    // Unless documented otherwise, auxiliary parameter arrays (batch-norm
    // scale/bias/statistics, normalization statistics) are Float32.

    virtual ExecutionBackend executionBackend() const noexcept = 0;

    virtual DnnSupport supports(
        const DnnCapabilityQuery& query,
        const ExecutionContext& ctx) const = 0;

    virtual void convolutionForward(
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& w_desc, const void* w,
        const ConvolutionDesc& conv_desc,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    virtual void convolutionBackwardData(
        const TensorDesc& dy_desc, const void* dy,
        const TensorDesc& w_desc, const void* w,
        const ConvolutionDesc& conv_desc,
        const TensorDesc& dx_desc, void* dx,
        const ExecutionContext& ctx) = 0;

    virtual void convolutionBackwardWeights(
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& dy_desc, const void* dy,
        const ConvolutionDesc& conv_desc,
        const TensorDesc& dw_desc, void* dw,
        const ExecutionContext& ctx) = 0;

    virtual void convolutionBackwardBias(
        const TensorDesc& dy_desc, const void* dy,
        const TensorDesc& db_desc, void* db,
        const ExecutionContext& ctx) = 0;

    virtual void convolutionTransposeForward(
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& w_desc, const void* w,
        const ConvolutionDesc& conv_desc,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    // bias: one value per output channel, same datatype as y.
    virtual void fusedConvolutionBiasActivation(
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& w_desc, const void* w,
        const void* bias,
        const ConvolutionDesc& conv_desc,
        const ActivationDesc& activation,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    virtual void activationForward(
        const ActivationDesc& activation,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    virtual void activationBackward(
        const ActivationDesc& activation,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& dy_desc, const void* dy,
        const TensorDesc& dx_desc, void* dx,
        const ExecutionContext& ctx) = 0;

    virtual void poolingForward(
        const PoolingDesc& pooling,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    virtual void poolingBackward(
        const PoolingDesc& pooling,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& y_desc, const void* y,
        const TensorDesc& dy_desc, const void* dy,
        const TensorDesc& dx_desc, void* dx,
        const ExecutionContext& ctx) = 0;

    // Axis-aware Softmax / LogSoftmax (SoftmaxDesc::log_softmax).
    virtual void softmaxForward(
        const SoftmaxDesc& softmax,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    // y is the forward output (softmax or log-softmax values).
    virtual void softmaxBackward(
        const SoftmaxDesc& softmax,
        const TensorDesc& y_desc, const void* y,
        const TensorDesc& dy_desc, const void* dy,
        const TensorDesc& dx_desc, void* dx,
        const ExecutionContext& ctx) = 0;

    // saved_mean/saved_variance are backend-native statistics; pass them
    // unchanged to batchNormBackward() of the same backend.
    virtual void batchNormForwardTraining(
        const BatchNormDesc& bn,
        const TensorDesc& x_desc, const void* x,
        const void* scale, const void* bias,
        void* running_mean, void* running_variance,
        void* saved_mean, void* saved_variance,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    virtual void batchNormForwardInference(
        const BatchNormDesc& bn,
        const TensorDesc& x_desc, const void* x,
        const void* scale, const void* bias,
        const void* mean, const void* variance,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    virtual void batchNormBackward(
        const BatchNormDesc& bn,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& dy_desc, const void* dy,
        const void* scale,
        const void* saved_mean, const void* saved_variance,
        const TensorDesc& dx_desc, void* dx,
        void* dscale, void* dbias,
        const ExecutionContext& ctx) = 0;

    // LayerNorm over the last dimension.  scale/bias/dscale/dbias follow
    // scale_desc (shape [inner]); saved_mean/saved_rstd are Float32 with one
    // value per normalized row and may be null in forward.
    virtual void layerNormForward(
        const LayerNormDesc& ln,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& scale_desc, const void* scale, const void* bias,
        const TensorDesc& y_desc, void* y,
        void* saved_mean, void* saved_rstd,
        const ExecutionContext& ctx) = 0;

    virtual void layerNormBackward(
        const LayerNormDesc& ln,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& dy_desc, const void* dy,
        const TensorDesc& scale_desc, const void* scale,
        const void* saved_mean, const void* saved_rstd,
        const TensorDesc& dx_desc, void* dx,
        void* dscale, void* dbias,
        const ExecutionContext& ctx) = 0;

    // RMSNorm (not LayerNorm).  weight/dweight follow weight_desc (shape
    // [inner]); saved_rstd is Float32 with one value per normalized row and
    // may be null in forward (inference).
    virtual void rmsNormForward(
        const RmsNormDesc& norm,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& weight_desc, const void* weight,
        const TensorDesc& y_desc, void* y,
        void* saved_rstd,
        const ExecutionContext& ctx) = 0;

    virtual void rmsNormBackward(
        const RmsNormDesc& norm,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& dy_desc, const void* dy,
        const TensorDesc& weight_desc, const void* weight,
        const void* saved_rstd,
        const TensorDesc& dx_desc, void* dx,
        void* dweight,
        const ExecutionContext& ctx) = 0;

    // Dropout keeps its mask in a backend-specific reserve space owned by the
    // caller.  Query its size, pass it to dropoutForward() and hand the same
    // buffer to dropoutBackward().
    virtual std::size_t dropoutReserveSpaceSize(
        const TensorDesc& x_desc,
        const ExecutionContext& ctx) = 0;

    virtual void dropoutForward(
        const DropoutDesc& dropout,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& y_desc, void* y,
        void* reserve_space, std::size_t reserve_space_bytes,
        const ExecutionContext& ctx) = 0;

    virtual void dropoutBackward(
        const DropoutDesc& dropout,
        const TensorDesc& dy_desc, const void* dy,
        const void* reserve_space, std::size_t reserve_space_bytes,
        const TensorDesc& dx_desc, void* dx,
        const ExecutionContext& ctx) = 0;

    virtual void lrnForward(
        const LrnDesc& lrn,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    virtual void lrnBackward(
        const LrnDesc& lrn,
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& y_desc, const void* y,
        const TensorDesc& dy_desc, const void* dy,
        const TensorDesc& dx_desc, void* dx,
        const ExecutionContext& ctx) = 0;

    virtual void tensorAdd(
        const TensorDesc& a_desc, const void* a,
        const TensorDesc& b_desc, const void* b,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    virtual void tensorMultiply(
        const TensorDesc& a_desc, const void* a,
        const TensorDesc& b_desc, const void* b,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    virtual void tensorMin(
        const TensorDesc& a_desc, const void* a,
        const TensorDesc& b_desc, const void* b,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    virtual void tensorMax(
        const TensorDesc& a_desc, const void* a,
        const TensorDesc& b_desc, const void* b,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    virtual void tensorReduceSum(
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    virtual void tensorReduceProduct(
        const TensorDesc& x_desc, const void* x,
        const TensorDesc& y_desc, void* y,
        const ExecutionContext& ctx) = 0;

    // Scaled dot-product attention forward (see AttentionDesc for shapes).
    // mask_desc/mask are optional (both null when absent); the mask is
    // additive and broadcastable to [B, Hq, Sq, Skv].
    virtual void scaledDotProductAttentionForward(
        const AttentionDesc& attention,
        const TensorDesc& q_desc, const void* q,
        const TensorDesc& k_desc, const void* k,
        const TensorDesc& v_desc, const void* v,
        const TensorDesc* mask_desc, const void* mask,
        const TensorDesc& o_desc, void* o,
        const ExecutionContext& ctx) = 0;
};

} // namespace clap
