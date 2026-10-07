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
#include "idnn_backend.hpp"

#include <cstddef>
#include <memory>

namespace clap {

class MiOpenBackend final : public IDnnBackend {
public:
    MiOpenBackend();
    ~MiOpenBackend() override;

    const char* name() const noexcept override;

    void convolutionForward(const TensorDesc&, const float*, const TensorDesc&, const float*, const ConvolutionDesc&, const TensorDesc&, float*) override;

    void convolutionBackwardData(const TensorDesc&, const float*, const TensorDesc&, const float*, const ConvolutionDesc&, const TensorDesc&, float*) override;

    void convolutionBackwardWeights(const TensorDesc&, const float*, const TensorDesc&, const float*, const ConvolutionDesc&, const TensorDesc&, float*) override;

    void convolutionBackwardBias(const TensorDesc&, const float*, const TensorDesc&, float*) override;

    void convolutionTransposeForward(const TensorDesc&, const float*, const TensorDesc&, const float*, const ConvolutionDesc&, const TensorDesc&, float*) override;

    void fusedConvolutionBiasActivation(const TensorDesc&, const float*, const TensorDesc&, const float*, const float*, const ConvolutionDesc&, const ActivationDesc&, const TensorDesc&, float*) override;

    void activationForward(const ActivationDesc&, const TensorDesc&, const float*, const TensorDesc&, float*) override;

    void activationBackward(const ActivationDesc&, const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) override;

    void poolingForward(const PoolingDesc&, const TensorDesc&, const float*, const TensorDesc&, float*) override;

    void poolingBackward(const PoolingDesc&, const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) override;

    void softmaxForward(const TensorDesc&, const float*, const TensorDesc&, float*) override;

    void softmaxBackward(const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) override;

    void batchNormForwardTraining(const BatchNormDesc&, const TensorDesc&, const float*, const float*, const float*, float*, float*, float*, float*, const TensorDesc&, float*) override;

    void batchNormForwardInference(const BatchNormDesc&, const TensorDesc&, const float*, const float*, const float*, const float*, const float*, const TensorDesc&, float*) override;

    void batchNormBackward(const BatchNormDesc&, const TensorDesc&, const float*, const TensorDesc&, const float*, const float*, const float*, const float*, const TensorDesc&, float*, float*, float*) override;

    void layerNormForward(const LayerNormDesc&, const TensorDesc&, const float*, const float*, const float*, const TensorDesc&, float*, float*, float*) override;

    void layerNormBackward(const LayerNormDesc&, const TensorDesc&, const float*, const TensorDesc&, const float*, const float*, const float*, const float*, const TensorDesc&, float*, float*, float*) override;

    void dropoutForward(const DropoutDesc&, const TensorDesc&, const float*, const TensorDesc&, float*, std::uint8_t*) override;

    void dropoutBackward(const DropoutDesc&, const TensorDesc&, const float*, const std::uint8_t*, const TensorDesc&, float*) override;

    void lrnForward(const LrnDesc&, const TensorDesc&, const float*, const TensorDesc&, float*) override;
    void lrnBackward(const LrnDesc&, const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) override;
    void tensorAdd(const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) override;
    void tensorMultiply(const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) override;
    void tensorMin(const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) override;
    void tensorMax(const TensorDesc&, const float*, const TensorDesc&, const float*, const TensorDesc&, float*) override;
    void tensorReduceSum(const TensorDesc&, const float*, const TensorDesc&, float*) override;
    void tensorReduceProduct(const TensorDesc&, const float*, const TensorDesc&, float*) override;

    ExecutionBackend executionBackend() const noexcept override;
    DnnSupport supports(const DnnCapabilityQuery&, const ExecutionContext&) const override;

    void convolutionForward(const TensorDesc&, const void*, const TensorDesc&, const void*, const ConvolutionDesc&, const TensorDesc&, void*, const ExecutionContext&) override;
    void convolutionBackwardData(const TensorDesc&, const void*, const TensorDesc&, const void*, const ConvolutionDesc&, const TensorDesc&, void*, const ExecutionContext&) override;
    void convolutionBackwardWeights(const TensorDesc&, const void*, const TensorDesc&, const void*, const ConvolutionDesc&, const TensorDesc&, void*, const ExecutionContext&) override;
    void convolutionBackwardBias(const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void convolutionTransposeForward(const TensorDesc&, const void*, const TensorDesc&, const void*, const ConvolutionDesc&, const TensorDesc&, void*, const ExecutionContext&) override;
    void fusedConvolutionBiasActivation(const TensorDesc&, const void*, const TensorDesc&, const void*, const void*, const ConvolutionDesc&, const ActivationDesc&, const TensorDesc&, void*, const ExecutionContext&) override;
    void activationForward(const ActivationDesc&, const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void activationBackward(const ActivationDesc&, const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void poolingForward(const PoolingDesc&, const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void poolingBackward(const PoolingDesc&, const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void softmaxForward(const SoftmaxDesc&, const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void softmaxBackward(const SoftmaxDesc&, const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void batchNormForwardTraining(const BatchNormDesc&, const TensorDesc&, const void*, const void*, const void*, void*, void*, void*, void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void batchNormForwardInference(const BatchNormDesc&, const TensorDesc&, const void*, const void*, const void*, const void*, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void batchNormBackward(const BatchNormDesc&, const TensorDesc&, const void*, const TensorDesc&, const void*, const void*, const void*, const void*, const TensorDesc&, void*, void*, void*, const ExecutionContext&) override;
    void layerNormForward(const LayerNormDesc&, const TensorDesc&, const void*, const TensorDesc&, const void*, const void*, const TensorDesc&, void*, void*, void*, const ExecutionContext&) override;
    void layerNormBackward(const LayerNormDesc&, const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, const void*, const void*, const void*, const TensorDesc&, void*, void*, void*, const ExecutionContext&) override;
    void rmsNormForward(const RmsNormDesc&, const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, void*, void*, const ExecutionContext&) override;
    void rmsNormBackward(const RmsNormDesc&, const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, const void*, const void*, const TensorDesc&, void*, void*, const ExecutionContext&) override;
    std::size_t dropoutReserveSpaceSize(const TensorDesc&, const ExecutionContext&) override;
    void dropoutForward(const DropoutDesc&, const TensorDesc&, const void*, const TensorDesc&, void*, void*, std::size_t, const ExecutionContext&) override;
    void dropoutBackward(const DropoutDesc&, const TensorDesc&, const void*, const void*, std::size_t, const TensorDesc&, void*, const ExecutionContext&) override;
    void lrnForward(const LrnDesc&, const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void lrnBackward(const LrnDesc&, const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void tensorAdd(const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void tensorMultiply(const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void tensorMin(const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void tensorMax(const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void tensorReduceSum(const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void tensorReduceProduct(const TensorDesc&, const void*, const TensorDesc&, void*, const ExecutionContext&) override;
    void scaledDotProductAttentionForward(const AttentionDesc&, const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc&, const void*, const TensorDesc*, const void*, const TensorDesc&, void*, const ExecutionContext&) override;

private:
    struct NativeState;
    std::unique_ptr<NativeState> native_;
};

}
