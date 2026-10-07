#include "clap/miopen_backend.hpp"
#include "clap/dnn_dyn_backends.hpp"

#include "clap/dnn_contract_utils.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace clap {
    namespace {
        using namespace abi;
        // miopenStatusNotImplemented / miopenStatusUnsupportedOp mean that
        // MIOpen has no native implementation for the configuration.
        void check(miopenStatus_t s, const char* what) {
            if (s == miopenStatusNotImplemented || s == miopenStatusUnsupportedOp) throw DnnUnsupportedError(std::string("CLAP_DNN MIOpen: no native support in ") + what + " status=" + std::to_string(s));
            if (s != miopenStatusSuccess) throw std::runtime_error(std::string("MIOpen call failed: ") + what + " status=" + std::to_string(s));
        }
        void checkHip(hipError_t s, const char* what) {
            if (s != 0) throw std::runtime_error(std::string("HIP call failed: ") + what + " status=" + std::to_string(s));
        }
        miopenDataType_t miopenType(DataType t) {
            switch (t) {
                case DataType::Float32:  return miopenFloat;
                case DataType::Float16:  return miopenHalf;
                case DataType::BFloat16: return miopenBFloat16;
            }
            return miopenFloat;
        }
        int toInt(std::int64_t v, const char* what) {
            if (v < 0 || v > INT_MAX) throw DnnUnsupportedError(std::string("CLAP_DNN MIOpen ") + what + ": tensor extent or stride does not fit the MIOpen int descriptor");
            return (int) v;
        }
        // MIOpen tensor descriptor with datatype, dims and element strides
        // (miopenSetTensorDescriptor, 1..5 dimensions).
        struct Tensor {
            miopenTensorDescriptor_t v {
            };
            explicit Tensor(const TensorDesc& d) {
                if (d.dims.empty() || d.dims.size() > 5) throw DnnUnsupportedError("CLAP_DNN MIOpen: tensors must have rank 1..5");
                const auto s = d.effectiveStrides();
                std::vector<int> dims, strides;
                for (std::size_t i = 0; i < d.dims.size(); ++i) {
                    dims.push_back(toInt(d.dims[i], "Tensor"));
                    strides.push_back(toInt(s[i], "Tensor"));
                }
                init(miopenType(d.type), dims, strides);
            }
            Tensor(DataType t, const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& strides) {
                std::vector<int> d32, s32;
                for (std::size_t i = 0; i < dims.size(); ++i) {
                    d32.push_back(toInt(dims[i], "Tensor"));
                    s32.push_back(toInt(strides[i], "Tensor"));
                }
                init(miopenType(t), d32, s32);
            }
            ~Tensor() {
                if (v) dnn_dyn::p_miopenDestroyTensorDescriptor(v);
            }
            Tensor(const Tensor&) = delete;
            Tensor& operator=(const Tensor&) = delete;
        private:
            void init(miopenDataType_t type, const std::vector<int>& dims, const std::vector<int>& strides) {
                check(dnn_dyn::p_miopenCreateTensorDescriptor( &v), "miopenCreateTensorDescriptor");
                const miopenStatus_t s = dnn_dyn::p_miopenSetTensorDescriptor(v, type, (int) dims.size(), dims.data(), strides.data());
                if (s != miopenStatusSuccess) {
                    dnn_dyn::p_miopenDestroyTensorDescriptor(v);
                    v = nullptr;
                    check(s, "miopenSetTensorDescriptor");
                }
            }
        };
        std::vector<std::int64_t> packedStrides(const std::vector<std::int64_t>& dims) {
            std::vector<std::int64_t> s(dims.size(), 1);
            std::int64_t running = 1;
            for (std::size_t i = dims.size(); i-- > 0;) {
                s[i] = running;
                running *= dims[i];
            }
            return s;
        }
        struct Conv {
            miopenConvolutionDescriptor_t v {
            };
            Conv(const ConvolutionDesc& c, miopenConvolutionMode_t mode = miopenConvolution) {
                check(dnn_dyn::p_miopenCreateConvolutionDescriptor( &v), "miopenCreateConvolutionDescriptor");
                check(dnn_dyn::p_miopenInitConvolutionDescriptor(v, mode, (int)c.pad_h, (int)c.pad_w, (int)c.stride_h, (int)c.stride_w, (int)c.dilation_h, (int)c.dilation_w), "miopenInitConvolutionDescriptor");
            }
            ~Conv() {
                if (v) dnn_dyn::p_miopenDestroyConvolutionDescriptor(v);
            }
        };
        // MIOpen has no SiLU activation mode; SiLU is composed from the
        // LOGISTIC activation and miopenOpTensor (see activationForward).
        miopenActivationMode_t actMode(ActivationMode m) {
            if (m == ActivationMode::ReLU) return miopenActivationRELU;
            if (m == ActivationMode::Tanh) return miopenActivationTANH;
            return miopenActivationLOGISTIC;
        }
        struct Act {
            miopenActivationDescriptor_t v {
            };
            explicit Act(const ActivationDesc& a) {
                check(dnn_dyn::p_miopenCreateActivationDescriptor( &v), "miopenCreateActivationDescriptor");
                miopenActivationMode_t m = actMode(a.mode);
                check(dnn_dyn::p_miopenSetActivationDescriptor(v, m, a.mode == ActivationMode::SiLU ? 0.0 : a.alpha, 0, 0), "miopenSetActivationDescriptor");
            }
            ~Act() {
                if (v) dnn_dyn::p_miopenDestroyActivationDescriptor(v);
            }
        };
        struct Pool {
            miopenPoolingDescriptor_t v {
            };
            explicit Pool(const PoolingDesc& p) {
                check(dnn_dyn::p_miopenCreatePoolingDescriptor( &v), "miopenCreatePoolingDescriptor");
                check(dnn_dyn::p_miopenSet2dPoolingDescriptor(v, p.mode == PoolingMode::Max ? miopenPoolingMax:miopenPoolingAverage, (int)p.window_h, (int)p.window_w, (int)p.pad_h, (int)p.pad_w, (int)p.stride_h, (int)p.stride_w), "miopenSet2dPoolingDescriptor");
            }
            ~Pool() {
                if (v) dnn_dyn::p_miopenDestroyPoolingDescriptor(v);
            }
        };
        // Device staging buffer used only by the historical host-pointer API.
        struct Buf {
            void* p {
            };
            std::size_t n {
            };
            explicit Buf(std::size_t bytes): n(bytes) {
                if (n) checkHip(dnn_dyn::p_hipMalloc( &p, n), "hipMalloc");
            }
            ~Buf() {
                if (p) dnn_dyn::p_hipFree(p);
            }
            void h2d(const void* src) {
                if (n) checkHip(dnn_dyn::p_hipMemcpy(p, src, n, hipMemcpyHostToDevice), "hipMemcpy H2D");
            }
            void d2h(void* dst) {
                if (n) checkHip(dnn_dyn::p_hipMemcpy(dst, p, n, hipMemcpyDeviceToHost), "hipMemcpy D2H");
            }
        };
        // Stream-ordered scratch memory for one framework call: carved from
        // ExecutionContext::workspace when possible, otherwise allocated with
        // hipMallocAsync / released with hipFreeAsync on the caller stream.
        struct Scratch {
            hipStream_t stream {
            };
            char* base {
            };
            std::size_t capacity {
            };
            std::size_t offset {
            };
            std::vector<void*> owned;
            explicit Scratch(const ExecutionContext& ctx): stream(ctx.stream), base(static_cast<char*>(ctx.workspace)), capacity(ctx.workspace ? ctx.workspace_bytes : 0) {
            }
            ~Scratch() {
                for (void* p : owned) dnn_dyn::p_hipFreeAsync(p, stream);
            }
            Scratch(const Scratch&) = delete;
            Scratch& operator=(const Scratch&) = delete;
            void* get(std::size_t bytes) {
                if (!bytes) return nullptr;
                const std::size_t aligned = (offset + 255) & ~static_cast<std::size_t>(255);
                if (base && aligned + bytes <= capacity) {
                    offset = aligned + bytes;
                    return base + aligned;
                }
                void* p = nullptr;
                checkHip(dnn_dyn::p_hipMallocAsync( &p, bytes, stream), "hipMallocAsync");
                owned.push_back(p);
                return p;
            }
        };
        std::size_t bytes(const TensorDesc& d) {
            if (d.type != DataType::Float32) throw std::runtime_error("CLAP_DNN MIOpen: the float* host API requires Float32 descriptors; use the ExecutionContext overloads for Float16/BFloat16");
            return d.bytes();
        }
        TensorDesc channelDesc(const TensorDesc& d) {
            return TensorDesc( {
                1, d.dims.at(1), 1, 1
            });
        }
        ExecutionContext legacyContext() {
            return ExecutionContext::rocm(nullptr);
        }
        void syncLegacy() {
            checkHip(dnn_dyn::p_hipStreamSynchronize(nullptr), "hipStreamSynchronize");
        }
        void requireRocm(const ExecutionContext& ctx, const char* operation) {
            dnn_contract::requireBackend(ctx, ExecutionBackend::ROCM, "MIOpen", operation);
        }
        // MIOpen handle of the current HIP device bound to the caller
        // stream; the backend mutex serializes binding and enqueue.
        struct Handle {
            std::lock_guard<std::recursive_mutex> lock;
            miopenHandle_t v {
            };
            Handle(std::recursive_mutex& m, std::map<int, miopenHandle_t>& handles, const ExecutionContext& ctx): lock(m) {
                int device = 0;
                checkHip(dnn_dyn::p_hipGetDevice( &device), "hipGetDevice");
                auto it = handles.find(device);
                if (it == handles.end()) {
                    miopenHandle_t h = nullptr;
                    check(dnn_dyn::p_miopenCreate( &h), "miopenCreate");
                    it = handles.emplace(device, h).first;
                }
                v = it->second;
                check(dnn_dyn::p_miopenSetStream(v, ctx.stream), "miopenSetStream");
            }
        };
    }

    namespace miopen_detail {
        // Cached MIOpen MHA solution (Find 2.0) for one problem shape.
        struct MhaSolution {
            miopenSolution_t solution = nullptr;
            std::size_t workspace = 0;
            ~MhaSolution() {
                if (solution && dnn_dyn::p_miopenDestroySolution) dnn_dyn::p_miopenDestroySolution(solution);
            }
        };
    }

    struct MiOpenBackend::NativeState {
        std::recursive_mutex mutex;
        std::map<int, miopenHandle_t> handles;
        std::map<std::string, std::unique_ptr<miopen_detail::MhaSolution>> mha;
        // Per-device constant MHA scalars (descale/scale factors, dropout
        // probability/seed/offset) initialized once.
        std::map<int, void*> mha_constants;

        ~NativeState() {
            mha.clear();
            for (auto& c : mha_constants)
                if (c.second && dnn_dyn::p_hipFree) dnn_dyn::p_hipFree(c.second);
            for (auto& h : handles)
                if (h.second && dnn_dyn::p_miopenDestroy) dnn_dyn::p_miopenDestroy(h.second);
        }
    };

    MiOpenBackend::MiOpenBackend() {
        if (!dnn_dyn::loadHipAndMiopen()) throw std::runtime_error(std::string("MIOpen/AMD runtime load failed: ") + dnn_dyn::lastError());
        native_.reset(new NativeState());
    }
    MiOpenBackend::~MiOpenBackend() = default;

    const char* MiOpenBackend::name() const noexcept {
        return "MIOpen/AMD";
    }

    ExecutionBackend MiOpenBackend::executionBackend() const noexcept {
        return ExecutionBackend::ROCM;
    }

    // ------------------------------------------------------------------
    // Historical host-pointer API: stage through device buffers, run the
    // framework-contract implementation on the default stream, copy back.
    // ------------------------------------------------------------------

    void MiOpenBackend::convolutionForward(const TensorDesc& xd, const float* x, const TensorDesc& wd, const float* w, const ConvolutionDesc& c, const TensorDesc& yd, float* y) {
        std::cerr<<"[MY CLAP] ROCm convolution called through MIOpen function pointers\n";
        Buf X(bytes(xd)), W(bytes(wd)), Y(bytes(yd));
        X.h2d(x);
        W.h2d(w);
        convolutionForward(xd, X.p, wd, W.p, c, yd, Y.p, legacyContext());
        syncLegacy();
        Y.d2h(y);
    }

    void MiOpenBackend::convolutionBackwardData(const TensorDesc& dyd, const float* dy, const TensorDesc& wd, const float* w, const ConvolutionDesc& c, const TensorDesc& dxd, float* dx) {
        Buf DY(bytes(dyd)), W(bytes(wd)), DX(bytes(dxd));
        DY.h2d(dy);
        W.h2d(w);
        convolutionBackwardData(dyd, DY.p, wd, W.p, c, dxd, DX.p, legacyContext());
        syncLegacy();
        DX.d2h(dx);
    }

    void MiOpenBackend::convolutionBackwardWeights(const TensorDesc& xd, const float* x, const TensorDesc& dyd, const float* dy, const ConvolutionDesc& c, const TensorDesc& dwd, float* dw) {
        Buf X(bytes(xd)), DY(bytes(dyd)), DW(bytes(dwd));
        X.h2d(x);
        DY.h2d(dy);
        convolutionBackwardWeights(xd, X.p, dyd, DY.p, c, dwd, DW.p, legacyContext());
        syncLegacy();
        DW.d2h(dw);
    }

    void MiOpenBackend::activationForward(const ActivationDesc& a, const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y) {
        Buf X(bytes(xd)), Y(bytes(yd));
        X.h2d(x);
        activationForward(a, xd, X.p, yd, Y.p, legacyContext());
        syncLegacy();
        Y.d2h(y);
    }

    void MiOpenBackend::activationBackward(const ActivationDesc& a, const TensorDesc& xd, const float* x, const TensorDesc& dyd, const float* dy, const TensorDesc& dxd, float* dx) {
        Buf X(bytes(xd)), DY(bytes(dyd)), DX(bytes(dxd));
        X.h2d(x);
        DY.h2d(dy);
        activationBackward(a, xd, X.p, dyd, DY.p, dxd, DX.p, legacyContext());
        syncLegacy();
        DX.d2h(dx);
    }

    void MiOpenBackend::poolingForward(const PoolingDesc& p, const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y) {
        Buf X(bytes(xd)), Y(bytes(yd));
        X.h2d(x);
        poolingForward(p, xd, X.p, yd, Y.p, legacyContext());
        syncLegacy();
        Y.d2h(y);
    }

    void MiOpenBackend::poolingBackward(const PoolingDesc& p, const TensorDesc& xd, const float* x, const TensorDesc& yd, const float* y, const TensorDesc& dyd, const float* dy, const TensorDesc& dxd, float* dx) {
        Buf X(bytes(xd)), Y(bytes(yd)), DY(bytes(dyd)), DX(bytes(dxd));
        X.h2d(x);
        Y.h2d(y);
        DY.h2d(dy);
        poolingBackward(p, xd, X.p, yd, Y.p, dyd, DY.p, dxd, DX.p, legacyContext());
        syncLegacy();
        DX.d2h(dx);
    }

    // Historical softmax: normalization over dimension 1 (channels).
    void MiOpenBackend::softmaxForward(const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y) {
        Buf X(bytes(xd)), Y(bytes(yd));
        X.h2d(x);
        SoftmaxDesc softmax;
        softmax.axis = 1;
        softmaxForward(softmax, xd, X.p, yd, Y.p, legacyContext());
        syncLegacy();
        Y.d2h(y);
    }

    void MiOpenBackend::softmaxBackward(const TensorDesc& yd, const float* y, const TensorDesc& dyd, const float* dy, const TensorDesc& dxd, float* dx) {
        Buf Y(bytes(yd)), DY(bytes(dyd)), DX(bytes(dxd));
        Y.h2d(y);
        DY.h2d(dy);
        SoftmaxDesc softmax;
        softmax.axis = 1;
        softmaxBackward(softmax, yd, Y.p, dyd, DY.p, dxd, DX.p, legacyContext());
        syncLegacy();
        DX.d2h(dx);
    }

    void MiOpenBackend::batchNormForwardTraining(const BatchNormDesc& bn, const TensorDesc& xd, const float* x, const float* scale, const float* bias, float* rm, float* rv, float* sm, float* sv, const TensorDesc& yd, float* y) {
        TensorDesc cd = channelDesc(xd);
        Buf X(bytes(xd)), Y(bytes(yd)), S(bytes(cd)), B(bytes(cd)), RM(bytes(cd)), RV(bytes(cd)), SM(bytes(cd)), SV(bytes(cd));
        X.h2d(x);
        S.h2d(scale);
        B.h2d(bias);
        if (rm) RM.h2d(rm);
        if (rv) RV.h2d(rv);
        batchNormForwardTraining(bn, xd, X.p, S.p, B.p, RM.p, RV.p, SM.p, SV.p, yd, Y.p, legacyContext());
        syncLegacy();
        Y.d2h(y);
        if (rm) RM.d2h(rm);
        if (rv) RV.d2h(rv);
        SM.d2h(sm);
        SV.d2h(sv);
    }

    void MiOpenBackend::batchNormForwardInference(const BatchNormDesc& bn, const TensorDesc& xd, const float* x, const float* scale, const float* bias, const float* mean, const float* var, const TensorDesc& yd, float* y) {
        TensorDesc cd = channelDesc(xd);
        Buf X(bytes(xd)), Y(bytes(yd)), S(bytes(cd)), B(bytes(cd)), M(bytes(cd)), V(bytes(cd));
        X.h2d(x);
        S.h2d(scale);
        B.h2d(bias);
        M.h2d(mean);
        V.h2d(var);
        batchNormForwardInference(bn, xd, X.p, S.p, B.p, M.p, V.p, yd, Y.p, legacyContext());
        syncLegacy();
        Y.d2h(y);
    }

    void MiOpenBackend::batchNormBackward(const BatchNormDesc& bn, const TensorDesc& xd, const float* x, const TensorDesc& dyd, const float* dy, const float* scale, const float* mean, const float* var, const TensorDesc& dxd, float* dx, float* ds, float* db) {
        TensorDesc cd = channelDesc(xd);
        Buf X(bytes(xd)), DY(bytes(dyd)), DX(bytes(dxd)), S(bytes(cd)), M(bytes(cd)), V(bytes(cd)), DS(bytes(cd)), DB(bytes(cd));
        X.h2d(x);
        DY.h2d(dy);
        S.h2d(scale);
        M.h2d(mean);
        V.h2d(var);
        batchNormBackward(bn, xd, X.p, dyd, DY.p, S.p, M.p, V.p, dxd, DX.p, DS.p, DB.p, legacyContext());
        syncLegacy();
        DX.d2h(dx);
        DS.d2h(ds);
        DB.d2h(db);
    }

    void MiOpenBackend::convolutionBackwardBias(const TensorDesc& dyd, const float* dy, const TensorDesc& dbd, float* db) {
        Buf DY(bytes(dyd)), DB(bytes(dbd));
        DY.h2d(dy);
        convolutionBackwardBias(dyd, DY.p, dbd, DB.p, legacyContext());
        syncLegacy();
        DB.d2h(db);
    }

    void MiOpenBackend::convolutionTransposeForward(const TensorDesc& xd, const float* x, const TensorDesc& wd, const float* w, const ConvolutionDesc& c, const TensorDesc& yd, float* y) {
        Buf X(bytes(xd)), W(bytes(wd)), Y(bytes(yd));
        X.h2d(x);
        W.h2d(w);
        convolutionTransposeForward(xd, X.p, wd, W.p, c, yd, Y.p, legacyContext());
        syncLegacy();
        Y.d2h(y);
    }

    void MiOpenBackend::fusedConvolutionBiasActivation(const TensorDesc& xd, const float* x, const TensorDesc& wd, const float* w, const float* bias, const ConvolutionDesc& c, const ActivationDesc& a, const TensorDesc& yd, float* y) {
        TensorDesc bd = channelDesc(yd);
        Buf X(bytes(xd)), W(bytes(wd)), Y(bytes(yd)), B(bytes(bd));
        X.h2d(x);
        W.h2d(w);
        B.h2d(bias);
        fusedConvolutionBiasActivation(xd, X.p, wd, W.p, B.p, c, a, yd, Y.p, legacyContext());
        syncLegacy();
        Y.d2h(y);
    }

    void MiOpenBackend::layerNormForward(const LayerNormDesc& ln, const TensorDesc& xd, const float* x, const float* scale, const float* bias, const TensorDesc& yd, float* y, float* mean, float* rstd) {
        TensorDesc sd( {
            (std::int64_t) xd.dims.back()
        });
        const std::size_t rows = xd.elements() / xd.dims.back();
        Buf X(bytes(xd)), Y(bytes(yd)), S(bytes(sd)), B(bytes(sd)), M(rows * sizeof(float)), R(rows * sizeof(float));
        X.h2d(x);
        S.h2d(scale);
        B.h2d(bias);
        layerNormForward(ln, xd, X.p, sd, S.p, B.p, yd, Y.p, M.p, R.p, legacyContext());
        syncLegacy();
        Y.d2h(y);
        if (mean) M.d2h(mean);
        if (rstd) R.d2h(rstd);
    }

    void MiOpenBackend::layerNormBackward(const LayerNormDesc& ln, const TensorDesc& xd, const float* x, const TensorDesc& dyd, const float* dy, const float* scale, const float* mean, const float* rstd, const TensorDesc& dxd, float* dx, float* dscale, float* dbias) {
        TensorDesc sd( {
            (std::int64_t) xd.dims.back()
        });
        const std::size_t rows = xd.elements() / xd.dims.back();
        Buf X(bytes(xd)), DY(bytes(dyd)), S(bytes(sd)), M(rows * sizeof(float)), R(rows * sizeof(float)), DX(bytes(dxd)), DS(bytes(sd)), DB(bytes(sd));
        X.h2d(x);
        DY.h2d(dy);
        S.h2d(scale);
        M.h2d(mean);
        R.h2d(rstd);
        layerNormBackward(ln, xd, X.p, dyd, DY.p, sd, S.p, M.p, R.p, dxd, DX.p, DS.p, DB.p, legacyContext());
        syncLegacy();
        DX.d2h(dx);
        DS.d2h(dscale);
        DB.d2h(dbias);
    }

    // Historical dropout API: the MIOpen reserve space doubles as the mask.
    void MiOpenBackend::dropoutForward(const DropoutDesc& dropout, const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y, std::uint8_t* mask) {
        const std::size_t rs = dropoutReserveSpaceSize(xd, legacyContext());
        Buf reserve(rs), X(bytes(xd)), Y(bytes(yd));
        X.h2d(x);
        dropoutForward(dropout, xd, X.p, yd, Y.p, reserve.p, rs, legacyContext());
        syncLegacy();
        Y.d2h(y);
        if (mask) {
            std::vector<std::uint8_t> tmp(rs);
            reserve.d2h(tmp.data());
            std::copy_n(tmp.data(), std::min < std::size_t > (rs, xd.elements()), mask);
        }
    }

    void MiOpenBackend::dropoutBackward(const DropoutDesc& dropout, const TensorDesc& dyd, const float* dy, const std::uint8_t* mask, const TensorDesc& dxd, float* dx) {
        const std::size_t rs = dropoutReserveSpaceSize(dyd, legacyContext());
        Buf reserve(rs), DY(bytes(dyd)), DX(bytes(dxd));
        DY.h2d(dy);
        if (mask) {
            std::vector<std::uint8_t> tmp(rs, 0);
            std::copy_n(mask, std::min < std::size_t > (rs, dyd.elements()), tmp.data());
            reserve.h2d(tmp.data());
        }
        dropoutBackward(dropout, dyd, DY.p, reserve.p, rs, dxd, DX.p, legacyContext());
        syncLegacy();
        DX.d2h(dx);
    }
void MiOpenBackend::lrnForward(const LrnDesc& lrn, const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y)
{
    Buf X(bytes(xd)), Y(bytes(yd));
    X.h2d(x);
    lrnForward(lrn, xd, X.p, yd, Y.p, legacyContext());
    syncLegacy();
    Y.d2h(y);
}

void MiOpenBackend::lrnBackward(const LrnDesc& lrn, const TensorDesc& xd, const float* x, const TensorDesc& yd, const float* y, const TensorDesc& dyd, const float* dy, const TensorDesc& dxd, float* dx)
{
    Buf X(bytes(xd)), Y(bytes(yd)), DY(bytes(dyd)), DX(bytes(dxd));
    X.h2d(x);
    Y.h2d(y);
    DY.h2d(dy);
    lrnBackward(lrn, xd, X.p, yd, Y.p, dyd, DY.p, dxd, DX.p, legacyContext());
    syncLegacy();
    DX.d2h(dx);
}

void MiOpenBackend::tensorAdd(const TensorDesc& ad, const float* a, const TensorDesc& bd, const float* b, const TensorDesc& yd, float* y)
{
    Buf A0(bytes(ad)), B0(bytes(bd)), C0(bytes(yd));
    A0.h2d(a);
    B0.h2d(b);
    tensorAdd(ad, A0.p, bd, B0.p, yd, C0.p, legacyContext());
    syncLegacy();
    C0.d2h(y);
}

void MiOpenBackend::tensorMultiply(const TensorDesc& ad, const float* a, const TensorDesc& bd, const float* b, const TensorDesc& yd, float* y)
{
    Buf A0(bytes(ad)), B0(bytes(bd)), C0(bytes(yd));
    A0.h2d(a);
    B0.h2d(b);
    tensorMultiply(ad, A0.p, bd, B0.p, yd, C0.p, legacyContext());
    syncLegacy();
    C0.d2h(y);
}

void MiOpenBackend::tensorMin(const TensorDesc& ad, const float* a, const TensorDesc& bd, const float* b, const TensorDesc& yd, float* y)
{
    Buf A0(bytes(ad)), B0(bytes(bd)), C0(bytes(yd));
    A0.h2d(a);
    B0.h2d(b);
    tensorMin(ad, A0.p, bd, B0.p, yd, C0.p, legacyContext());
    syncLegacy();
    C0.d2h(y);
}

void MiOpenBackend::tensorMax(const TensorDesc& ad, const float* a, const TensorDesc& bd, const float* b, const TensorDesc& yd, float* y)
{
    Buf A0(bytes(ad)), B0(bytes(bd)), C0(bytes(yd));
    A0.h2d(a);
    B0.h2d(b);
    tensorMax(ad, A0.p, bd, B0.p, yd, C0.p, legacyContext());
    syncLegacy();
    C0.d2h(y);
}

void MiOpenBackend::tensorReduceSum(const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y)
{
    Buf X0(bytes(xd)), Y0(bytes(yd));
    X0.h2d(x);
    tensorReduceSum(xd, X0.p, yd, Y0.p, legacyContext());
    syncLegacy();
    Y0.d2h(y);
}

void MiOpenBackend::tensorReduceProduct(const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y)
{
    Buf X0(bytes(xd)), Y0(bytes(yd));
    X0.h2d(x);
    tensorReduceProduct(xd, X0.p, yd, Y0.p, legacyContext());
    syncLegacy();
    Y0.d2h(y);
}

// ==========================================================================
// Framework execution contract
//
// Every tensor pointer is a HIP device pointer owned by the caller.  The
// MIOpen handle of the current device is bound to ctx.stream with
// miopenSetStream and the MIOpen primitive is enqueued on that stream.  No
// host staging, no duplicate input/output buffers, no device synchronization.
// ==========================================================================

void MiOpenBackend::convolutionForward(const TensorDesc& xd, const void* x, const TensorDesc& wd, const void* w, const ConvolutionDesc& c, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "convolutionForward");
    xd.require4d("convolutionForward");
    Handle h(native_->mutex, native_->handles, ctx);
    Tensor Xd(xd), Wd(wd), Yd(yd);
    Conv cd(c);
    float a = 1, b = 0;
    dnn_contract::trace("ROCM", "convolutionForward", "miopenConvolutionForward");
    check(
        dnn_dyn::p_miopenConvolutionForward(
            h.v, &a, Xd.v, x, Wd.v, w, cd.v, miopenConvolutionFwdAlgoGEMM,
            &b, Yd.v, y, nullptr, 0),
        "miopenConvolutionForward");
}

void MiOpenBackend::convolutionBackwardData(const TensorDesc& dyd, const void* dy, const TensorDesc& wd, const void* w, const ConvolutionDesc& c, const TensorDesc& dxd, void* dx, const ExecutionContext& ctx)
{
    requireRocm(ctx, "convolutionBackwardData");
    dyd.require4d("convolutionBackwardData");
    Handle h(native_->mutex, native_->handles, ctx);
    Tensor DYd(dyd), Wd(wd), DXd(dxd);
    Conv cd(c);
    float a = 1, b = 0;
    dnn_contract::trace("ROCM", "convolutionBackwardData", "miopenConvolutionBackwardData");
    check(
        dnn_dyn::p_miopenConvolutionBackwardData(
            h.v, &a, DYd.v, dy, Wd.v, w, cd.v, miopenConvolutionBwdDataAlgoGEMM,
            &b, DXd.v, dx, nullptr, 0),
        "miopenConvolutionBackwardData");
}

void MiOpenBackend::convolutionBackwardWeights(const TensorDesc& xd, const void* x, const TensorDesc& dyd, const void* dy, const ConvolutionDesc& c, const TensorDesc& dwd, void* dw, const ExecutionContext& ctx)
{
    requireRocm(ctx, "convolutionBackwardWeights");
    xd.require4d("convolutionBackwardWeights");
    Handle h(native_->mutex, native_->handles, ctx);
    Tensor Xd(xd), DYd(dyd), DWd(dwd);
    Conv cd(c);
    float a = 1, b = 0;
    dnn_contract::trace("ROCM", "convolutionBackwardWeights", "miopenConvolutionBackwardWeights");
    check(
        dnn_dyn::p_miopenConvolutionBackwardWeights(
            h.v, &a, DYd.v, dy, Xd.v, x, cd.v, miopenConvolutionBwdWeightsAlgoGEMM,
            &b, DWd.v, dw, nullptr, 0),
        "miopenConvolutionBackwardWeights");
}

// db may be described as [C] or [1, C, 1, 1]; MIOpen uses the latter.
void MiOpenBackend::convolutionBackwardBias(const TensorDesc& dyd, const void* dy, const TensorDesc& dbd, void* db, const ExecutionContext& ctx)
{
    requireRocm(ctx, "convolutionBackwardBias");
    dyd.require4d("convolutionBackwardBias");
    if (dbd.elements() != static_cast<std::size_t>(dyd.dims[1]))
        throw std::runtime_error("convolutionBackwardBias: db must hold one value per channel");
    Handle h(native_->mutex, native_->handles, ctx);
    const std::vector<std::int64_t> bdims = {1, dyd.dims[1], 1, 1};
    Tensor DYd(dyd), DBd(dbd.type, bdims, packedStrides(bdims));
    float a = 1, b = 0;
    dnn_contract::trace("ROCM", "convolutionBackwardBias", "miopenConvolutionBackwardBias");
    check(
        dnn_dyn::p_miopenConvolutionBackwardBias(
            h.v, &a, DYd.v, dy, &b, DBd.v, db),
        "miopenConvolutionBackwardBias");
}

void MiOpenBackend::convolutionTransposeForward(const TensorDesc& xd, const void* x, const TensorDesc& wd, const void* w, const ConvolutionDesc& c, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "convolutionTransposeForward");
    xd.require4d("convolutionTransposeForward");
    Handle h(native_->mutex, native_->handles, ctx);
    Tensor Xd(xd), Wd(wd), Yd(yd);
    Conv cd(c, miopenTranspose);
    float a = 1, b = 0;
    dnn_contract::trace("ROCM", "convolutionTransposeForward", "miopenConvolutionForward(miopenTranspose)");
    check(
        dnn_dyn::p_miopenConvolutionForward(
            h.v, &a, Xd.v, x, Wd.v, w, cd.v, miopenConvolutionFwdAlgoGEMM,
            &b, Yd.v, y, nullptr, 0),
        "miopenConvolutionForward(transpose)");
}

void MiOpenBackend::fusedConvolutionBiasActivation(const TensorDesc& xd, const void* x, const TensorDesc& wd, const void* w, const void* bias, const ConvolutionDesc& c, const ActivationDesc& a, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "fusedConvolutionBiasActivation");
    xd.require4d("fusedConvolutionBiasActivation");
    if (a.mode == ActivationMode::SiLU)
        dnn_contract::unsupported("MIOpen", "fusedConvolutionBiasActivation: MIOpen has no SiLU activation mode");
    Handle h(native_->mutex, native_->handles, ctx);
    const std::vector<std::int64_t> bdims = {1, yd.dims.at(1), 1, 1};
    Tensor Xd(xd), Wd(wd), Yd(yd), Bd(yd.type, bdims, packedStrides(bdims));
    Conv cd(c);
    Act ad(a);
    float one = 1, zero = 0;
    dnn_contract::trace("ROCM", "fusedConvolutionBiasActivation", "miopenConvolutionForward + miopenConvolutionForwardBias + miopenActivationForward");
    check(
        dnn_dyn::p_miopenConvolutionForward(
            h.v, &one, Xd.v, x, Wd.v, w, cd.v, miopenConvolutionFwdAlgoGEMM,
            &zero, Yd.v, y, nullptr, 0),
        "miopenConvolutionForward(fused)");
    check(
        dnn_dyn::p_miopenConvolutionForwardBias(
            h.v, &one, Bd.v, bias, &one, Yd.v, y),
        "miopenConvolutionForwardBias");
    check(
        dnn_dyn::p_miopenActivationForward(
            h.v, ad.v, &one, Yd.v, y, &zero, Yd.v, y),
        "miopenActivationForward(fused)");
}

// SiLU has no MIOpen activation mode.  It is composed from native MIOpen
// primitives on the caller stream:
//   s = LOGISTIC(x)                        miopenActivationForward
//   y = s * x                              miopenOpTensor(Mul)
void MiOpenBackend::activationForward(const ActivationDesc& a, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "activationForward");
    dnn_contract::requireSameShape(xd, yd, "activationForward");
    Handle h(native_->mutex, native_->handles, ctx);
    Tensor Xd(xd), Yd(yd);
    Act ad(a);
    float one = 1, zero = 0;

    if (a.mode != ActivationMode::SiLU) {
        dnn_contract::trace("ROCM", "activationForward", "miopenActivationForward");
        check(
            dnn_dyn::p_miopenActivationForward(
                h.v, ad.v, &one, Xd.v, x, &zero, Yd.v, y),
            "miopenActivationForward");
        return;
    }

    Scratch scratch(ctx);
    void* s = scratch.get(yd.bytes());

    dnn_contract::trace("ROCM", "activationForward(SiLU)", "miopenActivationForward(LOGISTIC) + miopenOpTensor(Mul)");
    check(
        dnn_dyn::p_miopenActivationForward(
            h.v, ad.v, &one, Xd.v, x, &zero, Yd.v, s),
        "miopenActivationForward(SiLU sigmoid)");
    check(
        dnn_dyn::p_miopenOpTensor(
            h.v, miopenTensorOpMul, &one, Yd.v, s, &one, Xd.v, x, &zero, Yd.v, y),
        "miopenOpTensor(SiLU multiply)");
}

// MIOpen activation backward consumes y; it is recomputed on the caller
// stream.  SiLU backward, dx = dy * (s + x * s * (1 - s)), is composed from
// native MIOpen primitives:
//   s  = LOGISTIC(x)                       miopenActivationForward
//   t  = x * dy                            miopenOpTensor(Mul)
//   dx = t * s * (1 - s)                   miopenActivationBackward(LOGISTIC)
//   dx = s * dy + dx                       miopenOpTensor(Mul, beta = 1)
void MiOpenBackend::activationBackward(const ActivationDesc& a, const TensorDesc& xd, const void* x, const TensorDesc& dyd, const void* dy, const TensorDesc& dxd, void* dx, const ExecutionContext& ctx)
{
    requireRocm(ctx, "activationBackward");
    dnn_contract::requireSameShape(xd, dyd, "activationBackward");
    dnn_contract::requireSameShape(xd, dxd, "activationBackward");
    Handle h(native_->mutex, native_->handles, ctx);
    Scratch scratch(ctx);
    Tensor Xd(xd), DYd(dyd), DXd(dxd);
    Act ad(a);
    float one = 1, zero = 0;

    void* s = scratch.get(xd.bytes());
    check(
        dnn_dyn::p_miopenActivationForward(
            h.v, ad.v, &one, Xd.v, x, &zero, Xd.v, s),
        "miopenActivationForward(for backward)");

    if (a.mode != ActivationMode::SiLU) {
        dnn_contract::trace("ROCM", "activationBackward", "miopenActivationBackward");
        check(
            dnn_dyn::p_miopenActivationBackward(
                h.v, ad.v, &one, Xd.v, s, DYd.v, dy, Xd.v, x, &zero, DXd.v, dx),
            "miopenActivationBackward");
        return;
    }

    void* t = scratch.get(xd.bytes());

    dnn_contract::trace("ROCM", "activationBackward(SiLU)", "miopenActivationForward/Backward(LOGISTIC) + miopenOpTensor(Mul)");
    check(
        dnn_dyn::p_miopenOpTensor(
            h.v, miopenTensorOpMul, &one, Xd.v, x, &one, DYd.v, dy, &zero, Xd.v, t),
        "miopenOpTensor(SiLU x * dy)");
    check(
        dnn_dyn::p_miopenActivationBackward(
            h.v, ad.v, &one, Xd.v, s, Xd.v, t, Xd.v, x, &zero, DXd.v, dx),
        "miopenActivationBackward(SiLU sigmoid')");
    check(
        dnn_dyn::p_miopenOpTensor(
            h.v, miopenTensorOpMul, &one, Xd.v, s, &one, DYd.v, dy, &one, DXd.v, dx),
        "miopenOpTensor(SiLU accumulate s * dy)");
}

void MiOpenBackend::poolingForward(const PoolingDesc& p, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "poolingForward");
    xd.require4d("poolingForward");
    Handle h(native_->mutex, native_->handles, ctx);
    Tensor Xd(xd), Yd(yd);
    Pool pd(p);
    float al = 1, be = 0;
    dnn_contract::trace("ROCM", "poolingForward", "miopenPoolingForward");
    check(
        dnn_dyn::p_miopenPoolingForward(
            h.v, pd.v, &al, Xd.v, x, &be, Yd.v, y, false, nullptr, 0),
        "miopenPoolingForward");
}

void MiOpenBackend::poolingBackward(const PoolingDesc& p, const TensorDesc& xd, const void* x, const TensorDesc& yd, const void* y, const TensorDesc& dyd, const void* dy, const TensorDesc& dxd, void* dx, const ExecutionContext& ctx)
{
    requireRocm(ctx, "poolingBackward");
    xd.require4d("poolingBackward");
    Handle h(native_->mutex, native_->handles, ctx);
    Tensor Xd(xd), Yd(yd), DYd(dyd), DXd(dxd);
    Pool pd(p);
    float al = 1, be = 0;
    dnn_contract::trace("ROCM", "poolingBackward", "miopenPoolingBackward");
    check(
        dnn_dyn::p_miopenPoolingBackward(
            h.v, pd.v, &al, Yd.v, y, DYd.v, dy, Xd.v, x, &be, DXd.v, dx, nullptr),
        "miopenPoolingBackward");
}

namespace {

// MIOpen softmax normalizes over C of an NCHW tensor (MIOPEN_SOFTMAX_MODE_CHANNEL).
// Any axis of an arbitrary-rank tensor maps exactly onto [outer, axis, inner, 1].
struct SoftmaxView {
    dnn_contract::AxisView v;
    std::unique_ptr<Tensor> t;
    SoftmaxView(const TensorDesc& d, int axis, const char* where)
    {
        if (!dnn_contract::makeAxisView(d, axis, v))
            throw DnnUnsupportedError(std::string("CLAP_DNN MIOpen ") + where +
                                      ": the dimensions around the softmax axis cannot be collapsed into an MIOpen [N, C, H, 1] view");
        t.reset(new Tensor(d.type, {v.outer, v.axis, v.inner, 1}, {v.outer_stride, v.axis_stride, v.inner_stride, 1}));
    }
};

void requireSameView(const SoftmaxView& a, const SoftmaxView& b, const char* where)
{
    if (a.v.outer != b.v.outer || a.v.axis != b.v.axis || a.v.inner != b.v.inner)
        throw std::runtime_error(std::string(where) + ": tensor shapes do not match");
}

} // namespace

void MiOpenBackend::softmaxForward(const SoftmaxDesc& softmax, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "softmaxForward");
    SoftmaxView xv(xd, xd.normalizeAxis(softmax.axis, "softmaxForward"), "softmaxForward");
    SoftmaxView yv(yd, yd.normalizeAxis(softmax.axis, "softmaxForward"), "softmaxForward");
    requireSameView(xv, yv, "softmaxForward");
    Handle h(native_->mutex, native_->handles, ctx);
    float a = 1, b = 0;
    dnn_contract::trace("ROCM", softmax.log_softmax ? "logSoftmaxForward" : "softmaxForward", "miopenSoftmaxForward_V2");
    check(
        dnn_dyn::p_miopenSoftmaxForward_V2(
            h.v, &a, xv.t->v, x, &b, yv.t->v, y,
            softmax.log_softmax ? MIOPEN_SOFTMAX_LOG : MIOPEN_SOFTMAX_ACCURATE,
            MIOPEN_SOFTMAX_MODE_CHANNEL),
        "miopenSoftmaxForward_V2");
}

void MiOpenBackend::softmaxBackward(const SoftmaxDesc& softmax, const TensorDesc& yd, const void* y, const TensorDesc& dyd, const void* dy, const TensorDesc& dxd, void* dx, const ExecutionContext& ctx)
{
    requireRocm(ctx, "softmaxBackward");
    SoftmaxView yv(yd, yd.normalizeAxis(softmax.axis, "softmaxBackward"), "softmaxBackward");
    SoftmaxView dyv(dyd, dyd.normalizeAxis(softmax.axis, "softmaxBackward"), "softmaxBackward");
    SoftmaxView dxv(dxd, dxd.normalizeAxis(softmax.axis, "softmaxBackward"), "softmaxBackward");
    requireSameView(yv, dyv, "softmaxBackward");
    requireSameView(yv, dxv, "softmaxBackward");
    Handle h(native_->mutex, native_->handles, ctx);
    float a = 1, b = 0;
    dnn_contract::trace("ROCM", softmax.log_softmax ? "logSoftmaxBackward" : "softmaxBackward", "miopenSoftmaxBackward_V2");
    check(
        dnn_dyn::p_miopenSoftmaxBackward_V2(
            h.v, &a, yv.t->v, y, dyv.t->v, dy, &b, dxv.t->v, dx,
            softmax.log_softmax ? MIOPEN_SOFTMAX_LOG : MIOPEN_SOFTMAX_ACCURATE,
            MIOPEN_SOFTMAX_MODE_CHANNEL),
        "miopenSoftmaxBackward_V2");
}

// MIOpen batch normalization; statistics are Float32 [C] device vectors and
// saved_variance holds MIOpen's saved inverse variance.
void MiOpenBackend::batchNormForwardTraining(const BatchNormDesc& bn, const TensorDesc& xd, const void* x, const void* scale, const void* bias, void* rm, void* rv, void* sm, void* sv, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "batchNormForwardTraining");
    xd.require4d("batchNormForwardTraining");
    Handle h(native_->mutex, native_->handles, ctx);
    TensorDesc cd = channelDesc(xd);
    Tensor Xd(xd), Yd(yd), Bd(cd);
    float a = 1, b = 0;
    dnn_contract::trace("ROCM", "batchNormForwardTraining", "miopenBatchNormalizationForwardTraining");
    check(
        dnn_dyn::p_miopenBatchNormalizationForwardTraining(
            h.v, miopenBNSpatial, &a, &b, Xd.v, x, Yd.v, y, Bd.v,
            const_cast<void*>(scale), const_cast<void*>(bias), 1.0, rm, rv, bn.epsilon, sm, sv),
        "miopenBatchNormalizationForwardTraining");
}

void MiOpenBackend::batchNormForwardInference(const BatchNormDesc& bn, const TensorDesc& xd, const void* x, const void* scale, const void* bias, const void* mean, const void* var, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "batchNormForwardInference");
    xd.require4d("batchNormForwardInference");
    Handle h(native_->mutex, native_->handles, ctx);
    TensorDesc cd = channelDesc(xd);
    Tensor Xd(xd), Yd(yd), Bd(cd);
    float a = 1, b = 0;
    dnn_contract::trace("ROCM", "batchNormForwardInference", "miopenBatchNormalizationForwardInference");
    check(
        dnn_dyn::p_miopenBatchNormalizationForwardInference(
            h.v, miopenBNSpatial, &a, &b, Xd.v, x, Yd.v, y, Bd.v,
            const_cast<void*>(scale), const_cast<void*>(bias), const_cast<void*>(mean), const_cast<void*>(var), bn.epsilon),
        "miopenBatchNormalizationForwardInference");
}

void MiOpenBackend::batchNormBackward(const BatchNormDesc& bn, const TensorDesc& xd, const void* x, const TensorDesc& dyd, const void* dy, const void* scale, const void* mean, const void* var, const TensorDesc& dxd, void* dx, void* ds, void* db, const ExecutionContext& ctx)
{
    requireRocm(ctx, "batchNormBackward");
    xd.require4d("batchNormBackward");
    Handle h(native_->mutex, native_->handles, ctx);
    TensorDesc cd = channelDesc(xd);
    Tensor Xd(xd), DYd(dyd), DXd(dxd), Bd(cd);
    float a = 1, b = 0;
    dnn_contract::trace("ROCM", "batchNormBackward", "miopenBatchNormalizationBackward");
    check(
        dnn_dyn::p_miopenBatchNormalizationBackward(
            h.v, miopenBNSpatial, &a, &b, &a, &b, Xd.v, x, DYd.v, dy, DXd.v, dx, Bd.v,
            scale, ds, db, bn.epsilon, mean, var),
        "miopenBatchNormalizationBackward");
}

namespace {

// Statistics descriptor [d0, ..., d(n-2), 1] for normalizations over the last
// dimension.  MIOpen writes them in the datatype of the normalized tensor.
std::vector<std::int64_t> statDims(const TensorDesc& d)
{
    std::vector<std::int64_t> dims = d.dims;
    dims.back() = 1;
    return dims;
}

std::size_t rowCount(const TensorDesc& d)
{
    return d.elements() / static_cast<std::size_t>(d.dims.back());
}

void requirePackedNorm(const TensorDesc& d, const char* where)
{
    if (!d.isPacked())
        throw DnnUnsupportedError(std::string("CLAP_DNN MIOpen ") + where + ": MIOpen normalization kernels require packed tensors");
}

void requireFloat32Stats(const TensorDesc& d, const char* where)
{
    if (d.type != DataType::Float32)
        throw DnnUnsupportedError(std::string("CLAP_DNN MIOpen ") + where +
                                  ": MIOpen stores normalization statistics in the tensor datatype; "
                                  "the CLAP Float32 statistics contract is met for Float32 tensors only");
}

} // namespace

// LayerNorm over the last dimension (miopenLayerNormForward with
// MIOPEN_WEIGHT_BIAS so that scale and bias are applied).
void MiOpenBackend::layerNormForward(const LayerNormDesc& ln, const TensorDesc& xd, const void* x, const TensorDesc& sd, const void* scale, const void* bias, const TensorDesc& yd, void* y, void* mean, void* rstd, const ExecutionContext& ctx)
{
    requireRocm(ctx, "layerNormForward");
    dnn_contract::requireSameShape(xd, yd, "layerNormForward");
    requirePackedNorm(xd, "layerNormForward");
    requirePackedNorm(yd, "layerNormForward");
    if (mean || rstd)
        requireFloat32Stats(xd, "layerNormForward");
    if (sd.elements() != static_cast<std::size_t>(xd.dims.back()) || !sd.isPacked())
        throw std::runtime_error("layerNormForward: scale/bias must be packed with one value per normalized element");

    Handle h(native_->mutex, native_->handles, ctx);
    Scratch scratch(ctx);
    const std::vector<std::int64_t> wdims = {xd.dims.back()};
    const std::vector<std::int64_t> mdims = statDims(xd);
    Tensor Xd(xd), Yd(yd), Sd(sd.type, wdims, packedStrides(wdims)), Md(xd.type, mdims, packedStrides(mdims));

    const std::size_t stat_bytes = rowCount(xd) * xd.elementSize();
    if (!mean) mean = scratch.get(stat_bytes);
    if (!rstd) rstd = scratch.get(stat_bytes);

    dnn_contract::trace("ROCM", "layerNormForward", "miopenLayerNormForward");
    check(
        dnn_dyn::p_miopenLayerNormForward(
            h.v, MIOPEN_WEIGHT_BIAS, Xd.v, x, Sd.v, scale, Sd.v, bias,
            (float) ln.epsilon, (std::int32_t) (xd.dims.size() - 1),
            Yd.v, y, Md.v, mean, Md.v, rstd),
        "miopenLayerNormForward");
}

void MiOpenBackend::layerNormBackward(const LayerNormDesc& , const TensorDesc& xd, const void* x, const TensorDesc& dyd, const void* dy, const TensorDesc& sd, const void* scale, const void* mean, const void* rstd, const TensorDesc& dxd, void* dx, void* dscale, void* dbias, const ExecutionContext& ctx)
{
    requireRocm(ctx, "layerNormBackward");
    dnn_contract::requireSameShape(xd, dyd, "layerNormBackward");
    dnn_contract::requireSameShape(xd, dxd, "layerNormBackward");
    requirePackedNorm(xd, "layerNormBackward");
    requireFloat32Stats(xd, "layerNormBackward");
    if (!mean || !rstd)
        throw std::runtime_error("layerNormBackward requires saved_mean and saved_rstd from layerNormForward");

    Handle h(native_->mutex, native_->handles, ctx);
    Scratch scratch(ctx);
    const std::vector<std::int64_t> wdims = {xd.dims.back()};
    const std::vector<std::int64_t> mdims = statDims(xd);
    Tensor Xd(xd), DYd(dyd), DXd(dxd), Sd(sd.type, wdims, packedStrides(wdims)), Md(xd.type, mdims, packedStrides(mdims));
    const std::int32_t normalized_dim = (std::int32_t) (xd.dims.size() - 1);

    if (!dscale) dscale = scratch.get(sd.bytes());
    if (!dbias) dbias = scratch.get(sd.bytes());

    std::size_t ws = 0;
    check(
        dnn_dyn::p_miopenGetLayerNormBackwardWorkspaceSize(
            h.v, MIOPEN_WEIGHT_BIAS, DYd.v, Xd.v, Sd.v, Md.v, Md.v, normalized_dim,
            DXd.v, Sd.v, Sd.v, &ws),
        "miopenGetLayerNormBackwardWorkspaceSize");
    void* W = scratch.get(ws);

    dnn_contract::trace("ROCM", "layerNormBackward", "miopenLayerNormBackward");
    check(
        dnn_dyn::p_miopenLayerNormBackward(
            h.v, MIOPEN_WEIGHT_BIAS, W, ws,
            DYd.v, dy, Xd.v, x, Sd.v, scale, Md.v, mean, Md.v, rstd,
            normalized_dim, DXd.v, dx, Sd.v, dscale, Sd.v, dbias),
        "miopenLayerNormBackward");
}

// RMSNorm through MIOpen T5LayerNorm (MIOPEN_WEIGHT_BIAS_T5), which computes
// y = x * rsqrt(mean(x^2) + eps) * weight over the last dimension.
void MiOpenBackend::rmsNormForward(const RmsNormDesc& norm, const TensorDesc& xd, const void* x, const TensorDesc& wd, const void* weight, const TensorDesc& yd, void* y, void* saved_rstd, const ExecutionContext& ctx)
{
    requireRocm(ctx, "rmsNormForward");
    if (!dnn_dyn::hasMiopenT5LayerNorm())
        dnn_contract::unsupported("MIOpen", "the installed MIOpen does not export miopenT5LayerNorm* (RMSNorm)");
    dnn_contract::normalizeLastAxis(xd, norm.axis, "rmsNormForward");
    dnn_contract::requireSameShape(xd, yd, "rmsNormForward");
    requirePackedNorm(xd, "rmsNormForward");
    requirePackedNorm(yd, "rmsNormForward");
    if (saved_rstd)
        requireFloat32Stats(yd, "rmsNormForward");
    if (wd.elements() != static_cast<std::size_t>(xd.dims.back()) || !wd.isPacked())
        throw std::runtime_error("rmsNormForward: weight must be packed with one value per normalized element");

    Handle h(native_->mutex, native_->handles, ctx);
    Scratch scratch(ctx);
    const std::vector<std::int64_t> wdims = {xd.dims.back()};
    const std::vector<std::int64_t> rdims = statDims(xd);
    Tensor Xd(xd), Yd(yd), Wd(wd.type, wdims, packedStrides(wdims)), Rd(yd.type, rdims, packedStrides(rdims));

    if (!saved_rstd)
        saved_rstd = scratch.get(rowCount(xd) * yd.elementSize());

    dnn_contract::trace("ROCM", "rmsNormForward", "miopenT5LayerNormForward");
    check(
        dnn_dyn::p_miopenT5LayerNormForward(
            h.v, MIOPEN_WEIGHT_BIAS_T5, Xd.v, x, Wd.v, weight,
            (float) norm.epsilon, Yd.v, y, Rd.v, saved_rstd),
        "miopenT5LayerNormForward");
}

void MiOpenBackend::rmsNormBackward(const RmsNormDesc& norm, const TensorDesc& xd, const void* x, const TensorDesc& dyd, const void* dy, const TensorDesc& wd, const void* weight, const void* saved_rstd, const TensorDesc& dxd, void* dx, void* dweight, const ExecutionContext& ctx)
{
    requireRocm(ctx, "rmsNormBackward");
    if (!dnn_dyn::hasMiopenT5LayerNorm())
        dnn_contract::unsupported("MIOpen", "the installed MIOpen does not export miopenT5LayerNorm* (RMSNorm)");
    dnn_contract::normalizeLastAxis(xd, norm.axis, "rmsNormBackward");
    dnn_contract::requireSameShape(xd, dyd, "rmsNormBackward");
    dnn_contract::requireSameShape(xd, dxd, "rmsNormBackward");
    requirePackedNorm(xd, "rmsNormBackward");
    requireFloat32Stats(dyd, "rmsNormBackward");
    if (!saved_rstd)
        throw std::runtime_error("rmsNormBackward requires saved_rstd from rmsNormForward");

    Handle h(native_->mutex, native_->handles, ctx);
    Scratch scratch(ctx);
    const std::vector<std::int64_t> wdims = {xd.dims.back()};
    const std::vector<std::int64_t> rdims = statDims(xd);
    Tensor Xd(xd), DYd(dyd), DXd(dxd), Wd(wd.type, wdims, packedStrides(wdims)), Rd(dyd.type, rdims, packedStrides(rdims));

    if (!dweight)
        dweight = scratch.get(wd.bytes());

    std::size_t ws = 0;
    check(
        dnn_dyn::p_miopenGetT5LayerNormBackwardWorkspaceSize(
            h.v, MIOPEN_WEIGHT_BIAS_T5, DYd.v, Xd.v, Wd.v, Rd.v, DXd.v, Wd.v, &ws),
        "miopenGetT5LayerNormBackwardWorkspaceSize");
    void* W = scratch.get(ws);

    dnn_contract::trace("ROCM", "rmsNormBackward", "miopenT5LayerNormBackward");
    check(
        dnn_dyn::p_miopenT5LayerNormBackward(
            h.v, MIOPEN_WEIGHT_BIAS_T5, W, ws,
            DYd.v, dy, Xd.v, x, Wd.v, weight, Rd.v, saved_rstd,
            DXd.v, dx, Wd.v, dweight),
        "miopenT5LayerNormBackward");
}

std::size_t MiOpenBackend::dropoutReserveSpaceSize(const TensorDesc& xd, const ExecutionContext& ctx)
{
    requireRocm(ctx, "dropoutReserveSpaceSize");
    Tensor Xd(xd);
    std::size_t rs = 0;
    check(dnn_dyn::p_miopenDropoutGetReserveSpaceSize(Xd.v, &rs), "miopenDropoutGetReserveSpaceSize");
    return rs;
}

// MIOpen dropout on caller memory; the PRNG states are stream-ordered scratch
// initialized by miopenSetDropoutDescriptor on the caller stream.
void MiOpenBackend::dropoutForward(const DropoutDesc& dropout, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, void* reserve, std::size_t reserve_bytes, const ExecutionContext& ctx)
{
    requireRocm(ctx, "dropoutForward");
    Handle h(native_->mutex, native_->handles, ctx);
    Scratch scratch(ctx);
    Tensor Xd(xd), Yd(yd);
    miopenDropoutDescriptor_t dd = nullptr;
    check(dnn_dyn::p_miopenCreateDropoutDescriptor(&dd), "miopenCreateDropoutDescriptor");
    try {
        std::size_t ss = 0;
        check(dnn_dyn::p_miopenDropoutGetStatesSize(h.v, &ss), "miopenDropoutGetStatesSize");
        void* states = scratch.get(ss);
        check(
            dnn_dyn::p_miopenSetDropoutDescriptor(
                dd, h.v, dropout.probability, states, ss, dropout.seed, false, false, 0),
            "miopenSetDropoutDescriptor");
        dnn_contract::trace("ROCM", "dropoutForward", "miopenDropoutForward");
        check(
            dnn_dyn::p_miopenDropoutForward(
                h.v, dd, Xd.v, Xd.v, x, Yd.v, y, reserve, reserve_bytes),
            "miopenDropoutForward");
        dnn_dyn::p_miopenDestroyDropoutDescriptor(dd);
    }
    catch (...) {
        dnn_dyn::p_miopenDestroyDropoutDescriptor(dd);
        throw;
    }
}

void MiOpenBackend::dropoutBackward(const DropoutDesc& dropout, const TensorDesc& dyd, const void* dy, const void* reserve, std::size_t reserve_bytes, const TensorDesc& dxd, void* dx, const ExecutionContext& ctx)
{
    requireRocm(ctx, "dropoutBackward");
    Handle h(native_->mutex, native_->handles, ctx);
    Scratch scratch(ctx);
    Tensor DYd(dyd), DXd(dxd);
    miopenDropoutDescriptor_t dd = nullptr;
    check(dnn_dyn::p_miopenCreateDropoutDescriptor(&dd), "miopenCreateDropoutDescriptor");
    try {
        std::size_t ss = 0;
        check(dnn_dyn::p_miopenDropoutGetStatesSize(h.v, &ss), "miopenDropoutGetStatesSize");
        void* states = scratch.get(ss);
        check(
            dnn_dyn::p_miopenSetDropoutDescriptor(
                dd, h.v, dropout.probability, states, ss, dropout.seed, false, false, 0),
            "miopenSetDropoutDescriptor");
        dnn_contract::trace("ROCM", "dropoutBackward", "miopenDropoutBackward");
        check(
            dnn_dyn::p_miopenDropoutBackward(
                h.v, dd, DYd.v, DYd.v, dy, DXd.v, dx, const_cast<void*>(reserve), reserve_bytes),
            "miopenDropoutBackward");
        dnn_dyn::p_miopenDestroyDropoutDescriptor(dd);
    }
    catch (...) {
        dnn_dyn::p_miopenDestroyDropoutDescriptor(dd);
        throw;
    }
}

void MiOpenBackend::lrnForward(const LrnDesc& lrn, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "lrnForward");
    xd.require4d("lrnForward");
    Handle h(native_->mutex, native_->handles, ctx);
    Tensor Xd(xd), Yd(yd);
    abi::miopenLRNDescriptor_t d = nullptr;
    check(dnn_dyn::p_miopenCreateLRNDescriptor(&d), "miopenCreateLRNDescriptor");
    try {
        check(
            dnn_dyn::p_miopenSetLRNDescriptor(
                d, miopenLRNCrossChannel, (unsigned)lrn.local_size, lrn.alpha, lrn.beta, lrn.k),
            "miopenSetLRNDescriptor");

        float one = 1, zero = 0;
        dnn_contract::trace("ROCM", "lrnForward", "miopenLRNForward");
        check(
            dnn_dyn::p_miopenLRNForward(
                h.v, d, &one, Xd.v, x, &zero, Yd.v, y, false, nullptr),
            "miopenLRNForward");
        dnn_dyn::p_miopenDestroyLRNDescriptor(d);
    }
    catch (...) {
        if (d)
            dnn_dyn::p_miopenDestroyLRNDescriptor(d);
        throw;
    }
}

void MiOpenBackend::lrnBackward(const LrnDesc& lrn, const TensorDesc& xd, const void* x, const TensorDesc& yd, const void* y, const TensorDesc& dyd, const void* dy, const TensorDesc& dxd, void* dx, const ExecutionContext& ctx)
{
    requireRocm(ctx, "lrnBackward");
    xd.require4d("lrnBackward");
    Handle h(native_->mutex, native_->handles, ctx);
    Tensor Xd(xd), Yd(yd), DYd(dyd), DXd(dxd);
    abi::miopenLRNDescriptor_t d = nullptr;
    check(dnn_dyn::p_miopenCreateLRNDescriptor(&d), "miopenCreateLRNDescriptor");
    try {
        check(
            dnn_dyn::p_miopenSetLRNDescriptor(
                d, miopenLRNCrossChannel, (unsigned)lrn.local_size, lrn.alpha, lrn.beta, lrn.k),
            "miopenSetLRNDescriptor");

        // miopenLRNBackward consumes the workspace written by a forward call
        // with do_backward=true.  The CLAP API does not carry that workspace,
        // so it is regenerated on the caller stream into scratch memory.
        Scratch scratch(ctx);
        std::size_t ws_bytes = 0;
        check(
            dnn_dyn::p_miopenLRNGetWorkSpaceSize(
                Yd.v,
                &ws_bytes),
            "miopenLRNGetWorkSpaceSize");
        void* workspace = scratch.get(ws_bytes);
        void* y_regen = scratch.get(yd.bytes());

        float one = 1, zero = 0;
        dnn_contract::trace("ROCM", "lrnBackward", "miopenLRNForward(do_backward)");
        check(
            dnn_dyn::p_miopenLRNForward(
                h.v,
                d,
                &one,
                Xd.v,
                x,
                &zero,
                Yd.v,
                y_regen,
                true,
                workspace),
            "miopenLRNForward");

        dnn_contract::trace("ROCM", "lrnBackward", "miopenLRNBackward");
        check(
            dnn_dyn::p_miopenLRNBackward(
                h.v,
                d,
                &one,
                Yd.v,
                y,
                DYd.v,
                dy,
                Xd.v,
                x,
                &zero,
                DXd.v,
                dx,
                workspace),
            "miopenLRNBackward");
        dnn_dyn::p_miopenDestroyLRNDescriptor(d);
    }
    catch (...) {
        if (d)
            dnn_dyn::p_miopenDestroyLRNDescriptor(d);
        throw;
    }
}

static void miopenOp(abi::miopenHandle_t h, abi::miopenTensorOp_t op, const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y)
{
    Tensor A(ad), B(bd), C(yd);
    float one = 1, zero = 0;
    dnn_contract::trace("ROCM", "tensorOp", "miopenOpTensor");
    check(
        dnn_dyn::p_miopenOpTensor(
            h, op, &one, A.v, a, &one, B.v, b, &zero, C.v, y),
        "miopenOpTensor");
}

void MiOpenBackend::tensorAdd(const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "tensorAdd");
    Handle h(native_->mutex, native_->handles, ctx);
    miopenOp(h.v, miopenTensorOpAdd, ad, a, bd, b, yd, y);
}

void MiOpenBackend::tensorMultiply(const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "tensorMultiply");
    Handle h(native_->mutex, native_->handles, ctx);
    miopenOp(h.v, miopenTensorOpMul, ad, a, bd, b, yd, y);
}

void MiOpenBackend::tensorMin(const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "tensorMin");
    Handle h(native_->mutex, native_->handles, ctx);
    miopenOp(h.v, miopenTensorOpMin, ad, a, bd, b, yd, y);
}

void MiOpenBackend::tensorMax(const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "tensorMax");
    Handle h(native_->mutex, native_->handles, ctx);
    miopenOp(h.v, miopenTensorOpMax, ad, a, bd, b, yd, y);
}

static void miopenReduce(abi::miopenHandle_t h, Scratch& scratch, abi::miopenReduceTensorOp_t op, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y)
{
    Tensor X(xd), Y(yd);

    abi::miopenReduceTensorDescriptor_t d = nullptr;
    check(dnn_dyn::p_miopenCreateReduceTensorDescriptor(&d), "miopenCreateReduceTensorDescriptor");
    try {
        check(
            dnn_dyn::p_miopenSetReduceTensorDescriptor(
                d, op, miopenFloat, miopenNotPropagateNaN, miopenReduceTensorNoIndices, miopen32BitIndices),
            "miopenSetReduceTensorDescriptor");

        std::size_t ws = 0;
        check(
            dnn_dyn::p_miopenGetReductionWorkspaceSize(h, d, X.v, Y.v, &ws),
            "miopenGetReductionWorkspaceSize");
        void* W = scratch.get(ws);
        float one = 1, zero = 0;
        dnn_contract::trace("ROCM", "tensorReduce", "miopenReduceTensor");
        check(
            dnn_dyn::p_miopenReduceTensor(
                h, d, nullptr, 0, W, ws, &one, X.v, x, &zero, Y.v, y),
            "miopenReduceTensor");
        dnn_dyn::p_miopenDestroyReduceTensorDescriptor(d);
    }
    catch (...) {
        dnn_dyn::p_miopenDestroyReduceTensorDescriptor(d);
        throw;
    }
}

void MiOpenBackend::tensorReduceSum(const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "tensorReduceSum");
    Handle h(native_->mutex, native_->handles, ctx);
    Scratch scratch(ctx);
    miopenReduce(h.v, scratch, miopenReduceTensorAdd, xd, x, yd, y);
}

void MiOpenBackend::tensorReduceProduct(const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    requireRocm(ctx, "tensorReduceProduct");
    Handle h(native_->mutex, native_->handles, ctx);
    Scratch scratch(ctx);
    miopenReduce(h.v, scratch, miopenReduceTensorMul, xd, x, yd, y);
}

// ==========================================================================
// Scaled dot-product attention through the MIOpen MHA Find-2.0 API
// (miopenCreateMhaProblem / miopenFindSolutions / miopenRunSolution).
//
// The MIOpen MHA forward solver computes softmax(scale * Q K^T) V for packed
// [B, H, S, D] Float32 tensors with equal query and key/value sequence
// lengths and equal head counts.  It has no additive-mask, causal or GQA
// support, so those configurations are reported as unsupported.
// ==========================================================================

namespace {

// Device layout of the constant MHA scalars shared by all calls on a device.
struct MhaConstants {
    float descale_k = 1.0f;
    float descale_q = 1.0f;
    float descale_v = 1.0f;
    float descale_s = 1.0f;
    float scale_s = 1.0f;
    float scale_o = 1.0f;
    float dropout_probability = 0.0f;
    float padding = 0.0f;
    std::int64_t dropout_seed = 0;
    std::int64_t dropout_offset = 0;
};

std::string mhaUnsupportedReason(const dnn_contract::AttentionShape& shape,
                                 const AttentionDesc& attention,
                                 const TensorDesc& qd,
                                 const TensorDesc& kd,
                                 const TensorDesc& vd,
                                 const TensorDesc& od)
{
    if (!dnn_dyn::hasMiopenMha())
        return "the installed MIOpen does not export the MHA Find-2.0 API";
    if (qd.type != DataType::Float32)
        return "the MIOpen MHA forward solver supports Float32 (and FP8) only";
    if (shape.has_mask)
        return "the MIOpen MHA forward solver has no additive attention mask";
    if (attention.causal)
        return "the MIOpen MHA forward solver does not implement causal masking";
    if (attention.dropout > 0.0f)
        return "attention dropout is not wired for the MIOpen MHA path in this release";
    if (shape.q_heads != shape.kv_heads)
        return "the MIOpen MHA forward solver requires equal query and key/value head counts (no GQA/MQA)";
    if (shape.q_len != shape.kv_len)
        return "the MIOpen MHA forward solver requires equal query and key/value sequence lengths";
    if (shape.head_dim != shape.value_dim)
        return "the MIOpen MHA forward solver requires equal Q/K and V head dimensions";
    if (!qd.isPacked() || !kd.isPacked() || !vd.isPacked() || !od.isPacked())
        return "the MIOpen MHA forward solver requires packed [B, H, S, D] tensors";
    return {};
}

} // namespace

void MiOpenBackend::scaledDotProductAttentionForward(const AttentionDesc& attention,
                                                     const TensorDesc& qd,
                                                     const void* q,
                                                     const TensorDesc& kd,
                                                     const void* k,
                                                     const TensorDesc& vd,
                                                     const void* v,
                                                     const TensorDesc* mask_desc,
                                                     const void* mask,
                                                     const TensorDesc& od,
                                                     void* o,
                                                     const ExecutionContext& ctx)
{
    requireRocm(ctx, "scaledDotProductAttentionForward");

    if ((mask_desc == nullptr) != (mask == nullptr))
        throw std::runtime_error("scaledDotProductAttentionForward: mask_desc and mask must both be set or both be null");

    const auto shape = dnn_contract::validateAttention(attention, qd, kd, vd, mask_desc, od);
    const std::string reason = mhaUnsupportedReason(shape, attention, qd, kd, vd, od);
    if (!reason.empty())
        dnn_contract::unsupported("MIOpen", reason);

    Handle h(native_->mutex, native_->handles, ctx);
    Scratch scratch(ctx);

    int device = 0;
    checkHip(dnn_dyn::p_hipGetDevice(&device), "hipGetDevice");

    // Constant scalars: one small device block per device, written once.
    void*& constants = native_->mha_constants[device];
    if (!constants) {
        const MhaConstants init;
        checkHip(dnn_dyn::p_hipMalloc(&constants, sizeof(MhaConstants)), "hipMalloc(MHA constants)");
        checkHip(dnn_dyn::p_hipMemcpy(constants, &init, sizeof(MhaConstants), hipMemcpyHostToDevice), "hipMemcpy(MHA constants)");
    }

    const std::vector<std::int64_t> bhsd = {shape.batch, shape.q_heads, shape.q_len, shape.head_dim};
    const std::vector<std::int64_t> bhs1 = {shape.batch, shape.q_heads, shape.q_len, 1};
    const std::vector<std::int64_t> one = {1, 1, 1, 1};

    Tensor T_qkvo(DataType::Float32, bhsd, packedStrides(bhsd));
    Tensor T_stats(DataType::Float32, bhs1, packedStrides(bhs1));
    Tensor T_scalar(DataType::Float32, one, one);

    miopenTensorDescriptor_t T_index = nullptr;
    check(dnn_dyn::p_miopenCreateTensorDescriptor(&T_index), "miopenCreateTensorDescriptor");
    struct IndexGuard {
        miopenTensorDescriptor_t v;
        ~IndexGuard() { if (v) dnn_dyn::p_miopenDestroyTensorDescriptor(v); }
    } index_guard{T_index};
    const int one_dims[4] = {1, 1, 1, 1};
    check(
        dnn_dyn::p_miopenSetTensorDescriptor(T_index, miopenInt64, 4, one_dims, one_dims),
        "miopenSetTensorDescriptor(int64 scalar)");

    std::ostringstream key;
    key << "mha|" << h.v << '|' << shape.batch << ',' << shape.q_heads << ',' << shape.q_len << ',' << shape.head_dim << '|' << shape.scale;

    auto& slot = native_->mha[key.str()];
    if (!slot) {
        miopenMhaDescriptor_t mha = nullptr;
        check(dnn_dyn::p_miopenCreateMhaDescriptor(&mha), "miopenCreateMhaDescriptor");
        check(dnn_dyn::p_miopenSetMhaDescriptor(mha, (float) shape.scale), "miopenSetMhaDescriptor");

        miopenProblem_t problem = nullptr;
        check(dnn_dyn::p_miopenCreateMhaProblem(&problem, mha, miopenProblemDirectionForward), "miopenCreateMhaProblem");

        struct ProblemGuard {
            miopenProblem_t v;
            ~ProblemGuard() { if (v) dnn_dyn::p_miopenDestroyProblem(v); }
        } problem_guard{problem};

        const struct {
            int id;
            miopenTensorDescriptor_t desc;
        } layout[] = {
            {miopenTensorMhaK, T_qkvo.v},
            {miopenTensorMhaQ, T_qkvo.v},
            {miopenTensorMhaV, T_qkvo.v},
            {miopenTensorMhaDescaleK, T_scalar.v},
            {miopenTensorMhaDescaleQ, T_scalar.v},
            {miopenTensorMhaDescaleV, T_scalar.v},
            {miopenTensorMhaDescaleS, T_scalar.v},
            {miopenTensorMhaScaleS, T_scalar.v},
            {miopenTensorMhaScaleO, T_scalar.v},
            {miopenTensorMhaDropoutProbability, T_scalar.v},
            {miopenTensorMhaDropoutSeed, T_index},
            {miopenTensorMhaDropoutOffset, T_index},
            {miopenTensorMhaO, T_qkvo.v},
            {miopenTensorMhaAmaxO, T_scalar.v},
            {miopenTensorMhaAmaxS, T_scalar.v},
            {miopenTensorMhaM, T_stats.v},
            {miopenTensorMhaZInv, T_stats.v}
        };

        for (const auto& entry : layout) {
            check(
                dnn_dyn::p_miopenSetProblemTensorDescriptor(problem, entry.id, entry.desc),
                "miopenSetProblemTensorDescriptor");
        }

        std::unique_ptr<miopen_detail::MhaSolution> found(new miopen_detail::MhaSolution());
        std::size_t count = 0;
        check(
            dnn_dyn::p_miopenFindSolutions(h.v, problem, nullptr, &found->solution, &count, 1),
            "miopenFindSolutions(MHA)");
        if (count == 0 || !found->solution)
            dnn_contract::unsupported("MIOpen", "miopenFindSolutions found no MHA forward solver for this problem");

        check(
            dnn_dyn::p_miopenGetSolutionWorkspaceSize(found->solution, &found->workspace),
            "miopenGetSolutionWorkspaceSize");
        slot = std::move(found);
    }

    char* c = static_cast<char*>(constants);
    const std::size_t stat_bytes = static_cast<std::size_t>(shape.batch * shape.q_heads * shape.q_len) * sizeof(float);
    void* m_stats = scratch.get(stat_bytes);
    void* z_inv = scratch.get(stat_bytes);
    void* amax_o = scratch.get(sizeof(float));
    void* amax_s = scratch.get(sizeof(float));
    void* workspace = scratch.get(slot->workspace);

    const miopenTensorArgument_t args[] = {
        {miopenTensorMhaK, nullptr, const_cast<void*>(k)},
        {miopenTensorMhaQ, nullptr, const_cast<void*>(q)},
        {miopenTensorMhaV, nullptr, const_cast<void*>(v)},
        {miopenTensorMhaDescaleK, nullptr, c + offsetof(MhaConstants, descale_k)},
        {miopenTensorMhaDescaleQ, nullptr, c + offsetof(MhaConstants, descale_q)},
        {miopenTensorMhaDescaleV, nullptr, c + offsetof(MhaConstants, descale_v)},
        {miopenTensorMhaDescaleS, nullptr, c + offsetof(MhaConstants, descale_s)},
        {miopenTensorMhaScaleS, nullptr, c + offsetof(MhaConstants, scale_s)},
        {miopenTensorMhaScaleO, nullptr, c + offsetof(MhaConstants, scale_o)},
        {miopenTensorMhaDropoutProbability, nullptr, c + offsetof(MhaConstants, dropout_probability)},
        {miopenTensorMhaDropoutSeed, nullptr, c + offsetof(MhaConstants, dropout_seed)},
        {miopenTensorMhaDropoutOffset, nullptr, c + offsetof(MhaConstants, dropout_offset)},
        {miopenTensorMhaO, nullptr, o},
        {miopenTensorMhaAmaxO, nullptr, amax_o},
        {miopenTensorMhaAmaxS, nullptr, amax_s},
        {miopenTensorMhaM, nullptr, m_stats},
        {miopenTensorMhaZInv, nullptr, z_inv}
    };

    dnn_contract::trace("ROCM", "scaledDotProductAttentionForward", "miopenRunSolution(MHA forward)");
    check(
        dnn_dyn::p_miopenRunSolution(
            h.v, slot->solution, sizeof(args) / sizeof(args[0]), args, workspace, slot->workspace),
        "miopenRunSolution(MHA)");
}

// ==========================================================================
// Capability model.  MIOpen capabilities are decided from the datatype,
// descriptor layout and the optional beta entry points that the installed
// MIOpen exports; MIOpen remains the final arbiter at execution time.
// ==========================================================================

DnnSupport MiOpenBackend::supports(const DnnCapabilityQuery& query, const ExecutionContext& ctx) const
{
    if (ctx.backend != ExecutionBackend::ROCM)
        return DnnSupport::no("the MIOpen backend executes on HIP device memory (ExecutionBackend::ROCM)");

    try {
        const TensorDesc& x = query.tensor(0);

        for (const auto& t : query.tensors) {
            if (t.dims.empty() || t.dims.size() > 5)
                return DnnSupport::no("MIOpen tensors must have rank 1..5");
            t.spanElements();
        }

        switch (query.operation) {
            case DnnOperation::ActivationForward:
            case DnnOperation::ActivationBackward:
                if (query.activation.mode == ActivationMode::SiLU)
                    return DnnSupport::yes("composed from native MIOpen primitives: LOGISTIC activation + miopenOpTensor (MIOpen has no SiLU mode)");
                return DnnSupport::yes("miopenActivation*");

            case DnnOperation::SoftmaxForward:
            case DnnOperation::SoftmaxBackward: {
                const int axis = x.normalizeAxis(query.softmax.axis, "supports(softmax)");
                dnn_contract::AxisView view;
                if (!dnn_contract::makeAxisView(x, axis, view))
                    return DnnSupport::no("the dimensions around the softmax axis cannot be collapsed into an MIOpen [N, C, H, 1] view");
                return DnnSupport::yes(query.softmax.log_softmax
                    ? "miopenSoftmax*_V2 with MIOPEN_SOFTMAX_LOG"
                    : "miopenSoftmax*_V2 with MIOPEN_SOFTMAX_ACCURATE");
            }

            case DnnOperation::LayerNormForward:
            case DnnOperation::LayerNormBackward:
                if (!x.isPacked())
                    return DnnSupport::no("MIOpen LayerNorm requires packed tensors");
                if (query.operation == DnnOperation::LayerNormBackward && x.type != DataType::Float32)
                    return DnnSupport::no("MIOpen stores LayerNorm statistics in the tensor datatype; Float32 statistics require Float32 tensors");
                return DnnSupport::yes("miopenLayerNorm* (MIOPEN_WEIGHT_BIAS)");

            case DnnOperation::RmsNormForward:
            case DnnOperation::RmsNormBackward:
                if (!dnn_dyn::hasMiopenT5LayerNorm())
                    return DnnSupport::no("the installed MIOpen does not export miopenT5LayerNorm* (RMSNorm)");
                dnn_contract::normalizeLastAxis(x, query.rms_norm.axis, "supports(rmsNorm)");
                if (!x.isPacked())
                    return DnnSupport::no("MIOpen T5LayerNorm requires packed tensors");
                if (query.operation == DnnOperation::RmsNormBackward && x.type != DataType::Float32)
                    return DnnSupport::no("MIOpen stores RMSNorm statistics in the tensor datatype; Float32 statistics require Float32 tensors");
                return DnnSupport::yes("miopenT5LayerNorm* (MIOPEN_WEIGHT_BIAS_T5)");

            case DnnOperation::ScaledDotProductAttentionForward: {
                const TensorDesc* mask = query.has_mask ? &query.tensor(4) : nullptr;
                const auto shape = dnn_contract::validateAttention(
                    query.attention, query.tensor(0), query.tensor(1), query.tensor(2), mask, query.tensor(3));
                const std::string reason = mhaUnsupportedReason(
                    shape, query.attention, query.tensor(0), query.tensor(1), query.tensor(2), query.tensor(3));
                if (!reason.empty())
                    return DnnSupport::no(reason);
                return DnnSupport::yes("MIOpen MHA Find-2.0 forward solver (solver availability is checked by miopenFindSolutions)");
            }

            case DnnOperation::FusedConvolutionBiasActivation:
                if (query.activation.mode == ActivationMode::SiLU)
                    return DnnSupport::no("MIOpen has no SiLU activation mode for the fused convolution path");
                if (x.rank() != 4)
                    return DnnSupport::no("convolution expects 4D NCHW tensors");
                return DnnSupport::yes("miopenConvolutionForward + miopenConvolutionForwardBias + miopenActivationForward");

            case DnnOperation::DropoutForward:
            case DnnOperation::DropoutBackward:
                return DnnSupport::yes("miopenDropout* with caller-owned reserve space");

            case DnnOperation::TensorAdd:
            case DnnOperation::TensorMultiply:
            case DnnOperation::TensorMin:
            case DnnOperation::TensorMax:
                return DnnSupport::yes("miopenOpTensor");

            case DnnOperation::TensorReduceSum:
            case DnnOperation::TensorReduceProduct:
                return DnnSupport::yes("miopenReduceTensor");

            default:
                if (x.rank() != 4)
                    return DnnSupport::no(std::string(dnnOperationName(query.operation)) + " expects 4D NCHW tensors");
                return DnnSupport::yes();
        }
    }
    catch (const std::exception& e) {
        return DnnSupport::no(e.what());
    }
}

}
// namespace clap
