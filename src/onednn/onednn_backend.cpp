#include "clap/onednn_backend.hpp"
#include "clap/dnn_dyn_backends.hpp"

#include "clap/dnn_contract_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
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
        void check(dnnl_status_t s, const char* what) {
            if (s != dnnl_success) throw std::runtime_error(std::string("oneDNN call failed: ") + what + " status=" + std::to_string(s));
        }
        // dnnl_unimplemented from a primitive descriptor constructor means that
        // the installed oneDNN has no implementation for the requested
        // datatype/layout/ISA combination.
        void checkPd(dnnl_status_t s, const char* what) {
            if (s == dnnl_unimplemented) throw DnnUnsupportedError(std::string("CLAP_DNN oneDNN: no native implementation for ") + what + " with the requested datatype/layout");
            check(s, what);
        }
        struct Engine {
            dnnl_engine_t v {
            };
            Engine() {
                check(dnn_dyn::p_dnnl_engine_create( &v, dnnl_cpu, 0), "dnnl_engine_create");
            }
            ~Engine() {
                if (v) dnn_dyn::p_dnnl_engine_destroy(v);
            }
        };
        dnnl_data_type_t dnnlType(DataType t) {
            switch (t) {
                case DataType::Float32:  return dnnl_f32;
                case DataType::Float16:  return dnnl_f16;
                case DataType::BFloat16: return dnnl_bf16;
            }
            return dnnl_f32;
        }
        struct Md {
            dnnl_memory_desc_t v {
            };
            explicit Md(const TensorDesc& d) {
                const int nd = (int)d.dims.size();
                std::vector<dnnl_dim_t> dims(d.dims.begin(), d.dims.end());
                std::vector<dnnl_dim_t> strides = d.effectiveStrides();
                check(dn_create(nd, dims.data(), dnnlType(d.type), strides.data()), "dnnl_memory_desc_create_with_strides");
            }
            Md(dnnl_data_type_t dt, const std::vector<dnnl_dim_t>& dims, const std::vector<dnnl_dim_t>& strides) {
                check(dn_create((int)dims.size(), dims.data(), dt, strides.data()), "dnnl_memory_desc_create_with_strides");
            }
            ~Md() {
                if (v) dnn_dyn::p_dnnl_memory_desc_destroy(v);
            }
            Md(const Md&) = delete;
            Md& operator=(const Md&) = delete;
        private:
            dnnl_status_t dn_create(int nd, const dnnl_dim_t* dims, dnnl_data_type_t dt, const dnnl_dim_t* strides) {
                return dnn_dyn::p_dnnl_memory_desc_create_with_strides( &v, nd, dims, dt, strides);
            }
        };
        std::vector<dnnl_dim_t> packed(const std::vector<dnnl_dim_t>& dims) {
            std::vector<dnnl_dim_t> s(dims.size(), 1);
            dnnl_dim_t running = 1;
            for (std::size_t i = dims.size(); i-- > 0;) {
                s[i] = running;
                running *= dims[i];
            }
            return s;
        }
        struct Mem {
            dnnl_memory_t v {
            };
            Mem(const Md& md, dnnl_engine_t e, void* p) {
                check(dnn_dyn::p_dnnl_memory_create( &v, md.v, e, p), "dnnl_memory_create");
            }
            Mem(const_dnnl_memory_desc_t md, dnnl_engine_t e, void* p) {
                check(dnn_dyn::p_dnnl_memory_create( &v, md, e, p), "dnnl_memory_create");
            }
            ~Mem() {
                if (v) dnn_dyn::p_dnnl_memory_destroy(v);
            }
            Mem(const Mem&) = delete;
            Mem& operator=(const Mem&) = delete;
        };
        struct Pd {
            dnnl_primitive_desc_t v {
            };
            ~Pd() {
                if (v) dnn_dyn::p_dnnl_primitive_desc_destroy(v);
            }
        };
        struct Prim {
            dnnl_primitive_t v {
            };
            explicit Prim(Pd& pd) {
                check(dnn_dyn::p_dnnl_primitive_create( &v, pd.v), "dnnl_primitive_create");
            }
            ~Prim() {
                if (v) dnn_dyn::p_dnnl_primitive_destroy(v);
            }
        };
        struct Attr {
            dnnl_primitive_attr_t v {
            };
            dnnl_post_ops_t ops {
            };
            Attr() {
                check(dnn_dyn::p_dnnl_primitive_attr_create( &v), "dnnl_primitive_attr_create");
            }
            ~Attr() {
                if (ops) dnn_dyn::p_dnnl_post_ops_destroy(ops);
                if (v) dnn_dyn::p_dnnl_primitive_attr_destroy(v);
            }
            void appendEltwise(dnnl_alg_kind_t alg, float alpha, float beta) {
                if (!ops) check(dnn_dyn::p_dnnl_post_ops_create( &ops), "dnnl_post_ops_create");
                check(dnn_dyn::p_dnnl_post_ops_append_eltwise(ops, alg, alpha, beta), "dnnl_post_ops_append_eltwise");
                check(dnn_dyn::p_dnnl_primitive_attr_set_post_ops(v, ops), "dnnl_primitive_attr_set_post_ops");
            }
        };

        // Execution target of one CLAP call.  When the caller supplies a
        // dnnl_stream_t the primitives are enqueued on it (and on its engine)
        // and CLAP does not wait; otherwise a stream owned by the call is
        // used and drained before returning, which preserves the historical
        // synchronous host-pointer behavior.
        struct Cpu {
            dnnl_engine_t e {
            };
            dnnl_stream_t s {
            };
            bool owned = false;
            Cpu(dnnl_engine_t backend_engine, const ExecutionContext& ctx) {
                dnn_contract::requireBackend(ctx, ExecutionBackend::CPU, "oneDNN", "execution");
                if (ctx.stream) {
                    s = ctx.stream;
                    check(dnn_dyn::p_dnnl_stream_get_engine(s, &e), "dnnl_stream_get_engine");
                } else {
                    e = backend_engine;
                    check(dnn_dyn::p_dnnl_stream_create( &s, e, dnnl_stream_in_order), "dnnl_stream_create");
                    owned = true;
                }
            }
            ~Cpu() {
                if (owned && s) dnn_dyn::p_dnnl_stream_destroy(s);
            }
            Cpu(const Cpu&) = delete;
            Cpu& operator=(const Cpu&) = delete;

            // Always drain: used before host scratch buffers go out of scope.
            void complete() {
                check(dnn_dyn::p_dnnl_stream_wait(s), "dnnl_stream_wait");
            }
        };
        void exec(Prim& p, Cpu& c, std::initializer_list<dnnl_exec_arg_t> a) {
            std::vector<dnnl_exec_arg_t> args(a);
            check(dnn_dyn::p_dnnl_primitive_execute(p.v, c.s, (int)args.size(), args.data()), "dnnl_primitive_execute");
            if (c.owned) check(dnn_dyn::p_dnnl_stream_wait(c.s), "dnnl_stream_wait");
        }
        dnnl_alg_kind_t act_alg(ActivationMode m) {
            if (m == ActivationMode::ReLU) return dnnl_eltwise_relu;
            if (m == ActivationMode::Sigmoid) return dnnl_eltwise_logistic;
            if (m == ActivationMode::SiLU) return dnnl_eltwise_swish;
            return dnnl_eltwise_tanh;
        }
        // SiLU is swish(x) = x * sigmoid(alpha * x) with alpha = 1.
        float act_alpha(const ActivationDesc& a) {
            return a.mode == ActivationMode::SiLU ? 1.0f : (float) a.alpha;
        }
        dnnl_alg_kind_t pool_alg(PoolingMode m) {
            return m == PoolingMode::Max ? dnnl_pooling_max: dnnl_pooling_avg_include_padding;
        }
        void* mut(const void* p) {
            return const_cast<void*>(p);
        }
        struct ConvGeometry {
            dnnl_dim_t st[2];
            dnnl_dim_t dil[2];
            dnnl_dim_t pl[2];
            dnnl_dim_t pr[2];
            explicit ConvGeometry(const ConvolutionDesc& c)
                : st{c.stride_h, c.stride_w},
                  dil{c.dilation_h - 1, c.dilation_w - 1},
                  pl{c.pad_h, c.pad_w},
                  pr{c.pad_h, c.pad_w} {
            }
        };
        struct PoolGeometry {
            dnnl_dim_t st[2];
            dnnl_dim_t k[2];
            dnnl_dim_t dil[2];
            dnnl_dim_t pl[2];
            dnnl_dim_t pr[2];
            explicit PoolGeometry(const PoolingDesc& p)
                : st{p.stride_h, p.stride_w},
                  k{p.window_h, p.window_w},
                  dil{0, 0},
                  pl{p.pad_h, p.pad_w},
                  pr{p.pad_h, p.pad_w} {
            }
        };
        std::size_t workspaceBytes(const Pd& pd) {
            const_dnnl_memory_desc_t md = dnn_dyn::p_dnnl_primitive_desc_query_md(pd.v, dnnl_query_workspace_md, 0);
            return md ? dnn_dyn::p_dnnl_memory_desc_get_size(md) : 0;
        }

        // [rows, inner] view of a tensor normalized over its last dimension.
        struct RowsMd {
            dnn_contract::RowsView view;
            std::unique_ptr<Md> data;
            std::unique_ptr<Md> stats;
            RowsMd(const TensorDesc& d, const char* where) {
                if (!dnn_contract::makeRowsView(d, view))
                    throw DnnUnsupportedError(std::string("CLAP_DNN oneDNN ") + where + ": rows of the normalized tensor are not uniformly strided");
                data.reset(new Md(dnnlType(d.type), {view.rows, view.inner}, {view.row_stride, 1}));
                stats.reset(new Md(dnnl_f32, {view.rows}, {1}));
            }
        };
        std::unique_ptr<Md> vectorMd(const TensorDesc& d, std::int64_t inner, const char* where) {
            if (d.elements() != (std::size_t) inner || !d.isPacked())
                throw std::runtime_error(std::string("CLAP_DNN oneDNN ") + where + ": scale/weight must be a packed tensor with one value per normalized element");
            return std::unique_ptr<Md>(new Md(dnnlType(d.type), {inner}, {1}));
        }

        // Layer normalization primitive descriptors.  The original C entry
        // points assume Float32 scale/shift; other weight datatypes go through
        // the _v2 entry points (oneDNN >= 3.3).
        dnnl_status_t lnForwardPd(Pd& pd, dnnl_engine_t e, dnnl_prop_kind_t prop, const Md& src, const Md& dst, const Md& stat, DataType scale_type, double eps, unsigned flags) {
            if (scale_type == DataType::Float32)
                return dnn_dyn::p_dnnl_layer_normalization_forward_primitive_desc_create( &pd.v, e, prop, src.v, dst.v, stat.v, (float) eps, flags, nullptr);
            if (!dnn_dyn::p_dnnl_layer_normalization_forward_primitive_desc_create_v2)
                throw DnnUnsupportedError("CLAP_DNN oneDNN: Float16/BFloat16 normalization weights require dnnl_layer_normalization_*_v2 (oneDNN >= 3.3)");
            return dnn_dyn::p_dnnl_layer_normalization_forward_primitive_desc_create_v2( &pd.v, e, prop, src.v, dst.v, stat.v, dnnlType(scale_type), (float) eps, flags, nullptr);
        }
        dnnl_status_t lnBackwardPd(Pd& pd, dnnl_engine_t e, const Md& diff_src, const Md& diff_dst, const Md& src, const Md& stat, DataType scale_type, double eps, unsigned flags, const Pd& hint) {
            if (scale_type == DataType::Float32)
                return dnn_dyn::p_dnnl_layer_normalization_backward_primitive_desc_create( &pd.v, e, dnnl_backward, diff_src.v, diff_dst.v, src.v, stat.v, (float) eps, flags, hint.v, nullptr);
            if (!dnn_dyn::p_dnnl_layer_normalization_backward_primitive_desc_create_v2)
                throw DnnUnsupportedError("CLAP_DNN oneDNN: Float16/BFloat16 normalization weights require dnnl_layer_normalization_*_v2 (oneDNN >= 3.3)");
            return dnn_dyn::p_dnnl_layer_normalization_backward_primitive_desc_create_v2( &pd.v, e, dnnl_backward, diff_src.v, diff_dst.v, src.v, stat.v, dnnlType(scale_type), dnnlType(scale_type), (float) eps, flags, hint.v, nullptr);
        }

        // Conversions between the oneDNN normalization statistic (variance
        // or mean square) and the CLAP saved_rstd convention, expressed as
        // oneDNN eltwise primitives executed in order on the call stream:
        //   rstd     = (variance + eps)^-0.5   linear(1, eps)  then pow(1, -0.5)
        //   variance = rstd^-2 - eps           pow(1, -2)      then linear(1, -eps)
        void eltwiseF32(Cpu& q, dnnl_alg_kind_t alg, float alpha, float beta, std::int64_t n, const void* src, void* dst, const char* what) {
            Md m(dnnl_f32, {n}, {1});
            Mem S(m, q.e, mut(src)), D(m, q.e, dst);
            Pd pd;
            check(dnn_dyn::p_dnnl_eltwise_forward_primitive_desc_create( &pd.v, q.e, dnnl_forward_inference, alg, m.v, m.v, alpha, beta, nullptr), what);
            Prim p(pd);
            exec(p, q, {
                {DNNL_ARG_SRC, S.v},
                {DNNL_ARG_DST, D.v}
            });
        }
        void varianceToRstd(Cpu& q, std::int64_t rows, float* stat, double eps) {
            eltwiseF32(q, dnnl_eltwise_linear, 1.0f, (float) eps, rows, stat, stat, "dnnl_eltwise_forward_primitive_desc_create(variance + eps)");
            eltwiseF32(q, dnnl_eltwise_pow, 1.0f, -0.5f, rows, stat, stat, "dnnl_eltwise_forward_primitive_desc_create(rsqrt)");
        }
        void rstdToVariance(Cpu& q, std::int64_t rows, const void* rstd, float* variance, double eps) {
            eltwiseF32(q, dnnl_eltwise_pow, 1.0f, -2.0f, rows, rstd, variance, "dnnl_eltwise_forward_primitive_desc_create(rstd^-2)");
            eltwiseF32(q, dnnl_eltwise_linear, 1.0f, (float) -eps, rows, variance, variance, "dnnl_eltwise_forward_primitive_desc_create(- eps)");
        }
        // Copies a Float32 vector with an eltwise linear(1, 0) primitive so the
        // copy is ordered on the same oneDNN stream as the producer.
        void copyF32(Cpu& q, std::int64_t n, const void* src, void* dst) {
            eltwiseF32(q, dnnl_eltwise_linear, 1.0f, 0.0f, n, src, dst, "dnnl_eltwise_forward_primitive_desc_create(copy)");
        }
        void conv_fwd(Cpu& q, const TensorDesc& xd, const void* x, const TensorDesc& wd, const void* w, const ConvolutionDesc& c, const TensorDesc& yd, void* y, const TensorDesc* bd = nullptr, const void* bias = nullptr, dnnl_primitive_attr_t attr = nullptr) {
            Md xm(xd), wm(wd), ym(yd);
            std::unique_ptr<Md> bm;
            if (bd) bm.reset(new Md( *bd));
            Mem X(xm, q.e, mut(x)), W(wm, q.e, mut(w)), Y(ym, q.e, y);
            std::unique_ptr<Mem> B;
            if (bm) B.reset(new Mem( *bm, q.e, mut(bias)));
            Pd pd;
            ConvGeometry g(c);
            checkPd(dnn_dyn::p_dnnl_convolution_forward_primitive_desc_create( &pd.v, q.e, dnnl_forward_inference, dnnl_convolution_direct, xm.v, wm.v, bm ? bm->v : nullptr, ym.v, g.st, g.dil, g.pl, g.pr, attr), "dnnl_convolution_forward_primitive_desc_create");
            Prim p(pd);
            if (B) {
                exec(p, q, {
                    {DNNL_ARG_SRC, X.v},
                    {DNNL_ARG_WEIGHTS, W.v},
                    {DNNL_ARG_BIAS, B->v},
                    {DNNL_ARG_DST, Y.v}
                });
            } else {
                exec(p, q, {
                    {
                        DNNL_ARG_SRC, X.v
                    }, {
                        DNNL_ARG_WEIGHTS, W.v
                    }, {
                        DNNL_ARG_DST, Y.v
                    }
                });
            }
        }
    }

    // ------------------------------------------------------------------
    // Backend-owned oneDNN state.
    // ------------------------------------------------------------------
    namespace onednn_detail {
        // A compiled oneDNN Graph SDPA partition.  inputs holds the logical
        // tensors in compile/execute order; slot_* map them back to CLAP
        // arguments.
        struct CompiledSdpa {
            dnnl_graph_compiled_partition_t cp = nullptr;
            std::vector<dnnl_graph_logical_tensor_t> inputs;
            dnnl_graph_logical_tensor_t output {};
            int slot_q = -1;
            int slot_k = -1;
            int slot_v = -1;
            int slot_scale = -1;
            int slot_mask = -1;
            int slot_neg_inf = -1;
            ~CompiledSdpa() {
                if (cp) dnn_dyn::p_dnnl_graph_compiled_partition_destroy(cp);
            }
        };
    }

    struct OneDnnBackend::NativeState {
        dnnl_engine_t engine = nullptr;
        std::mutex sdpa_mutex;
        std::map<std::string, std::unique_ptr<onednn_detail::CompiledSdpa>> sdpa;

        ~NativeState() {
            sdpa.clear();
            if (engine) dnn_dyn::p_dnnl_engine_destroy(engine);
        }
    };

    OneDnnBackend::OneDnnBackend() {
        if (!dnn_dyn::loadOneDnn()) throw std::runtime_error(std::string("oneDNN runtime load failed: ") + dnn_dyn::lastError());
        native_.reset(new NativeState());
        check(dnn_dyn::p_dnnl_engine_create( &native_->engine, dnnl_cpu, 0), "dnnl_engine_create");
    }
    OneDnnBackend::~OneDnnBackend() = default;

    const char* OneDnnBackend::name() const noexcept {
        return "oneDNN";
    }

    ExecutionBackend OneDnnBackend::executionBackend() const noexcept {
        return ExecutionBackend::CPU;
    }

    // ------------------------------------------------------------------
    // Historical host-pointer API.  oneDNN executes on host memory, so these
    // forward directly to the framework contract with a CPU context.
    // ------------------------------------------------------------------

    void OneDnnBackend::convolutionForward(const TensorDesc& xd, const float* x, const TensorDesc& wd, const float* w, const ConvolutionDesc& c, const TensorDesc& yd, float* y) {
        std::cerr<<"[MY CLAP] CPU convolution called through oneDNN function pointers\n";
        convolutionForward(xd, static_cast<const void*>(x), wd, static_cast<const void*>(w), c, yd, static_cast<void*>(y), ExecutionContext::cpu());
    }

    void OneDnnBackend::convolutionBackwardData(const TensorDesc& dyd, const float* dy, const TensorDesc& wd, const float* w, const ConvolutionDesc& c, const TensorDesc& dxd, float* dx) {
        convolutionBackwardData(dyd, static_cast<const void*>(dy), wd, static_cast<const void*>(w), c, dxd, static_cast<void*>(dx), ExecutionContext::cpu());
    }

    void OneDnnBackend::convolutionBackwardWeights(const TensorDesc& xd, const float* x, const TensorDesc& dyd, const float* dy, const ConvolutionDesc& c, const TensorDesc& dwd, float* dw) {
        convolutionBackwardWeights(xd, static_cast<const void*>(x), dyd, static_cast<const void*>(dy), c, dwd, static_cast<void*>(dw), ExecutionContext::cpu());
    }

    void OneDnnBackend::activationForward(const ActivationDesc& a, const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y) {
        activationForward(a, xd, static_cast<const void*>(x), yd, static_cast<void*>(y), ExecutionContext::cpu());
    }

    void OneDnnBackend::activationBackward(const ActivationDesc& a, const TensorDesc& xd, const float* x, const TensorDesc& dyd, const float* dy, const TensorDesc& dxd, float* dx) {
        activationBackward(a, xd, static_cast<const void*>(x), dyd, static_cast<const void*>(dy), dxd, static_cast<void*>(dx), ExecutionContext::cpu());
    }

    void OneDnnBackend::poolingForward(const PoolingDesc& p, const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y) {
        poolingForward(p, xd, static_cast<const void*>(x), yd, static_cast<void*>(y), ExecutionContext::cpu());
    }

    void OneDnnBackend::poolingBackward(const PoolingDesc& p, const TensorDesc& xd, const float* x, const TensorDesc& yd, const float* y, const TensorDesc& dyd, const float* dy, const TensorDesc& dxd, float* dx) {
        poolingBackward(p, xd, static_cast<const void*>(x), yd, static_cast<const void*>(y), dyd, static_cast<const void*>(dy), dxd, static_cast<void*>(dx), ExecutionContext::cpu());
    }

    // Historical softmax: normalization over dimension 1 (channels).
    void OneDnnBackend::softmaxForward(const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y) {
        SoftmaxDesc softmax;
        softmax.axis = 1;
        softmaxForward(softmax, xd, static_cast<const void*>(x), yd, static_cast<void*>(y), ExecutionContext::cpu());
    }

    void OneDnnBackend::softmaxBackward(const TensorDesc& yd, const float* y, const TensorDesc& dyd, const float* dy, const TensorDesc& dxd, float* dx) {
        SoftmaxDesc softmax;
        softmax.axis = 1;
        softmaxBackward(softmax, yd, static_cast<const void*>(y), dyd, static_cast<const void*>(dy), dxd, static_cast<void*>(dx), ExecutionContext::cpu());
    }

    void OneDnnBackend::batchNormForwardTraining(const BatchNormDesc& bn, const TensorDesc& xd, const float* x, const float* scale, const float* bias, float* rm, float* rv, float* sm, float* sv, const TensorDesc& yd, float* y) {
        batchNormForwardTraining(bn, xd, static_cast<const void*>(x), scale, bias, rm, rv, sm, sv, yd, static_cast<void*>(y), ExecutionContext::cpu());
    }

    void OneDnnBackend::batchNormForwardInference(const BatchNormDesc& bn, const TensorDesc& xd, const float* x, const float* scale, const float* bias, const float* mean, const float* var, const TensorDesc& yd, float* y) {
        batchNormForwardInference(bn, xd, static_cast<const void*>(x), scale, bias, mean, var, yd, static_cast<void*>(y), ExecutionContext::cpu());
    }

    void OneDnnBackend::batchNormBackward(const BatchNormDesc& bn, const TensorDesc& xd, const float* x, const TensorDesc& dyd, const float* dy, const float* scale, const float* mean, const float* var, const TensorDesc& dxd, float* dx, float* ds, float* db) {
        batchNormBackward(bn, xd, static_cast<const void*>(x), dyd, static_cast<const void*>(dy), scale, mean, var, dxd, static_cast<void*>(dx), ds, db, ExecutionContext::cpu());
    }

    void OneDnnBackend::convolutionBackwardBias(const TensorDesc& dyd, const float* dy, const TensorDesc& dbd, float* db) {
        convolutionBackwardBias(dyd, static_cast<const void*>(dy), dbd, static_cast<void*>(db), ExecutionContext::cpu());
    }

    void OneDnnBackend::convolutionTransposeForward(const TensorDesc& xd, const float* x, const TensorDesc& wd, const float* w, const ConvolutionDesc& c, const TensorDesc& yd, float* y) {
        convolutionTransposeForward(xd, static_cast<const void*>(x), wd, static_cast<const void*>(w), c, yd, static_cast<void*>(y), ExecutionContext::cpu());
    }

    void OneDnnBackend::fusedConvolutionBiasActivation(const TensorDesc& xd, const float* x, const TensorDesc& wd, const float* w, const float* bias, const ConvolutionDesc& c, const ActivationDesc& a, const TensorDesc& yd, float* y) {
        fusedConvolutionBiasActivation(xd, static_cast<const void*>(x), wd, static_cast<const void*>(w), static_cast<const void*>(bias), c, a, yd, static_cast<void*>(y), ExecutionContext::cpu());
    }

    void OneDnnBackend::layerNormForward(const LayerNormDesc& ln, const TensorDesc& xd, const float* x, const float* scale, const float* bias, const TensorDesc& yd, float* y, float* mean, float* rstd) {
        TensorDesc sd( {
            xd.dims.back()
        });
        layerNormForward(ln, xd, static_cast<const void*>(x), sd, scale, bias, yd, static_cast<void*>(y), mean, rstd, ExecutionContext::cpu());
    }

    void OneDnnBackend::layerNormBackward(const LayerNormDesc& ln, const TensorDesc& xd, const float* x, const TensorDesc& dyd, const float* dy, const float* scale, const float* mean, const float* rstd, const TensorDesc& dxd, float* dx, float* dscale, float* dbias) {
        TensorDesc sd( {
            xd.dims.back()
        });
        layerNormBackward(ln, xd, static_cast<const void*>(x), dyd, static_cast<const void*>(dy), sd, scale, mean, rstd, dxd, static_cast<void*>(dx), dscale, dbias, ExecutionContext::cpu());
    }

    void OneDnnBackend::dropoutForward(const DropoutDesc& , const TensorDesc& , const float* , const TensorDesc& , float* , std::uint8_t* ) {
        throw DnnUnsupportedError("oneDNN dropout requires a graph/attribute execution path; reference kernel removed");
    }

    void OneDnnBackend::dropoutBackward(const DropoutDesc& , const TensorDesc& , const float* , const std::uint8_t* , const TensorDesc& , float* ) {
        throw DnnUnsupportedError("oneDNN dropout requires a graph/attribute execution path; reference kernel removed");
    }
void OneDnnBackend::lrnForward(const LrnDesc& lrn, const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y)
{
    lrnForward(lrn, xd, static_cast<const void*>(x), yd, static_cast<void*>(y), ExecutionContext::cpu());
}

void OneDnnBackend::lrnBackward(const LrnDesc& lrn, const TensorDesc& xd, const float* x, const TensorDesc& yd, const float* y, const TensorDesc& dyd, const float* dy, const TensorDesc& dxd, float* dx)
{
    lrnBackward(
        lrn,
        xd, static_cast<const void*>(x),
        yd, static_cast<const void*>(y),
        dyd, static_cast<const void*>(dy),
        dxd, static_cast<void*>(dx),
        ExecutionContext::cpu());
}

void OneDnnBackend::tensorAdd(const TensorDesc& ad, const float* a, const TensorDesc& bd, const float* b, const TensorDesc& yd, float* y)
{
    tensorAdd(ad, static_cast<const void*>(a), bd, static_cast<const void*>(b), yd, static_cast<void*>(y), ExecutionContext::cpu());
}

void OneDnnBackend::tensorMultiply(const TensorDesc& ad, const float* a, const TensorDesc& bd, const float* b, const TensorDesc& yd, float* y)
{
    tensorMultiply(ad, static_cast<const void*>(a), bd, static_cast<const void*>(b), yd, static_cast<void*>(y), ExecutionContext::cpu());
}

void OneDnnBackend::tensorMin(const TensorDesc& ad, const float* a, const TensorDesc& bd, const float* b, const TensorDesc& yd, float* y)
{
    tensorMin(ad, static_cast<const void*>(a), bd, static_cast<const void*>(b), yd, static_cast<void*>(y), ExecutionContext::cpu());
}

void OneDnnBackend::tensorMax(const TensorDesc& ad, const float* a, const TensorDesc& bd, const float* b, const TensorDesc& yd, float* y)
{
    tensorMax(ad, static_cast<const void*>(a), bd, static_cast<const void*>(b), yd, static_cast<void*>(y), ExecutionContext::cpu());
}

void OneDnnBackend::tensorReduceSum(const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y)
{
    tensorReduceSum(xd, static_cast<const void*>(x), yd, static_cast<void*>(y), ExecutionContext::cpu());
}

void OneDnnBackend::tensorReduceProduct(const TensorDesc& xd, const float* x, const TensorDesc& yd, float* y)
{
    tensorReduceProduct(xd, static_cast<const void*>(x), yd, static_cast<void*>(y), ExecutionContext::cpu());
}

// ==========================================================================
// Framework execution contract (host pointers, optional caller dnnl_stream_t)
// ==========================================================================

void OneDnnBackend::convolutionForward(const TensorDesc& xd, const void* x, const TensorDesc& wd, const void* w, const ConvolutionDesc& c, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    dnn_contract::trace("CPU", "convolutionForward", "dnnl convolution forward");
    conv_fwd(q, xd, x, wd, w, c, yd, y);
}

void OneDnnBackend::convolutionBackwardData(const TensorDesc& dyd, const void* dy, const TensorDesc& wd, const void* w, const ConvolutionDesc& c, const TensorDesc& dxd, void* dx, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    Md dym(dyd), wm(wd), dxm(dxd);
    Mem DY(dym, q.e, mut(dy)), W(wm, q.e, mut(w)), DX(dxm, q.e, dx);
    ConvGeometry g(c);

    Pd hint;
    checkPd(
        dnn_dyn::p_dnnl_convolution_forward_primitive_desc_create(
            &hint.v, q.e, dnnl_forward_training, dnnl_convolution_direct,
            dxm.v, wm.v, nullptr, dym.v,
            g.st, g.dil, g.pl, g.pr, nullptr),
        "dnnl_convolution_forward_primitive_desc_create(hint)");

    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_convolution_backward_data_primitive_desc_create(
            &pd.v, q.e, dnnl_convolution_direct,
            dxm.v, wm.v, dym.v,
            g.st, g.dil, g.pl, g.pr, hint.v, nullptr),
        "dnnl_convolution_backward_data_primitive_desc_create");

    Prim p(pd);
    exec(p, q, {
        {DNNL_ARG_DIFF_DST, DY.v},
        {DNNL_ARG_WEIGHTS, W.v},
        {DNNL_ARG_DIFF_SRC, DX.v}
    });
}

void OneDnnBackend::convolutionBackwardWeights(const TensorDesc& xd, const void* x, const TensorDesc& dyd, const void* dy, const ConvolutionDesc& c, const TensorDesc& dwd, void* dw, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    Md xm(xd), dym(dyd), dwm(dwd);
    Mem X(xm, q.e, mut(x)), DY(dym, q.e, mut(dy)), DW(dwm, q.e, dw);
    ConvGeometry g(c);

    Pd hint;
    checkPd(
        dnn_dyn::p_dnnl_convolution_forward_primitive_desc_create(
            &hint.v, q.e, dnnl_forward_training, dnnl_convolution_direct,
            xm.v, dwm.v, nullptr, dym.v,
            g.st, g.dil, g.pl, g.pr, nullptr),
        "dnnl_convolution_forward_primitive_desc_create(hint)");

    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_convolution_backward_weights_primitive_desc_create(
            &pd.v, q.e, dnnl_convolution_direct,
            xm.v, dwm.v, nullptr, dym.v,
            g.st, g.dil, g.pl, g.pr, hint.v, nullptr),
        "dnnl_convolution_backward_weights_primitive_desc_create");

    Prim p(pd);
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DIFF_DST, DY.v},
        {DNNL_ARG_DIFF_WEIGHTS, DW.v}
    });
}

// Bias gradient = sum of dy over every dimension except channels, computed
// with the oneDNN reduction primitive.
void OneDnnBackend::convolutionBackwardBias(const TensorDesc& dyd, const void* dy, const TensorDesc& dbd, void* db, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);

    if (dyd.rank() < 2)
        throw std::runtime_error("convolutionBackwardBias: dy must have a channel dimension");
    if (dbd.elements() != static_cast<std::size_t>(dyd.dims[1]))
        throw std::runtime_error("convolutionBackwardBias: db must hold one value per channel");

    std::vector<dnnl_dim_t> bdims(dyd.dims.size(), 1);
    bdims[1] = dyd.dims[1];

    Md dym(dyd);
    Md dbm(dnnlType(dbd.type), bdims, packed(bdims));
    Mem DY(dym, q.e, mut(dy)), DB(dbm, q.e, db);

    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_reduction_primitive_desc_create(
            &pd.v, q.e, dnnl_reduction_sum, dym.v, dbm.v, 0.0f, 0.0f, nullptr),
        "dnnl_reduction_primitive_desc_create(bias gradient)");

    Prim p(pd);
    dnn_contract::trace("CPU", "convolutionBackwardBias", "dnnl reduction sum");
    exec(p, q, {
        {DNNL_ARG_SRC, DY.v},
        {DNNL_ARG_DST, DB.v}
    });
}

void OneDnnBackend::convolutionTransposeForward(const TensorDesc& xd, const void* x, const TensorDesc& wd, const void* w, const ConvolutionDesc& c, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    Md xm(xd), wm(wd), ym(yd);
    Mem X(xm, q.e, mut(x)), W(wm, q.e, mut(w)), Y(ym, q.e, y);
    ConvGeometry g(c);

    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_deconvolution_forward_primitive_desc_create(
            &pd.v, q.e, dnnl_forward_inference, dnnl_deconvolution_direct,
            xm.v, wm.v, nullptr, ym.v,
            g.st, g.dil, g.pl, g.pr, nullptr),
        "dnnl_deconvolution_forward_primitive_desc_create");

    Prim p(pd);
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_WEIGHTS, W.v},
        {DNNL_ARG_DST, Y.v}
    });
}

void OneDnnBackend::fusedConvolutionBiasActivation(const TensorDesc& xd, const void* x, const TensorDesc& wd, const void* w, const void* bias, const ConvolutionDesc& c, const ActivationDesc& a, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    TensorDesc bd(yd.type, {yd.dims.at(1)});

    Attr attr;
    attr.appendEltwise(act_alg(a.mode), act_alpha(a), 0.0f);

    dnn_contract::trace("CPU", "fusedConvolutionBiasActivation", "dnnl convolution + eltwise post-op");
    conv_fwd(q, xd, x, wd, w, c, yd, y, &bd, bias, attr.v);
}

void OneDnnBackend::activationForward(const ActivationDesc& a, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    Md xm(xd), ym(yd);
    Mem X(xm, q.e, mut(x)), Y(ym, q.e, y);

    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_eltwise_forward_primitive_desc_create(
            &pd.v, q.e, dnnl_forward_inference, act_alg(a.mode),
            xm.v, ym.v, act_alpha(a), 0.0f, nullptr),
        "dnnl_eltwise_forward_primitive_desc_create");

    Prim p(pd);
    dnn_contract::trace("CPU", "activationForward", "dnnl eltwise forward");
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DST, Y.v}
    });
}

void OneDnnBackend::activationBackward(const ActivationDesc& a, const TensorDesc& xd, const void* x, const TensorDesc& dyd, const void* dy, const TensorDesc& dxd, void* dx, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    Md xm(xd), dym(dyd), dxm(dxd);
    Mem X(xm, q.e, mut(x)), DY(dym, q.e, mut(dy)), DX(dxm, q.e, dx);

    Pd hint;
    checkPd(
        dnn_dyn::p_dnnl_eltwise_forward_primitive_desc_create(
            &hint.v, q.e, dnnl_forward_training, act_alg(a.mode),
            xm.v, xm.v, act_alpha(a), 0.0f, nullptr),
        "dnnl_eltwise_forward_primitive_desc_create(hint)");

    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_eltwise_backward_primitive_desc_create(
            &pd.v, q.e, act_alg(a.mode),
            dxm.v, dym.v, xm.v,
            act_alpha(a), 0.0f, hint.v, nullptr),
        "dnnl_eltwise_backward_primitive_desc_create");

    Prim p(pd);
    dnn_contract::trace("CPU", "activationBackward", "dnnl eltwise backward");
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DIFF_DST, DY.v},
        {DNNL_ARG_DIFF_SRC, DX.v}
    });
}

void OneDnnBackend::poolingForward(const PoolingDesc& pool, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    Md xm(xd), ym(yd);
    Mem X(xm, q.e, mut(x)), Y(ym, q.e, y);
    PoolGeometry g(pool);

    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_pooling_forward_primitive_desc_create(
            &pd.v, q.e, dnnl_forward_inference, pool_alg(pool.mode),
            xm.v, ym.v, g.st, g.k, g.dil, g.pl, g.pr, nullptr),
        "dnnl_pooling_forward_primitive_desc_create");

    Prim p(pd);
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DST, Y.v}
    });
}


// Max pooling backward needs the forward workspace (argmax positions).  It is
// regenerated by running the oneDNN forward-training primitive on x into host
// scratch memory owned by this call.
void OneDnnBackend::poolingBackward(const PoolingDesc& pool, const TensorDesc& xd, const void* x, const TensorDesc& yd, const void* , const TensorDesc& dyd, const void* dy, const TensorDesc& dxd, void* dx, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    Md xm(xd), ym(yd), dym(dyd), dxm(dxd);
    Mem X(xm, q.e, mut(x)), DY(dym, q.e, mut(dy)), DX(dxm, q.e, dx);
    PoolGeometry g(pool);

    Pd hint;
    checkPd(
        dnn_dyn::p_dnnl_pooling_forward_primitive_desc_create(
            &hint.v, q.e, dnnl_forward_training, pool_alg(pool.mode),
            xm.v, ym.v, g.st, g.k, g.dil, g.pl, g.pr, nullptr),
        "dnnl_pooling_forward_primitive_desc_create(hint)");

    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_pooling_backward_primitive_desc_create(
            &pd.v, q.e, pool_alg(pool.mode),
            dxm.v, dym.v, g.st, g.k, g.dil, g.pl, g.pr, hint.v, nullptr),
        "dnnl_pooling_backward_primitive_desc_create");

    const std::size_t ws_bytes = workspaceBytes(hint);

    if (ws_bytes == 0) {
        Prim p(pd);
        exec(p, q, {
            {DNNL_ARG_DIFF_DST, DY.v},
            {DNNL_ARG_DIFF_SRC, DX.v}
        });
        return;
    }

    std::vector<std::uint8_t> ws(ws_bytes);
    std::vector<std::uint8_t> y_scratch(yd.bytes());
    const_dnnl_memory_desc_t ws_md = dnn_dyn::p_dnnl_primitive_desc_query_md(hint.v, dnnl_query_workspace_md, 0);
    Mem WS(ws_md, q.e, ws.data());
    Mem YS(ym, q.e, y_scratch.data());

    Prim fwd(hint);
    exec(fwd, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DST, YS.v},
        {DNNL_ARG_WORKSPACE, WS.v}
    });

    Prim bwd(pd);
    exec(bwd, q, {
        {DNNL_ARG_DIFF_DST, DY.v},
        {DNNL_ARG_WORKSPACE, WS.v},
        {DNNL_ARG_DIFF_SRC, DX.v}
    });

    q.complete();
}

void OneDnnBackend::softmaxForward(const SoftmaxDesc& softmax, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    const int axis = xd.normalizeAxis(softmax.axis, "softmaxForward");
    Md xm(xd), ym(yd);
    Mem X(xm, q.e, mut(x)), Y(ym, q.e, y);

    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_softmax_forward_primitive_desc_create(
            &pd.v, q.e, dnnl_forward_inference,
            softmax.log_softmax ? dnnl_softmax_log : dnnl_softmax_accurate,
            xm.v, ym.v, axis, nullptr),
        "dnnl_softmax_forward_primitive_desc_create");

    Prim p(pd);
    dnn_contract::trace("CPU", softmax.log_softmax ? "logSoftmaxForward" : "softmaxForward", "dnnl softmax forward");
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DST, Y.v}
    });
}

void OneDnnBackend::softmaxBackward(const SoftmaxDesc& softmax, const TensorDesc& yd, const void* y, const TensorDesc& dyd, const void* dy, const TensorDesc& dxd, void* dx, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    const int axis = yd.normalizeAxis(softmax.axis, "softmaxBackward");
    const dnnl_alg_kind_t alg = softmax.log_softmax ? dnnl_softmax_log : dnnl_softmax_accurate;
    Md ym(yd), dym(dyd), dxm(dxd);
    Mem Y(ym, q.e, mut(y)), DY(dym, q.e, mut(dy)), DX(dxm, q.e, dx);

    Pd hint;
    checkPd(
        dnn_dyn::p_dnnl_softmax_forward_primitive_desc_create(
            &hint.v, q.e, dnnl_forward_training, alg,
            dxm.v, ym.v, axis, nullptr),
        "dnnl_softmax_forward_primitive_desc_create(hint)");

    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_softmax_backward_primitive_desc_create(
            &pd.v, q.e, alg, dxm.v, dym.v, ym.v, axis, hint.v, nullptr),
        "dnnl_softmax_backward_primitive_desc_create");

    Prim p(pd);
    dnn_contract::trace("CPU", softmax.log_softmax ? "logSoftmaxBackward" : "softmaxBackward", "dnnl softmax backward");
    exec(p, q, {
        {DNNL_ARG_DST, Y.v},
        {DNNL_ARG_DIFF_DST, DY.v},
        {DNNL_ARG_DIFF_SRC, DX.v}
    });
}

// Batch normalization statistics on oneDNN are Float32 [C] vectors and
// saved_variance holds the batch variance.
void OneDnnBackend::batchNormForwardTraining(const BatchNormDesc& bn, const TensorDesc& xd, const void* x, const void* scale, const void* bias, void* rm, void* rv, void* sm, void* sv, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    const dnnl_dim_t c = xd.dims.at(1);
    Md xm(xd), ym(yd), cm(dnnl_f32, {c}, {1});

    std::vector<float> mean_scratch, var_scratch;
    if (!sm) {
        mean_scratch.resize(static_cast<std::size_t>(c));
        sm = mean_scratch.data();
    }
    if (!sv) {
        var_scratch.resize(static_cast<std::size_t>(c));
        sv = var_scratch.data();
    }

    Mem X(xm, q.e, mut(x)), Y(ym, q.e, y), S(cm, q.e, mut(scale)), B(cm, q.e, mut(bias)), M(cm, q.e, sm), V(cm, q.e, sv);

    Pd pd;
    unsigned flags = dnnl_use_scale | dnnl_use_shift;
    checkPd(
        dnn_dyn::p_dnnl_batch_normalization_forward_primitive_desc_create(
            &pd.v, q.e, dnnl_forward_training, xm.v, ym.v, (float) bn.epsilon, flags, nullptr),
        "dnnl_batch_normalization_forward_primitive_desc_create");

    Prim p(pd);
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_SCALE, S.v},
        {DNNL_ARG_SHIFT, B.v},
        {DNNL_ARG_MEAN, M.v},
        {DNNL_ARG_VARIANCE, V.v},
        {DNNL_ARG_DST, Y.v}
    });

    // Historical CLAP behavior: running statistics receive the batch
    // statistics (exponential average factor 1.0, as in the cuDNN backend).
    if (rm) copyF32(q, c, sm, rm);
    if (rv) copyF32(q, c, sv, rv);

    if (!mean_scratch.empty() || !var_scratch.empty())
        q.complete();
}

void OneDnnBackend::batchNormForwardInference(const BatchNormDesc& bn, const TensorDesc& xd, const void* x, const void* scale, const void* bias, const void* mean, const void* var, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    const dnnl_dim_t c = xd.dims.at(1);
    Md xm(xd), ym(yd), cm(dnnl_f32, {c}, {1});
    Mem X(xm, q.e, mut(x)), Y(ym, q.e, y), S(cm, q.e, mut(scale)), B(cm, q.e, mut(bias)), M(cm, q.e, mut(mean)), V(cm, q.e, mut(var));

    Pd pd;
    unsigned flags = dnnl_use_global_stats | dnnl_use_scale | dnnl_use_shift;
    checkPd(
        dnn_dyn::p_dnnl_batch_normalization_forward_primitive_desc_create(
            &pd.v, q.e, dnnl_forward_inference, xm.v, ym.v, (float) bn.epsilon, flags, nullptr),
        "dnnl_batch_normalization_forward_primitive_desc_create");

    Prim p(pd);
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_SCALE, S.v},
        {DNNL_ARG_SHIFT, B.v},
        {DNNL_ARG_MEAN, M.v},
        {DNNL_ARG_VARIANCE, V.v},
        {DNNL_ARG_DST, Y.v}
    });
}

void OneDnnBackend::batchNormBackward(const BatchNormDesc& bn, const TensorDesc& xd, const void* x, const TensorDesc& dyd, const void* dy, const void* scale, const void* mean, const void* var, const TensorDesc& dxd, void* dx, void* ds, void* db, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    const dnnl_dim_t c = xd.dims.at(1);
    Md xm(xd), dym(dyd), dxm(dxd), cm(dnnl_f32, {c}, {1});

    std::vector<float> ds_scratch, db_scratch;
    if (!ds) {
        ds_scratch.resize(static_cast<std::size_t>(c));
        ds = ds_scratch.data();
    }
    if (!db) {
        db_scratch.resize(static_cast<std::size_t>(c));
        db = db_scratch.data();
    }

    Mem X(xm, q.e, mut(x)), DY(dym, q.e, mut(dy)), DX(dxm, q.e, dx), S(cm, q.e, mut(scale)), M(cm, q.e, mut(mean)), V(cm, q.e, mut(var)), DS(cm, q.e, ds), DB(cm, q.e, db);
    unsigned flags = dnnl_use_scale | dnnl_use_shift;

    Pd hint;
    checkPd(
        dnn_dyn::p_dnnl_batch_normalization_forward_primitive_desc_create(
            &hint.v, q.e, dnnl_forward_training, xm.v, xm.v, (float) bn.epsilon, flags, nullptr),
        "dnnl_batch_normalization_forward_primitive_desc_create(hint)");

    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_batch_normalization_backward_primitive_desc_create(
            &pd.v, q.e, dnnl_backward, dxm.v, dym.v, xm.v, (float) bn.epsilon, flags, hint.v, nullptr),
        "dnnl_batch_normalization_backward_primitive_desc_create");

    Prim p(pd);
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DIFF_DST, DY.v},
        {DNNL_ARG_SCALE, S.v},
        {DNNL_ARG_MEAN, M.v},
        {DNNL_ARG_VARIANCE, V.v},
        {DNNL_ARG_DIFF_SRC, DX.v},
        {DNNL_ARG_DIFF_SCALE, DS.v},
        {DNNL_ARG_DIFF_SHIFT, DB.v}
    });

    if (!ds_scratch.empty() || !db_scratch.empty())
        q.complete();
}

// LayerNorm over the last dimension (oneDNN layer normalization primitive).
// saved_rstd follows the CLAP convention 1/sqrt(var + eps); oneDNN produces
// the variance, which is converted in place by a oneDNN eltwise primitive.
void OneDnnBackend::layerNormForward(const LayerNormDesc& ln, const TensorDesc& xd, const void* x, const TensorDesc& sd, const void* scale, const void* bias, const TensorDesc& yd, void* y, void* saved_mean, void* saved_rstd, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    dnn_contract::requireSameShape(xd, yd, "layerNormForward");
    RowsMd xr(xd, "layerNormForward"), yr(yd, "layerNormForward");
    auto sm = vectorMd(sd, xr.view.inner, "layerNormForward");
    Mem X( *xr.data, q.e, mut(x)), Y( *yr.data, q.e, y), S( *sm, q.e, mut(scale)), B( *sm, q.e, mut(bias));

    const bool training = saved_mean || saved_rstd;
    unsigned flags = dnnl_normalization_use_scale | dnnl_normalization_use_shift;

    Pd pd;
    checkPd(
        lnForwardPd(
            pd, q.e, training ? dnnl_forward_training : dnnl_forward_inference,
            *xr.data, *yr.data, *xr.stats, sd.type, ln.epsilon, flags),
        "dnnl_layer_normalization_forward_primitive_desc_create");
    Prim p(pd);
    dnn_contract::trace("CPU", "layerNormForward", "dnnl layer normalization forward");

    if (!training) {
        exec(p, q, {
            {DNNL_ARG_SRC, X.v},
            {DNNL_ARG_SCALE, S.v},
            {DNNL_ARG_SHIFT, B.v},
            {DNNL_ARG_DST, Y.v}
        });
        return;
    }

    std::vector<float> mean_scratch, var_scratch;
    float* mean_ptr = static_cast<float*>(saved_mean);
    float* var_ptr = static_cast<float*>(saved_rstd);
    if (!mean_ptr) {
        mean_scratch.resize(static_cast<std::size_t>(xr.view.rows));
        mean_ptr = mean_scratch.data();
    }
    if (!var_ptr) {
        var_scratch.resize(static_cast<std::size_t>(xr.view.rows));
        var_ptr = var_scratch.data();
    }
    Mem M( *xr.stats, q.e, mean_ptr), V( *xr.stats, q.e, var_ptr);

    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_SCALE, S.v},
        {DNNL_ARG_SHIFT, B.v},
        {DNNL_ARG_DST, Y.v},
        {DNNL_ARG_MEAN, M.v},
        {DNNL_ARG_VARIANCE, V.v}
    });

    if (saved_rstd)
        varianceToRstd(q, xr.view.rows, var_ptr, ln.epsilon);
    if (!mean_scratch.empty() || !var_scratch.empty())
        q.complete();
}

void OneDnnBackend::layerNormBackward(const LayerNormDesc& ln, const TensorDesc& xd, const void* x, const TensorDesc& dyd, const void* dy, const TensorDesc& sd, const void* scale, const void* saved_mean, const void* saved_rstd, const TensorDesc& dxd, void* dx, void* dscale, void* dbias, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    if (!saved_mean || !saved_rstd)
        throw std::runtime_error("layerNormBackward requires saved_mean and saved_rstd from layerNormForward");
    dnn_contract::requireSameShape(xd, dyd, "layerNormBackward");
    dnn_contract::requireSameShape(xd, dxd, "layerNormBackward");

    RowsMd xr(xd, "layerNormBackward"), dyr(dyd, "layerNormBackward"), dxr(dxd, "layerNormBackward");
    auto sm = vectorMd(sd, xr.view.inner, "layerNormBackward");
    const std::size_t inner_bytes = static_cast<std::size_t>(xr.view.inner) * sd.elementSize();

    std::vector<float> variance(static_cast<std::size_t>(xr.view.rows));
    std::vector<std::uint8_t> shift_zero(inner_bytes, 0), ds_scratch, db_scratch;
    if (!dscale) {
        ds_scratch.resize(inner_bytes);
        dscale = ds_scratch.data();
    }
    if (!dbias) {
        db_scratch.resize(inner_bytes);
        dbias = db_scratch.data();
    }

    rstdToVariance(q, xr.view.rows, saved_rstd, variance.data(), ln.epsilon);

    Mem X( *xr.data, q.e, mut(x)), DY( *dyr.data, q.e, mut(dy)), DX( *dxr.data, q.e, dx);
    Mem S( *sm, q.e, mut(scale)), B( *sm, q.e, shift_zero.data()), DS( *sm, q.e, dscale), DB( *sm, q.e, dbias);
    Mem M( *xr.stats, q.e, mut(saved_mean)), V( *xr.stats, q.e, variance.data());
    unsigned flags = dnnl_normalization_use_scale | dnnl_normalization_use_shift;

    Pd hint;
    checkPd(
        lnForwardPd(
            hint, q.e, dnnl_forward_training,
            *xr.data, *xr.data, *xr.stats, sd.type, ln.epsilon, flags),
        "dnnl_layer_normalization_forward_primitive_desc_create(hint)");

    Pd pd;
    checkPd(
        lnBackwardPd(
            pd, q.e, *dxr.data, *dyr.data, *xr.data, *xr.stats, sd.type, ln.epsilon, flags, hint),
        "dnnl_layer_normalization_backward_primitive_desc_create");

    Prim p(pd);
    dnn_contract::trace("CPU", "layerNormBackward", "dnnl layer normalization backward");
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DIFF_DST, DY.v},
        {DNNL_ARG_SCALE, S.v},
        {DNNL_ARG_SHIFT, B.v},
        {DNNL_ARG_MEAN, M.v},
        {DNNL_ARG_VARIANCE, V.v},
        {DNNL_ARG_DIFF_SRC, DX.v},
        {DNNL_ARG_DIFF_SCALE, DS.v},
        {DNNL_ARG_DIFF_SHIFT, DB.v}
    });

    q.complete();
}

// oneDNN 3.x CPU implementations of dnnl_rms_norm backward (ref and simple
// layer normalization, verified on oneDNN 3.11.4) subtract the LayerNorm
// mean-gradient term mean(dy * gamma) from diff_src, which RMSNorm does not
// have; the resulting gradient disagrees with finite differences.  CLAP_DNN
// therefore reports RMSNorm backward on oneDNN as unsupported instead of
// returning wrong gradients.  Once a corrected oneDNN is installed, set
// CLAP_DNN_ONEDNN_RMSNORM_BACKWARD=1 to use the native primitive.
static bool rmsBackwardTrusted()
{
    static const bool trusted = std::getenv("CLAP_DNN_ONEDNN_RMSNORM_BACKWARD") != nullptr;
    return trusted;
}

static std::string rmsBackwardDefect()
{
    return "CLAP_DNN oneDNN: dnnl_rms_norm backward includes the LayerNorm mean-gradient term "
           "(incorrect RMSNorm diff_src in oneDNN 3.x CPU implementations); "
           "set CLAP_DNN_ONEDNN_RMSNORM_BACKWARD=1 with a corrected oneDNN";
}

// RMSNorm through the oneDNN layer normalization primitive with the
// dnnl_rms_norm flag (oneDNN >= 3.4).  The statistic returned by oneDNN is the
// mean square, converted to saved_rstd = 1/sqrt(mean_square + eps).
void OneDnnBackend::rmsNormForward(const RmsNormDesc& norm, const TensorDesc& xd, const void* x, const TensorDesc& wd, const void* weight, const TensorDesc& yd, void* y, void* saved_rstd, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    dnn_contract::normalizeLastAxis(xd, norm.axis, "rmsNormForward");
    dnn_contract::requireSameShape(xd, yd, "rmsNormForward");
    RowsMd xr(xd, "rmsNormForward"), yr(yd, "rmsNormForward");
    auto wm = vectorMd(wd, xr.view.inner, "rmsNormForward");
    Mem X( *xr.data, q.e, mut(x)), Y( *yr.data, q.e, y), W( *wm, q.e, mut(weight));

    const bool training = saved_rstd != nullptr;
    unsigned flags = dnnl_rms_norm | dnnl_normalization_use_scale;

    Pd pd;
    checkPd(
        lnForwardPd(
            pd, q.e, training ? dnnl_forward_training : dnnl_forward_inference,
            *xr.data, *yr.data, *xr.stats, wd.type, norm.epsilon, flags),
        "dnnl_layer_normalization_forward_primitive_desc_create(rms_norm)");
    Prim p(pd);
    dnn_contract::trace("CPU", "rmsNormForward", "dnnl layer normalization forward (dnnl_rms_norm)");

    if (!training) {
        exec(p, q, {
            {DNNL_ARG_SRC, X.v},
            {DNNL_ARG_SCALE, W.v},
            {DNNL_ARG_DST, Y.v}
        });
        return;
    }

    Mem V( *xr.stats, q.e, saved_rstd);
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_SCALE, W.v},
        {DNNL_ARG_DST, Y.v},
        {DNNL_ARG_VARIANCE, V.v}
    });

    varianceToRstd(q, xr.view.rows, static_cast<float*>(saved_rstd), norm.epsilon);
}

void OneDnnBackend::rmsNormBackward(const RmsNormDesc& norm, const TensorDesc& xd, const void* x, const TensorDesc& dyd, const void* dy, const TensorDesc& wd, const void* weight, const void* saved_rstd, const TensorDesc& dxd, void* dx, void* dweight, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    if (!rmsBackwardTrusted())
        throw DnnUnsupportedError(rmsBackwardDefect());
    if (!saved_rstd)
        throw std::runtime_error("rmsNormBackward requires saved_rstd from rmsNormForward");
    dnn_contract::normalizeLastAxis(xd, norm.axis, "rmsNormBackward");
    dnn_contract::requireSameShape(xd, dyd, "rmsNormBackward");
    dnn_contract::requireSameShape(xd, dxd, "rmsNormBackward");

    RowsMd xr(xd, "rmsNormBackward"), dyr(dyd, "rmsNormBackward"), dxr(dxd, "rmsNormBackward");
    auto wm = vectorMd(wd, xr.view.inner, "rmsNormBackward");

    std::vector<float> mean_square(static_cast<std::size_t>(xr.view.rows));
    std::vector<std::uint8_t> dw_scratch;
    if (!dweight) {
        dw_scratch.resize(static_cast<std::size_t>(xr.view.inner) * wd.elementSize());
        dweight = dw_scratch.data();
    }

    rstdToVariance(q, xr.view.rows, saved_rstd, mean_square.data(), norm.epsilon);

    Mem X( *xr.data, q.e, mut(x)), DY( *dyr.data, q.e, mut(dy)), DX( *dxr.data, q.e, dx);
    Mem W( *wm, q.e, mut(weight)), DW( *wm, q.e, dweight), V( *xr.stats, q.e, mean_square.data());
    unsigned flags = dnnl_rms_norm | dnnl_normalization_use_scale;

    Pd hint;
    checkPd(
        lnForwardPd(
            hint, q.e, dnnl_forward_training,
            *xr.data, *xr.data, *xr.stats, wd.type, norm.epsilon, flags),
        "dnnl_layer_normalization_forward_primitive_desc_create(rms_norm hint)");

    Pd pd;
    checkPd(
        lnBackwardPd(
            pd, q.e, *dxr.data, *dyr.data, *xr.data, *xr.stats, wd.type, norm.epsilon, flags, hint),
        "dnnl_layer_normalization_backward_primitive_desc_create(rms_norm)");

    Prim p(pd);
    dnn_contract::trace("CPU", "rmsNormBackward", "dnnl layer normalization backward (dnnl_rms_norm)");
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DIFF_DST, DY.v},
        {DNNL_ARG_SCALE, W.v},
        {DNNL_ARG_VARIANCE, V.v},
        {DNNL_ARG_DIFF_SRC, DX.v},
        {DNNL_ARG_DIFF_SCALE, DW.v}
    });

    q.complete();
}

// oneDNN exposes dropout only as an attribute fused into other primitives;
// there is no standalone dropout primitive to map this operation onto.
std::size_t OneDnnBackend::dropoutReserveSpaceSize(const TensorDesc& , const ExecutionContext& )
{
    throw DnnUnsupportedError("CLAP_DNN oneDNN: standalone dropout is not available as a native oneDNN primitive");
}

void OneDnnBackend::dropoutForward(const DropoutDesc& , const TensorDesc& , const void* , const TensorDesc& , void* , void* , std::size_t , const ExecutionContext& )
{
    throw DnnUnsupportedError("CLAP_DNN oneDNN: standalone dropout is not available as a native oneDNN primitive");
}

void OneDnnBackend::dropoutBackward(const DropoutDesc& , const TensorDesc& , const void* , const void* , std::size_t , const TensorDesc& , void* , const ExecutionContext& )
{
    throw DnnUnsupportedError("CLAP_DNN oneDNN: standalone dropout is not available as a native oneDNN primitive");
}

void OneDnnBackend::lrnForward(const LrnDesc& lrn, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    Md xm(xd), ym(yd);
    Mem X(xm, q.e, mut(x)), Y(ym, q.e, y);
    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_lrn_forward_primitive_desc_create(
            &pd.v, q.e, dnnl_forward_inference, dnnl_lrn_across_channels,
            xm.v, ym.v, lrn.local_size, (float)lrn.alpha, (float)lrn.beta, (float)lrn.k, nullptr),
        "dnnl_lrn_forward_primitive_desc_create");
    Prim p(pd);
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DST, Y.v}
    });
}

// LRN backward may need the forward workspace; when the selected oneDNN
// implementation requires one it is regenerated with the forward-training
// primitive into host scratch memory owned by this call.
void OneDnnBackend::lrnBackward(const LrnDesc& lrn, const TensorDesc& xd, const void* x, const TensorDesc& yd, const void* y, const TensorDesc& dyd, const void* dy, const TensorDesc& dxd, void* dx, const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);
    Md xm(xd), ym(yd), dym(dyd), dxm(dxd);
    Mem X(xm, q.e, mut(x)), Y(ym, q.e, mut(y)), DY(dym, q.e, mut(dy)), DX(dxm, q.e, dx);
    Pd fwd, bwd;
    checkPd(
        dnn_dyn::p_dnnl_lrn_forward_primitive_desc_create(
            &fwd.v, q.e, dnnl_forward_training, dnnl_lrn_across_channels,
            xm.v, ym.v, lrn.local_size, (float)lrn.alpha, (float)lrn.beta, (float)lrn.k, nullptr),
        "dnnl_lrn_forward_primitive_desc_create");
    checkPd(
        dnn_dyn::p_dnnl_lrn_backward_primitive_desc_create(
            &bwd.v, q.e, dnnl_lrn_across_channels, dxm.v, dym.v, xm.v,
            lrn.local_size, (float)lrn.alpha, (float)lrn.beta, (float)lrn.k, fwd.v, nullptr),
        "dnnl_lrn_backward_primitive_desc_create");

    const std::size_t ws_bytes = workspaceBytes(fwd);

    if (ws_bytes == 0) {
        Prim p(bwd);
        exec(p, q, {
            {DNNL_ARG_SRC, X.v},
            {DNNL_ARG_DST, Y.v},
            {DNNL_ARG_DIFF_DST, DY.v},
            {DNNL_ARG_DIFF_SRC, DX.v}
        });
        return;
    }

    std::vector<std::uint8_t> ws(ws_bytes);
    std::vector<std::uint8_t> y_scratch(yd.bytes());
    const_dnnl_memory_desc_t ws_md = dnn_dyn::p_dnnl_primitive_desc_query_md(fwd.v, dnnl_query_workspace_md, 0);
    Mem WS(ws_md, q.e, ws.data());
    Mem YS(ym, q.e, y_scratch.data());

    Prim pf(fwd);
    exec(pf, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DST, YS.v},
        {DNNL_ARG_WORKSPACE, WS.v}
    });

    Prim pb(bwd);
    exec(pb, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DST, Y.v},
        {DNNL_ARG_DIFF_DST, DY.v},
        {DNNL_ARG_WORKSPACE, WS.v},
        {DNNL_ARG_DIFF_SRC, DX.v}
    });

    q.complete();
}

static void onednnBinary(dnnl_engine_t engine, const ExecutionContext& ctx, dnnl_alg_kind_t alg, const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y)
{
    Cpu q(engine, ctx);
    Md am(ad), bm(bd), ym(yd);
    Mem A(am, q.e, mut(a)), B(bm, q.e, mut(b)), Y(ym, q.e, y);
    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_binary_primitive_desc_create(
            &pd.v, q.e, alg, am.v, bm.v, ym.v, nullptr),
        "dnnl_binary_primitive_desc_create");
    Prim p(pd);
    exec(p, q, {
        {DNNL_ARG_SRC_0, A.v},
        {DNNL_ARG_SRC_1, B.v},
        {DNNL_ARG_DST, Y.v}
    });
}

void OneDnnBackend::tensorAdd(const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    onednnBinary(native_->engine, ctx, dnnl_binary_add, ad, a, bd, b, yd, y);
}

void OneDnnBackend::tensorMultiply(const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    onednnBinary(native_->engine, ctx, dnnl_binary_mul, ad, a, bd, b, yd, y);
}

void OneDnnBackend::tensorMin(const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    onednnBinary(native_->engine, ctx, dnnl_binary_min, ad, a, bd, b, yd, y);
}

void OneDnnBackend::tensorMax(const TensorDesc& ad, const void* a, const TensorDesc& bd, const void* b, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    onednnBinary(native_->engine, ctx, dnnl_binary_max, ad, a, bd, b, yd, y);
}

static void onednnReduce(dnnl_engine_t engine, const ExecutionContext& ctx, dnnl_alg_kind_t alg, const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y)
{
    Cpu q(engine, ctx);
    Md xm(xd), ym(yd);
    Mem X(xm, q.e, mut(x)), Y(ym, q.e, y);
    Pd pd;
    checkPd(
        dnn_dyn::p_dnnl_reduction_primitive_desc_create(
            &pd.v, q.e, alg, xm.v, ym.v, 0, 0, nullptr),
        "dnnl_reduction_primitive_desc_create");
    Prim p(pd);
    exec(p, q, {
        {DNNL_ARG_SRC, X.v},
        {DNNL_ARG_DST, Y.v}
    });
}

void OneDnnBackend::tensorReduceSum(const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    onednnReduce(native_->engine, ctx, dnnl_reduction_sum, xd, x, yd, y);
}

void OneDnnBackend::tensorReduceProduct(const TensorDesc& xd, const void* x, const TensorDesc& yd, void* y, const ExecutionContext& ctx)
{
    onednnReduce(native_->engine, ctx, dnnl_reduction_mul, xd, x, yd, y);
}

// ==========================================================================
// Scaled dot-product attention: oneDNN Graph API SDPA fusion
//
//   MatMul(Q, K^T) -> Multiply(scale) -> [Add(mask)] -> [causal Select]
//   -> SoftMax(axis=-1) -> MatMul(V)
//
// The subgraph must be fused into a single oneDNN partition; otherwise the
// call is reported as unsupported.  GQA/MQA uses the documented 5-D grouped
// pattern Q [B, Hkv, G, S, D] x K [B, Hkv, 1, S, D].
// ==========================================================================

namespace {

struct GraphOp {
    dnnl_graph_op_t v = nullptr;

    GraphOp(std::size_t id, dnnl_graph_op_kind_t kind, const char* name)
    {
        check(dnn_dyn::p_dnnl_graph_op_create(&v, id, kind, name), "dnnl_graph_op_create");
    }

    ~GraphOp()
    {
        if (v) dnn_dyn::p_dnnl_graph_op_destroy(v);
    }

    GraphOp(const GraphOp&) = delete;
    GraphOp& operator=(const GraphOp&) = delete;

    void input(const dnnl_graph_logical_tensor_t& lt)
    {
        check(dnn_dyn::p_dnnl_graph_op_add_input(v, &lt), "dnnl_graph_op_add_input");
    }

    void output(const dnnl_graph_logical_tensor_t& lt)
    {
        check(dnn_dyn::p_dnnl_graph_op_add_output(v, &lt), "dnnl_graph_op_add_output");
    }
};

struct GraphHandle {
    dnnl_graph_graph_t v = nullptr;

    GraphHandle()
    {
        check(dnn_dyn::p_dnnl_graph_graph_create(&v, dnnl_cpu), "dnnl_graph_graph_create");
    }

    ~GraphHandle()
    {
        if (v) dnn_dyn::p_dnnl_graph_graph_destroy(v);
    }
};

struct PartitionHandle {
    dnnl_graph_partition_t v = nullptr;

    ~PartitionHandle()
    {
        if (v) dnn_dyn::p_dnnl_graph_partition_destroy(v);
    }
};

struct GraphTensors {
    std::vector<dnnl_graph_tensor_t> v;

    ~GraphTensors()
    {
        for (auto t : v)
            if (t) dnn_dyn::p_dnnl_graph_tensor_destroy(t);
    }

    void add(const dnnl_graph_logical_tensor_t& lt, dnnl_engine_t engine, const void* data)
    {
        dnnl_graph_tensor_t t = nullptr;
        check(dnn_dyn::p_dnnl_graph_tensor_create(&t, &lt, engine, mut(data)), "dnnl_graph_tensor_create");
        v.push_back(t);
    }
};

dnnl_graph_logical_tensor_t logicalTensor(std::size_t id,
                                          dnnl_data_type_t dt,
                                          const std::vector<dnnl_dim_t>& dims,
                                          const std::vector<dnnl_dim_t>& strides)
{
    dnnl_graph_logical_tensor_t lt {};
    check(
        dnn_dyn::p_dnnl_graph_logical_tensor_init_with_strides(
            &lt,
            id,
            dt,
            static_cast<std::int32_t>(dims.size()),
            dims.data(),
            strides.data(),
            dnnl_graph_tensor_property_variable),
        "dnnl_graph_logical_tensor_init_with_strides");
    return lt;
}

struct AttnView {
    std::vector<dnnl_dim_t> dims;
    std::vector<dnnl_dim_t> strides;
};

// [B, Hq, S, D] -> [B, Hkv, G, S, D] for grouped-query attention.
AttnView queryView(const TensorDesc& d, std::int64_t kv_heads, bool grouped)
{
    const auto s = d.effectiveStrides();
    if (!grouped)
        return {d.dims, s};

    const std::int64_t g = d.dims[1] / kv_heads;
    return {
        {d.dims[0], kv_heads, g, d.dims[2], d.dims[3]},
        {s[0], s[1] * g, s[1], s[2], s[3]}
    };
}

// [B, Hkv, S, D] -> [B, Hkv, 1, S, D] for grouped-query attention.
AttnView keyValueView(const TensorDesc& d, bool grouped)
{
    const auto s = d.effectiveStrides();
    if (!grouped)
        return {d.dims, s};

    return {
        {d.dims[0], d.dims[1], 1, d.dims[2], d.dims[3]},
        {s[0], s[1], s[1], s[2], s[3]}
    };
}

AttnView maskView(const TensorDesc& d, std::int64_t kv_heads, bool grouped)
{
    const auto s = d.effectiveStrides();
    if (!grouped)
        return {d.dims, s};

    if (d.dims[1] == 1) {
        return {
            {d.dims[0], 1, 1, d.dims[2], d.dims[3]},
            {s[0], s[1], s[1], s[2], s[3]}
        };
    }

    const std::int64_t g = d.dims[1] / kv_heads;
    return {
        {d.dims[0], kv_heads, g, d.dims[2], d.dims[3]},
        {s[0], s[1] * g, s[1], s[2], s[3]}
    };
}

void appendKey(std::ostringstream& key, const TensorDesc& d)
{
    key << static_cast<int>(d.type) << ':';
    for (auto v : d.dims) key << v << ',';
    key << '/';
    for (auto v : d.effectiveStrides()) key << v << ',';
    key << ';';
}

std::unique_ptr<onednn_detail::CompiledSdpa> compileSdpa(dnnl_engine_t engine,
                                                         const dnn_contract::AttentionShape& shape,
                                                         bool causal,
                                                         const TensorDesc& qd,
                                                         const TensorDesc& kd,
                                                         const TensorDesc& vd,
                                                         const TensorDesc* mask_desc,
                                                         const TensorDesc& od)
{
    const bool grouped = shape.q_heads != shape.kv_heads;
    const dnnl_data_type_t dt = dnnlType(qd.type);

    const AttnView q_view = queryView(qd, shape.kv_heads, grouped);
    const AttnView k_view = keyValueView(kd, grouped);
    const AttnView v_view = keyValueView(vd, grouped);
    const AttnView o_view = queryView(od, shape.kv_heads, grouped);

    std::vector<dnnl_dim_t> score_dims = q_view.dims;
    score_dims.back() = shape.kv_len;
    const std::vector<dnnl_dim_t> score_strides = packed(score_dims);

    std::size_t id = 0;
    std::vector<std::unique_ptr<GraphOp>> ops;

    auto lt_q = logicalTensor(id++, dt, q_view.dims, q_view.strides);
    auto lt_k = logicalTensor(id++, dt, k_view.dims, k_view.strides);
    auto lt_score = logicalTensor(id++, dnnl_f32, score_dims, score_strides);

    ops.emplace_back(new GraphOp(id++, dnnl_graph_op_matmul, "clap_sdpa_qk"));
    const std::uint8_t transpose_b = 1;
    check(
        dnn_dyn::p_dnnl_graph_op_set_attr_bool(
            ops.back()->v, dnnl_graph_op_attr_transpose_b, &transpose_b, 1),
        "dnnl_graph_op_set_attr_bool(transpose_b)");
    ops.back()->input(lt_q);
    ops.back()->input(lt_k);
    ops.back()->output(lt_score);

    auto lt_scale = logicalTensor(id++, dnnl_f32, {1}, {1});
    auto lt_scaled = logicalTensor(id++, dnnl_f32, score_dims, score_strides);

    ops.emplace_back(new GraphOp(id++, dnnl_graph_op_multiply, "clap_sdpa_scale"));
    ops.back()->input(lt_score);
    ops.back()->input(lt_scale);
    ops.back()->output(lt_scaled);

    dnnl_graph_logical_tensor_t last = lt_scaled;
    dnnl_graph_logical_tensor_t lt_mask {};
    dnnl_graph_logical_tensor_t lt_neg_inf {};

    if (mask_desc) {
        const AttnView m_view = maskView(*mask_desc, shape.kv_heads, grouped);
        lt_mask = logicalTensor(id++, dnnlType(mask_desc->type), m_view.dims, m_view.strides);
        auto lt_masked = logicalTensor(id++, dnnl_f32, score_dims, score_strides);

        ops.emplace_back(new GraphOp(id++, dnnl_graph_op_add, "clap_sdpa_mask"));
        ops.back()->input(last);
        ops.back()->input(lt_mask);
        ops.back()->output(lt_masked);
        last = lt_masked;
    }

    if (causal) {
        auto lt_row = logicalTensor(id++, dnnl_s32, score_dims, score_strides);
        auto lt_col = logicalTensor(id++, dnnl_s32, score_dims, score_strides);
        auto lt_keep = logicalTensor(id++, dnnl_boolean, score_dims, score_strides);
        lt_neg_inf = logicalTensor(id++, dnnl_f32, {1}, {1});
        auto lt_selected = logicalTensor(id++, dnnl_f32, score_dims, score_strides);

        const std::int64_t row_axis = -2;
        const std::int64_t col_axis = -1;

        ops.emplace_back(new GraphOp(id++, dnnl_graph_op_gen_index, "clap_sdpa_row_index"));
        check(
            dnn_dyn::p_dnnl_graph_op_set_attr_s64(
                ops.back()->v, dnnl_graph_op_attr_axis, &row_axis, 1),
            "dnnl_graph_op_set_attr_s64(axis)");
        ops.back()->input(last);
        ops.back()->output(lt_row);

        ops.emplace_back(new GraphOp(id++, dnnl_graph_op_gen_index, "clap_sdpa_col_index"));
        check(
            dnn_dyn::p_dnnl_graph_op_set_attr_s64(
                ops.back()->v, dnnl_graph_op_attr_axis, &col_axis, 1),
            "dnnl_graph_op_set_attr_s64(axis)");
        ops.back()->input(last);
        ops.back()->output(lt_col);

        ops.emplace_back(new GraphOp(id++, dnnl_graph_op_greater_equal, "clap_sdpa_causal"));
        ops.back()->input(lt_row);
        ops.back()->input(lt_col);
        ops.back()->output(lt_keep);

        ops.emplace_back(new GraphOp(id++, dnnl_graph_op_select, "clap_sdpa_causal_select"));
        ops.back()->input(lt_keep);
        ops.back()->input(last);
        ops.back()->input(lt_neg_inf);
        ops.back()->output(lt_selected);
        last = lt_selected;
    }

    auto lt_probs = logicalTensor(id++, dt, score_dims, score_strides);
    const std::int64_t softmax_axis = -1;

    ops.emplace_back(new GraphOp(id++, dnnl_graph_op_softmax, "clap_sdpa_softmax"));
    check(
        dnn_dyn::p_dnnl_graph_op_set_attr_s64(
            ops.back()->v, dnnl_graph_op_attr_axis, &softmax_axis, 1),
        "dnnl_graph_op_set_attr_s64(axis)");
    ops.back()->input(last);
    ops.back()->output(lt_probs);

    auto lt_v = logicalTensor(id++, dt, v_view.dims, v_view.strides);
    auto lt_o = logicalTensor(id++, dt, o_view.dims, o_view.strides);

    ops.emplace_back(new GraphOp(id++, dnnl_graph_op_matmul, "clap_sdpa_pv"));
    ops.back()->input(lt_probs);
    ops.back()->input(lt_v);
    ops.back()->output(lt_o);

    GraphHandle graph;
    for (auto& op : ops)
        check(dnn_dyn::p_dnnl_graph_add_op(graph.v, op->v), "dnnl_graph_add_op");

    check(dnn_dyn::p_dnnl_graph_graph_finalize(graph.v), "dnnl_graph_graph_finalize");
    check(
        dnn_dyn::p_dnnl_graph_graph_filter(graph.v, dnnl_graph_partition_policy_fusion),
        "dnnl_graph_graph_filter");

    std::size_t partitions = 0;
    check(
        dnn_dyn::p_dnnl_graph_graph_get_partition_num(graph.v, &partitions),
        "dnnl_graph_graph_get_partition_num");
    if (partitions != 1) {
        throw DnnUnsupportedError(
            "CLAP_DNN oneDNN: the SDPA subgraph was split into " + std::to_string(partitions) +
            " partitions; no fused native SDPA kernel for this configuration");
    }

    PartitionHandle partition;
    check(
        dnn_dyn::p_dnnl_graph_graph_get_partitions(graph.v, 1, &partition.v),
        "dnnl_graph_graph_get_partitions");

    std::uint8_t supported = 0;
    check(
        dnn_dyn::p_dnnl_graph_partition_is_supported(partition.v, &supported),
        "dnnl_graph_partition_is_supported");
    if (!supported)
        throw DnnUnsupportedError("CLAP_DNN oneDNN: the SDPA partition is not supported by this oneDNN build");

    std::unique_ptr<onednn_detail::CompiledSdpa> sdpa(new onednn_detail::CompiledSdpa());

    sdpa->slot_q = static_cast<int>(sdpa->inputs.size());
    sdpa->inputs.push_back(lt_q);
    sdpa->slot_k = static_cast<int>(sdpa->inputs.size());
    sdpa->inputs.push_back(lt_k);
    sdpa->slot_scale = static_cast<int>(sdpa->inputs.size());
    sdpa->inputs.push_back(lt_scale);
    if (mask_desc) {
        sdpa->slot_mask = static_cast<int>(sdpa->inputs.size());
        sdpa->inputs.push_back(lt_mask);
    }
    if (causal) {
        sdpa->slot_neg_inf = static_cast<int>(sdpa->inputs.size());
        sdpa->inputs.push_back(lt_neg_inf);
    }
    sdpa->slot_v = static_cast<int>(sdpa->inputs.size());
    sdpa->inputs.push_back(lt_v);
    sdpa->output = lt_o;

    std::vector<const dnnl_graph_logical_tensor_t*> in_ptrs;
    for (const auto& lt : sdpa->inputs)
        in_ptrs.push_back(&lt);
    const dnnl_graph_logical_tensor_t* out_ptr = &sdpa->output;

    check(
        dnn_dyn::p_dnnl_graph_compiled_partition_create(&sdpa->cp, partition.v),
        "dnnl_graph_compiled_partition_create");

    const dnnl_status_t status =
        dnn_dyn::p_dnnl_graph_partition_compile(
            partition.v,
            sdpa->cp,
            in_ptrs.size(),
            in_ptrs.data(),
            1,
            &out_ptr,
            engine);
    checkPd(status, "dnnl_graph_partition_compile(SDPA)");

    return sdpa;
}

} // namespace

static onednn_detail::CompiledSdpa& sdpaPartition(std::mutex& mutex,
                                                  std::map<std::string, std::unique_ptr<onednn_detail::CompiledSdpa>>& cache,
                                                  dnnl_engine_t engine,
                                                  const dnn_contract::AttentionShape& shape,
                                                  bool causal,
                                                  const TensorDesc& qd,
                                                  const TensorDesc& kd,
                                                  const TensorDesc& vd,
                                                  const TensorDesc* mask_desc,
                                                  const TensorDesc& od)
{
    std::ostringstream key;
    key << engine << '|' << causal << '|';
    appendKey(key, qd);
    appendKey(key, kd);
    appendKey(key, vd);
    appendKey(key, od);
    if (mask_desc)
        appendKey(key, *mask_desc);

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key.str());
    if (it == cache.end())
        it = cache.emplace(key.str(), compileSdpa(engine, shape, causal, qd, kd, vd, mask_desc, od)).first;
    return *it->second;
}

void OneDnnBackend::scaledDotProductAttentionForward(const AttentionDesc& attention,
                                                     const TensorDesc& qd,
                                                     const void* q_ptr,
                                                     const TensorDesc& kd,
                                                     const void* k_ptr,
                                                     const TensorDesc& vd,
                                                     const void* v_ptr,
                                                     const TensorDesc* mask_desc,
                                                     const void* mask,
                                                     const TensorDesc& od,
                                                     void* o_ptr,
                                                     const ExecutionContext& ctx)
{
    Cpu q(native_->engine, ctx);

    if ((mask_desc == nullptr) != (mask == nullptr))
        throw std::runtime_error("scaledDotProductAttentionForward: mask_desc and mask must both be set or both be null");

    const auto shape = dnn_contract::validateAttention(attention, qd, kd, vd, mask_desc, od);

    if (!dnn_dyn::hasOneDnnGraph())
        dnn_contract::unsupported("oneDNN", "the installed oneDNN does not export the Graph API required for fused SDPA");
    if (attention.dropout > 0.0f)
        dnn_contract::unsupported("oneDNN", "attention dropout is not part of the oneDNN SDPA inference pattern");

    auto& sdpa = sdpaPartition(native_->sdpa_mutex, native_->sdpa, q.e, shape, attention.causal, qd, kd, vd, mask_desc, od);

    const float scale = static_cast<float>(shape.scale);
    const float neg_inf = -std::numeric_limits<float>::infinity();

    GraphTensors inputs;
    for (int i = 0; i < static_cast<int>(sdpa.inputs.size()); ++i) {
        const void* data = nullptr;
        if (i == sdpa.slot_q) data = q_ptr;
        else if (i == sdpa.slot_k) data = k_ptr;
        else if (i == sdpa.slot_v) data = v_ptr;
        else if (i == sdpa.slot_scale) data = &scale;
        else if (i == sdpa.slot_mask) data = mask;
        else if (i == sdpa.slot_neg_inf) data = &neg_inf;
        inputs.add(sdpa.inputs[i], q.e, data);
    }

    GraphTensors outputs;
    outputs.add(sdpa.output, q.e, o_ptr);

    dnn_contract::trace("CPU", "scaledDotProductAttentionForward", "oneDNN Graph fused SDPA partition");

    std::vector<const_dnnl_graph_tensor_t> in_tensors(inputs.v.begin(), inputs.v.end());
    std::vector<const_dnnl_graph_tensor_t> out_tensors(outputs.v.begin(), outputs.v.end());

    check(
        dnn_dyn::p_dnnl_graph_compiled_partition_execute(
            sdpa.cp,
            q.s,
            in_tensors.size(),
            in_tensors.data(),
            out_tensors.size(),
            out_tensors.data()),
        "dnnl_graph_compiled_partition_execute");

    // scale and neg_inf live on this stack frame.
    q.complete();
}

// ==========================================================================
// Capability model: oneDNN capabilities are probed by creating the primitive
// descriptor (or compiling the graph partition) that execution would use.
// ==========================================================================

static DnnSupport probeResult(dnnl_status_t status, const char* what)
{
    if (status == dnnl_success)
        return DnnSupport::yes();
    if (status == dnnl_unimplemented)
        return DnnSupport::no(std::string("oneDNN has no implementation of ") + what + " for this datatype/layout on this CPU");
    return DnnSupport::no(std::string("oneDNN rejected the ") + what + " descriptor (status=" + std::to_string(status) + ")");
}

DnnSupport OneDnnBackend::supports(const DnnCapabilityQuery& query, const ExecutionContext& ctx) const
{
    if (ctx.backend != ExecutionBackend::CPU)
        return DnnSupport::no("the oneDNN backend executes on host memory (ExecutionBackend::CPU)");

    try {
        dnnl_engine_t engine = native_->engine;
        const TensorDesc& x = query.tensor(0);

        switch (query.operation) {
            case DnnOperation::ActivationForward:
            case DnnOperation::ActivationBackward: {
                Md xm(x);
                Pd pd;
                return probeResult(
                    dnn_dyn::p_dnnl_eltwise_forward_primitive_desc_create(
                        &pd.v, engine, dnnl_forward_training, act_alg(query.activation.mode),
                        xm.v, xm.v, act_alpha(query.activation), 0.0f, nullptr),
                    "eltwise");
            }

            case DnnOperation::SoftmaxForward:
            case DnnOperation::SoftmaxBackward: {
                const int axis = x.normalizeAxis(query.softmax.axis, "supports(softmax)");
                Md xm(x);
                Pd pd;
                return probeResult(
                    dnn_dyn::p_dnnl_softmax_forward_primitive_desc_create(
                        &pd.v, engine, dnnl_forward_training,
                        query.softmax.log_softmax ? dnnl_softmax_log : dnnl_softmax_accurate,
                        xm.v, xm.v, axis, nullptr),
                    query.softmax.log_softmax ? "log-softmax" : "softmax");
            }

            case DnnOperation::LayerNormForward:
            case DnnOperation::LayerNormBackward:
            case DnnOperation::RmsNormForward:
            case DnnOperation::RmsNormBackward: {
                const bool rms = query.operation == DnnOperation::RmsNormForward ||
                                 query.operation == DnnOperation::RmsNormBackward;
                if (rms)
                    dnn_contract::normalizeLastAxis(x, query.rms_norm.axis, "supports(rmsNorm)");
                RowsMd xr(x, "supports(normalization)");
                const unsigned flags = rms
                    ? (dnnl_rms_norm | dnnl_normalization_use_scale)
                    : (dnnl_normalization_use_scale | dnnl_normalization_use_shift);
                if (query.operation == DnnOperation::RmsNormBackward && !rmsBackwardTrusted())
                    return DnnSupport::no(rmsBackwardDefect());
                const DataType weight_type = query.tensors.size() > 1 ? query.tensor(1).type : x.type;
                Pd pd;
                return probeResult(
                    lnForwardPd(
                        pd, engine, dnnl_forward_training,
                        *xr.data, *xr.data, *xr.stats, weight_type, 1.0e-5, flags),
                    rms ? "layer normalization (dnnl_rms_norm)" : "layer normalization");
            }

            case DnnOperation::ScaledDotProductAttentionForward: {
                if (!dnn_dyn::hasOneDnnGraph())
                    return DnnSupport::no("the installed oneDNN does not export the Graph API required for fused SDPA");
                if (query.attention.dropout > 0.0f)
                    return DnnSupport::no("attention dropout is not part of the oneDNN SDPA inference pattern");
                const TensorDesc* mask = query.has_mask ? &query.tensor(4) : nullptr;
                const auto shape = dnn_contract::validateAttention(
                    query.attention, query.tensor(0), query.tensor(1), query.tensor(2), mask, query.tensor(3));
                sdpaPartition(native_->sdpa_mutex, native_->sdpa, engine, shape, query.attention.causal,
                              query.tensor(0), query.tensor(1), query.tensor(2), mask, query.tensor(3));
                return DnnSupport::yes("fused oneDNN Graph SDPA partition");
            }

            case DnnOperation::DropoutForward:
            case DnnOperation::DropoutBackward:
                return DnnSupport::no("oneDNN has no standalone dropout primitive (dropout is only an attribute of other primitives)");

            case DnnOperation::TensorAdd:
            case DnnOperation::TensorMultiply:
            case DnnOperation::TensorMin:
            case DnnOperation::TensorMax: {
                Md am(query.tensor(0)), bm(query.tensor(1)), ym(query.tensor(2));
                const dnnl_alg_kind_t alg =
                    query.operation == DnnOperation::TensorAdd ? dnnl_binary_add :
                    query.operation == DnnOperation::TensorMultiply ? dnnl_binary_mul :
                    query.operation == DnnOperation::TensorMin ? dnnl_binary_min : dnnl_binary_max;
                Pd pd;
                return probeResult(
                    dnn_dyn::p_dnnl_binary_primitive_desc_create(&pd.v, engine, alg, am.v, bm.v, ym.v, nullptr),
                    "binary");
            }

            case DnnOperation::TensorReduceSum:
            case DnnOperation::TensorReduceProduct: {
                if (query.tensors.size() < 2)
                    return DnnSupport::no("reduction capability queries need {x, y} descriptors");
                Md xm(query.tensor(0)), ym(query.tensor(1));
                Pd pd;
                return probeResult(
                    dnn_dyn::p_dnnl_reduction_primitive_desc_create(
                        &pd.v, engine,
                        query.operation == DnnOperation::TensorReduceSum ? dnnl_reduction_sum : dnnl_reduction_mul,
                        xm.v, ym.v, 0.0f, 0.0f, nullptr),
                    "reduction");
            }

            default:
                if (x.rank() != 4)
                    return DnnSupport::no(std::string(dnnOperationName(query.operation)) + " expects 4D NCHW tensors");
                return DnnSupport::yes("validated by oneDNN at primitive creation");
        }
    }
    catch (const std::exception& e) {
        return DnnSupport::no(e.what());
    }
}

}
// namespace clap
