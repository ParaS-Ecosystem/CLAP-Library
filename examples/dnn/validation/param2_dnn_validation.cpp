#include <dnn.hh>
#include "clap/examples/dnn_example_device.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;
int g_skip = 0;

void report(const std::string& name, bool ok, const std::string& detail = {})
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name;
    if (!detail.empty())
        std::cout << "  (" << detail << ")";
    std::cout << '\n';
    (ok ? g_pass : g_fail)++;
}

void skip(const std::string& name, const std::string& why)
{
    std::cout << "[SKIP] " << name << "  (" << why << ")\n";
    g_skip++;
}

bool close(const std::vector<float>& got, const std::vector<float>& ref, double tol, std::string& detail)
{
    if (got.size() != ref.size()) {
        detail = "size mismatch";
        return false;
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::isnan(ref[i]) != std::isnan(got[i])) {
            detail = "NaN mismatch at " + std::to_string(i);
            return false;
        }
        if (std::isnan(ref[i]))
            continue;
        const double err = std::fabs(static_cast<double>(got[i]) - ref[i]) / std::max(1.0, std::fabs(static_cast<double>(ref[i])));
        worst = std::max(worst, err);
    }
    detail = "max rel err " + std::to_string(worst);
    return worst <= tol;
}

double tolerance(clap::DataType type)
{
    switch (type) {
        case clap::DataType::Float32:  return 2.0e-5;
        case clap::DataType::Float16:  return 4.0e-3;
        case clap::DataType::BFloat16: return 3.0e-2;
    }
    return 1.0e-5;
}

std::vector<float> pattern(std::size_t n, float scale, float phase)
{
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = scale * std::sin(0.37f * static_cast<float>(i) + phase);
    return v;
}

std::vector<float> quantize(const std::vector<float>& v, clap::DataType type)
{
    return clap_example::decode(clap_example::encode(v, type), type);
}

std::size_t offsetOf(const std::vector<std::int64_t>& index, const std::vector<std::int64_t>& strides)
{
    std::size_t off = 0;
    for (std::size_t i = 0; i < index.size(); ++i)
        off += static_cast<std::size_t>(index[i] * strides[i]);
    return off;
}

void forEachIndex(const std::vector<std::int64_t>& dims, const std::function<void(const std::vector<std::int64_t>&)>& fn)
{
    std::vector<std::int64_t> idx(dims.size(), 0);
    std::size_t total = 1;
    for (auto d : dims) total *= static_cast<std::size_t>(d);
    for (std::size_t n = 0; n < total; ++n) {
        fn(idx);
        for (std::size_t k = dims.size(); k-- > 0;) {
            if (++idx[k] < dims[k]) break;
            idx[k] = 0;
        }
    }
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void refSoftmax(const std::vector<float>& x, std::vector<float>& y,
                const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& strides,
                int axis, bool log_softmax)
{
    std::vector<std::int64_t> outer = dims;
    outer[static_cast<std::size_t>(axis)] = 1;
    forEachIndex(outer, [&](const std::vector<std::int64_t>& base) {
        std::vector<std::int64_t> idx = base;
        double mx = -std::numeric_limits<double>::infinity();
        for (std::int64_t a = 0; a < dims[static_cast<std::size_t>(axis)]; ++a) {
            idx[static_cast<std::size_t>(axis)] = a;
            mx = std::max(mx, static_cast<double>(x[offsetOf(idx, strides)]));
        }
        double sum = 0.0;
        for (std::int64_t a = 0; a < dims[static_cast<std::size_t>(axis)]; ++a) {
            idx[static_cast<std::size_t>(axis)] = a;
            sum += std::exp(x[offsetOf(idx, strides)] - mx);
        }
        for (std::int64_t a = 0; a < dims[static_cast<std::size_t>(axis)]; ++a) {
            idx[static_cast<std::size_t>(axis)] = a;
            const double v = x[offsetOf(idx, strides)] - mx;
            y[offsetOf(idx, strides)] = static_cast<float>(log_softmax ? v - std::log(sum) : std::exp(v) / sum);
        }
    });
}

void refRmsNorm(const std::vector<float>& x, const std::vector<float>& w, std::size_t rows, std::size_t inner,
                double eps, std::vector<float>& y, std::vector<float>& rstd)
{
    for (std::size_t r = 0; r < rows; ++r) {
        double ms = 0.0;
        for (std::size_t j = 0; j < inner; ++j) ms += static_cast<double>(x[r * inner + j]) * x[r * inner + j];
        ms /= static_cast<double>(inner);
        const double rs = 1.0 / std::sqrt(ms + eps);
        rstd[r] = static_cast<float>(rs);
        for (std::size_t j = 0; j < inner; ++j)
            y[r * inner + j] = static_cast<float>(x[r * inner + j] * rs * w[j]);
    }
}

void refRmsNormBackward(const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>& dy,
                        std::size_t rows, std::size_t inner, double eps,
                        std::vector<float>& dx, std::vector<float>& dw)
{
    std::fill(dw.begin(), dw.end(), 0.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        double ms = 0.0;
        for (std::size_t j = 0; j < inner; ++j) ms += static_cast<double>(x[r * inner + j]) * x[r * inner + j];
        ms /= static_cast<double>(inner);
        const double rs = 1.0 / std::sqrt(ms + eps);
        double dot = 0.0;
        for (std::size_t j = 0; j < inner; ++j) dot += static_cast<double>(dy[r * inner + j]) * w[j] * x[r * inner + j];
        for (std::size_t j = 0; j < inner; ++j) {
            const double g = static_cast<double>(dy[r * inner + j]) * w[j];
            dx[r * inner + j] = static_cast<float>(rs * g - rs * rs * rs * x[r * inner + j] * dot / static_cast<double>(inner));
            dw[j] += static_cast<float>(dy[r * inner + j] * x[r * inner + j] * rs);
        }
    }
}

void refLayerNorm(const std::vector<float>& x, const std::vector<float>& s, const std::vector<float>& b,
                  std::size_t rows, std::size_t inner, double eps,
                  std::vector<float>& y, std::vector<float>& mean, std::vector<float>& rstd)
{
    for (std::size_t r = 0; r < rows; ++r) {
        double mu = 0.0, var = 0.0;
        for (std::size_t j = 0; j < inner; ++j) mu += x[r * inner + j];
        mu /= static_cast<double>(inner);
        for (std::size_t j = 0; j < inner; ++j) var += (x[r * inner + j] - mu) * (x[r * inner + j] - mu);
        var /= static_cast<double>(inner);
        const double rs = 1.0 / std::sqrt(var + eps);
        mean[r] = static_cast<float>(mu);
        rstd[r] = static_cast<float>(rs);
        for (std::size_t j = 0; j < inner; ++j)
            y[r * inner + j] = static_cast<float>((x[r * inner + j] - mu) * rs * s[j] + b[j]);
    }
}

void refAttention(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<float>* mask, const std::vector<std::int64_t>& mask_dims,
                  std::int64_t B, std::int64_t Hq, std::int64_t Hkv, std::int64_t Sq, std::int64_t Skv,
                  std::int64_t D, double scale, bool causal, std::vector<float>& o)
{
    const std::int64_t group = Hq / Hkv;
    for (std::int64_t b = 0; b < B; ++b)
    for (std::int64_t h = 0; h < Hq; ++h)
    for (std::int64_t i = 0; i < Sq; ++i) {
        const std::int64_t hk = h / group;
        std::vector<double> s(static_cast<std::size_t>(Skv));
        double mx = -std::numeric_limits<double>::infinity();
        for (std::int64_t j = 0; j < Skv; ++j) {
            double dot = 0.0;
            for (std::int64_t d = 0; d < D; ++d)
                dot += static_cast<double>(q[((b * Hq + h) * Sq + i) * D + d]) * k[((b * Hkv + hk) * Skv + j) * D + d];
            dot *= scale;
            if (mask) {
                const std::int64_t mb = mask_dims[0] == 1 ? 0 : b;
                const std::int64_t mh = mask_dims[1] == 1 ? 0 : h;
                const std::int64_t mi = mask_dims[2] == 1 ? 0 : i;
                const std::int64_t mj = mask_dims[3] == 1 ? 0 : j;
                dot += (*mask)[((mb * mask_dims[1] + mh) * mask_dims[2] + mi) * mask_dims[3] + mj];
            }
            if (causal && j > i)
                dot = -std::numeric_limits<double>::infinity();
            s[static_cast<std::size_t>(j)] = dot;
            mx = std::max(mx, dot);
        }
        double sum = 0.0;
        for (auto& e : s) { e = std::exp(e - mx); sum += e; }
        for (std::int64_t d = 0; d < D; ++d) {
            double acc = 0.0;
            for (std::int64_t j = 0; j < Skv; ++j)
                acc += s[static_cast<std::size_t>(j)] / sum * v[((b * Hkv + hk) * Skv + j) * D + d];
            o[((b * Hq + h) * Sq + i) * D + d] = static_cast<float>(acc);
        }
    }
}

void checkSilu(clap::IDnnBackend& backend, clap_example::Device& dev, clap::DataType type)
{
    const std::string name = std::string("SiLU forward/backward ") + clap::dataTypeName(type) + " [2,3,5]";
    clap::TensorDesc d(type, {2, 3, 5});
    clap::ActivationDesc silu;
    silu.mode = clap::ActivationMode::SiLU;

    const auto s = backend.supports(clap::DnnCapabilityQuery::forActivation(clap::DnnOperation::ActivationForward, silu, d), dev.ctx);
    if (!s) { skip(name, s.reason); return; }

    const auto x = quantize(pattern(30, 3.0f, 0.1f), type);
    const auto dy = quantize(pattern(30, 1.0f, 0.7f), type);
    std::vector<float> ry(30), rdx(30);
    for (std::size_t i = 0; i < 30; ++i) {
        const float sg = sigmoid(x[i]);
        ry[i] = x[i] * sg;
        rdx[i] = dy[i] * sg * (1.0f + x[i] * (1.0f - sg));
    }

    void* X = dev.uploadVector(clap_example::encode(x, type));
    void* DY = dev.uploadVector(clap_example::encode(dy, type));
    void* Y = dev.allocate(d.bytes());
    void* DX = dev.allocate(d.bytes());

    try {
        backend.activationForward(silu, d, X, d, Y, dev.ctx);
        std::string detail;
        report(name + " forward", close(clap_example::decode(dev.downloadVector<std::uint8_t>(Y, d.bytes()), type), ry, tolerance(type), detail), detail);
    } catch (const clap::DnnUnsupportedError& e) {
        skip(name + " forward", e.what());
    }

    try {
        backend.activationBackward(silu, d, X, d, DY, d, DX, dev.ctx);
        std::string detail;
        report(name + " backward", close(clap_example::decode(dev.downloadVector<std::uint8_t>(DX, d.bytes()), type), rdx, 4 * tolerance(type), detail), detail);
    } catch (const clap::DnnUnsupportedError& e) {
        skip(name + " backward", e.what());
    }
}

void checkSoftmax(clap::IDnnBackend& backend, clap_example::Device& dev, clap::DataType type,
                  const std::vector<std::int64_t>& dims, const std::vector<std::int64_t>& strides,
                  int axis, bool log_softmax)
{
    clap::TensorDesc d(type, dims, strides);
    const auto eff = d.effectiveStrides();
    std::string name = std::string(log_softmax ? "LogSoftmax" : "Softmax") + " " + clap::dataTypeName(type) + " dims[";
    for (auto v : dims) name += std::to_string(v) + ",";
    name.back() = ']';
    name += strides.empty() ? "" : " strided";
    name += " axis " + std::to_string(axis);

    clap::SoftmaxDesc sm;
    sm.axis = axis;
    sm.log_softmax = log_softmax;

    const auto s = backend.supports(clap::DnnCapabilityQuery::forSoftmax(clap::DnnOperation::SoftmaxForward, sm, d), dev.ctx);
    if (!s) { skip(name, s.reason); return; }

    const std::size_t span = d.spanElements();
    const auto x = quantize(pattern(span, 2.0f, 0.3f), type);
    std::vector<float> ry(span, 0.0f);
    refSoftmax(x, ry, dims, eff, d.normalizeAxis(axis, "test"), log_softmax);

    const std::size_t guard = 64;
    std::vector<float> y_init(span + guard, 7.0f);
    void* X = dev.uploadVector(clap_example::encode(x, type));
    void* Y = dev.uploadVector(clap_example::encode(y_init, type));

    backend.softmaxForward(sm, d, X, d, Y, dev.ctx);
    const auto got = clap_example::decode(dev.downloadVector<std::uint8_t>(Y, (span + guard) * d.elementSize()), type);

    std::vector<float> got_tensor(span, 0.0f), ref_tensor(span, 0.0f);
    forEachIndex(dims, [&](const std::vector<std::int64_t>& idx) {
        const std::size_t o = offsetOf(idx, eff);
        got_tensor[o] = got[o];
        ref_tensor[o] = ry[o];
    });

    bool guard_ok = true;
    for (std::size_t i = span; i < span + guard; ++i)
        guard_ok = guard_ok && got[i] == 7.0f;

    std::string detail;
    const bool ok = close(got_tensor, ref_tensor, tolerance(type), detail) && guard_ok;
    report(name, ok, detail + (guard_ok ? "" : ", wrote outside tensor"));
}

void checkSoftmaxBackward(clap::IDnnBackend& backend, clap_example::Device& dev, bool log_softmax)
{
    const std::string name = std::string(log_softmax ? "LogSoftmax" : "Softmax") + " backward Float32 [3,7] axis -1";
    clap::TensorDesc d(clap::DataType::Float32, {3, 7});
    clap::SoftmaxDesc sm;
    sm.axis = -1;
    sm.log_softmax = log_softmax;

    const auto s = backend.supports(clap::DnnCapabilityQuery::forSoftmax(clap::DnnOperation::SoftmaxBackward, sm, d), dev.ctx);
    if (!s) { skip(name, s.reason); return; }

    const auto x = pattern(21, 2.0f, 0.2f);
    const auto dy = pattern(21, 1.0f, 1.1f);
    std::vector<float> y(21), rdx(21);
    refSoftmax(x, y, {3, 7}, {7, 1}, 1, log_softmax);
    for (int r = 0; r < 3; ++r) {
        double acc = 0.0;
        for (int j = 0; j < 7; ++j)
            acc += log_softmax ? dy[r * 7 + j] : static_cast<double>(dy[r * 7 + j]) * y[r * 7 + j];
        for (int j = 0; j < 7; ++j) {
            const std::size_t i = static_cast<std::size_t>(r * 7 + j);
            rdx[i] = log_softmax
                ? static_cast<float>(dy[i] - std::exp(y[i]) * acc)
                : static_cast<float>(y[i] * (dy[i] - acc));
        }
    }

    void* Y = dev.uploadVector(y);
    void* DY = dev.uploadVector(dy);
    void* DX = dev.allocate(d.bytes());
    backend.softmaxBackward(sm, d, Y, d, DY, d, DX, dev.ctx);

    std::string detail;
    report(name, close(dev.downloadVector<float>(DX, 21), rdx, 1.0e-4, detail), detail);
}

void checkLegacySoftmax(clap::IDnnBackend& backend, clap_example::Device& dev)
{
    const std::string name = "Historical softmaxForward(float*) == SoftmaxDesc{axis=1}";
    if (dev.onGpu()) {
        clap::TensorDesc d({2, 3, 2, 2});
        const auto x = pattern(24, 2.0f, 0.0f);
        std::vector<float> legacy(24), ref(24);
        backend.softmaxForward(d, x.data(), d, legacy.data());
        refSoftmax(x, ref, {2, 3, 2, 2}, {12, 4, 2, 1}, 1, false);
        std::string detail;
        report(name, close(legacy, ref, 1.0e-5, detail), detail);
        return;
    }

    clap::TensorDesc d({2, 3, 2, 2});
    const auto x = pattern(24, 2.0f, 0.0f);
    std::vector<float> legacy(24), modern(24);
    backend.softmaxForward(d, x.data(), d, legacy.data());
    clap::SoftmaxDesc sm;
    sm.axis = 1;
    backend.softmaxForward(sm, d, x.data(), d, modern.data(), clap::ExecutionContext::cpu());
    std::string detail;
    report(name, close(legacy, modern, 0.0, detail), detail);
}

void checkRmsNorm(clap::IDnnBackend& backend, clap_example::Device& dev, clap::DataType type, bool training)
{
    const std::size_t rows = 6, inner = 16;
    const std::string name = std::string("RMSNorm ") + (training ? "train fwd+bwd " : "inference fwd ") + clap::dataTypeName(type) + " [2,3,16]";
    clap::TensorDesc d(type, {2, 3, 16});
    clap::TensorDesc wd(type, {16});
    clap::RmsNormDesc norm;
    norm.epsilon = 1.0e-6;

    const auto s = backend.supports(clap::DnnCapabilityQuery::forRmsNorm(clap::DnnOperation::RmsNormForward, norm, d, wd), dev.ctx);
    if (!s) { skip(name, s.reason); return; }

    const auto x = quantize(pattern(rows * inner, 2.0f, 0.4f), type);
    const auto w = quantize(pattern(inner, 0.5f, 1.3f), type);
    const auto dy = quantize(pattern(rows * inner, 1.0f, 2.1f), type);

    std::vector<float> ry(rows * inner), rrstd(rows), rdx(rows * inner), rdw(inner);
    refRmsNorm(x, w, rows, inner, norm.epsilon, ry, rrstd);
    refRmsNormBackward(x, w, dy, rows, inner, norm.epsilon, rdx, rdw);

    void* X = dev.uploadVector(clap_example::encode(x, type));
    void* W = dev.uploadVector(clap_example::encode(w, type));
    void* DY = dev.uploadVector(clap_example::encode(dy, type));
    void* Y = dev.allocate(d.bytes());
    void* R = training ? dev.allocate(rows * sizeof(float)) : nullptr;
    void* DX = dev.allocate(d.bytes());
    void* DW = dev.allocate(wd.bytes());

    try {
        backend.rmsNormForward(norm, d, X, wd, W, d, Y, R, dev.ctx);
        std::string detail;
        bool ok = close(clap_example::decode(dev.downloadVector<std::uint8_t>(Y, d.bytes()), type), ry, tolerance(type), detail);
        if (training) {
            std::string rd;
            const bool rok = close(dev.downloadVector<float>(R, rows), rrstd, 1.0e-4, rd);
            ok = ok && rok;
            detail += ", rstd " + rd;
        }
        report(name + " forward", ok, detail);
    } catch (const clap::DnnUnsupportedError& e) {
        skip(name + " forward", e.what());
        return;
    }

    if (!training)
        return;

    const auto sb = backend.supports(clap::DnnCapabilityQuery::forRmsNorm(clap::DnnOperation::RmsNormBackward, norm, d, wd), dev.ctx);
    if (!sb) { skip(name + " backward", sb.reason); return; }

    try {
        backend.rmsNormBackward(norm, d, X, d, DY, wd, W, R, d, DX, DW, dev.ctx);
        std::string dxd, dwd;
        const bool ok1 = close(clap_example::decode(dev.downloadVector<std::uint8_t>(DX, d.bytes()), type), rdx, 4 * tolerance(type), dxd);
        const bool ok2 = close(clap_example::decode(dev.downloadVector<std::uint8_t>(DW, wd.bytes()), type), rdw, 4 * tolerance(type), dwd);
        report(name + " backward", ok1 && ok2, "dx " + dxd + ", dweight " + dwd);
    } catch (const clap::DnnUnsupportedError& e) {
        skip(name + " backward", e.what());
    }
}

void checkLayerNorm(clap::IDnnBackend& backend, clap_example::Device& dev)
{
    const std::size_t rows = 6, inner = 8;
    const std::string name = "LayerNorm (preserved) Float32 [2,3,8] via ExecutionContext";
    clap::TensorDesc d(clap::DataType::Float32, {2, 3, 8});
    clap::TensorDesc sd(clap::DataType::Float32, {8});
    clap::LayerNormDesc ln;

    const auto s = backend.supports(clap::DnnCapabilityQuery::forOperation(clap::DnnOperation::LayerNormForward, d), dev.ctx);
    if (!s) { skip(name, s.reason); return; }

    const auto x = pattern(rows * inner, 2.0f, 0.9f);
    const auto sc = pattern(inner, 0.5f, 0.2f);
    const auto bi = pattern(inner, 0.3f, 1.7f);
    std::vector<float> ry(rows * inner), rmean(rows), rrstd(rows);
    refLayerNorm(x, sc, bi, rows, inner, ln.epsilon, ry, rmean, rrstd);

    void* X = dev.uploadVector(x);
    void* S = dev.uploadVector(sc);
    void* B = dev.uploadVector(bi);
    void* Y = dev.allocate(d.bytes());
    void* M = dev.allocate(rows * sizeof(float));
    void* R = dev.allocate(rows * sizeof(float));

    backend.layerNormForward(ln, d, X, sd, S, B, d, Y, M, R, dev.ctx);

    std::string yd, md, rd;
    const bool ok = close(dev.downloadVector<float>(Y, rows * inner), ry, 1.0e-4, yd) &&
                    close(dev.downloadVector<float>(M, rows), rmean, 1.0e-4, md) &&
                    close(dev.downloadVector<float>(R, rows), rrstd, 1.0e-4, rd);
    report(name + " forward", ok, "y " + yd + ", mean " + md + ", rstd " + rd);

    const auto dy = pattern(rows * inner, 1.0f, 0.5f);
    std::vector<float> rdx(rows * inner), rds(inner, 0.0f), rdb(inner, 0.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        double sum_g = 0.0, sum_gx = 0.0;
        for (std::size_t j = 0; j < inner; ++j) {
            const double xh = (x[r * inner + j] - rmean[r]) * rrstd[r];
            const double g = dy[r * inner + j] * sc[j];
            sum_g += g;
            sum_gx += g * xh;
            rds[j] += static_cast<float>(dy[r * inner + j] * xh);
            rdb[j] += dy[r * inner + j];
        }
        for (std::size_t j = 0; j < inner; ++j) {
            const double xh = (x[r * inner + j] - rmean[r]) * rrstd[r];
            const double g = dy[r * inner + j] * sc[j];
            rdx[r * inner + j] = static_cast<float>(rrstd[r] / inner * (inner * g - sum_g - xh * sum_gx));
        }
    }

    void* DY = dev.uploadVector(dy);
    void* DX = dev.allocate(d.bytes());
    void* DS = dev.allocate(sd.bytes());
    void* DB = dev.allocate(sd.bytes());
    backend.layerNormBackward(ln, d, X, d, DY, sd, S, M, R, d, DX, DS, DB, dev.ctx);

    std::string a, b2, c;
    const bool ok2 = close(dev.downloadVector<float>(DX, rows * inner), rdx, 1.0e-4, a) &&
                     close(dev.downloadVector<float>(DS, inner), rds, 1.0e-4, b2) &&
                     close(dev.downloadVector<float>(DB, inner), rdb, 1.0e-4, c);
    report(name + " backward", ok2, "dx " + a + ", dscale " + b2 + ", dbias " + c);
}

void checkSdpa(clap::IDnnBackend& backend, clap_example::Device& dev, clap::DataType type,
               std::int64_t Hq, std::int64_t Hkv, std::int64_t Sq, std::int64_t Skv,
               bool causal, bool with_mask)
{
    const std::int64_t B = 2, D = 16;
    std::string name = std::string("SDPA ") + clap::dataTypeName(type) +
        " Hq=" + std::to_string(Hq) + " Hkv=" + std::to_string(Hkv) +
        " Sq=" + std::to_string(Sq) + " Skv=" + std::to_string(Skv) +
        (causal ? " causal" : "") + (with_mask ? " +mask" : "");

    clap::TensorDesc qd(type, {B, Hq, Sq, D});
    clap::TensorDesc kd(type, {B, Hkv, Skv, D});
    clap::TensorDesc od(type, {B, Hq, Sq, D});
    const std::vector<std::int64_t> mask_dims = {B, 1, Sq, Skv};
    clap::TensorDesc md(type, mask_dims);

    clap::AttentionDesc att;
    att.causal = causal;

    const auto s = backend.supports(clap::DnnCapabilityQuery::forAttention(att, qd, kd, kd, od, with_mask ? &md : nullptr), dev.ctx);
    if (!s) { skip(name, s.reason); return; }

    const auto q = quantize(pattern(static_cast<std::size_t>(B * Hq * Sq * D), 1.0f, 0.1f), type);
    const auto k = quantize(pattern(static_cast<std::size_t>(B * Hkv * Skv * D), 1.0f, 0.6f), type);
    const auto v = quantize(pattern(static_cast<std::size_t>(B * Hkv * Skv * D), 1.0f, 1.9f), type);
    std::vector<float> mask(static_cast<std::size_t>(B * Sq * Skv));
    for (std::size_t i = 0; i < mask.size(); ++i)
        mask[i] = (i % 5 == 3) ? -1.0e4f : 0.1f * static_cast<float>(i % 3);
    mask = quantize(mask, type);

    std::vector<float> ro(static_cast<std::size_t>(B * Hq * Sq * D));
    refAttention(q, k, v, with_mask ? &mask : nullptr, mask_dims, B, Hq, Hkv, Sq, Skv, D,
                 1.0 / std::sqrt(static_cast<double>(D)), causal, ro);

    void* Q = dev.uploadVector(clap_example::encode(q, type));
    void* K = dev.uploadVector(clap_example::encode(k, type));
    void* V = dev.uploadVector(clap_example::encode(v, type));
    void* M = dev.uploadVector(clap_example::encode(mask, type));
    void* O = dev.allocate(od.bytes());

    try {
        backend.scaledDotProductAttentionForward(att, qd, Q, kd, K, kd, V,
                                                 with_mask ? &md : nullptr, with_mask ? M : nullptr,
                                                 od, O, dev.ctx);
        std::string detail;
        report(name, close(clap_example::decode(dev.downloadVector<std::uint8_t>(O, od.bytes()), type), ro, 4 * tolerance(type), detail), detail);
    } catch (const clap::DnnUnsupportedError& e) {
        skip(name, e.what());
    }
}

void checkCapabilities(clap::IDnnBackend& backend, clap_example::Device& dev)
{
    clap::TensorDesc d(clap::DataType::Float32, {2, 8});
    const auto dropout = backend.supports(clap::DnnCapabilityQuery::forOperation(clap::DnnOperation::DropoutForward, d), dev.ctx);
    std::cout << "[INFO] Dropout capability: " << (dropout ? "supported" : "unsupported") << " (" << dropout.reason << ")\n";

    clap::ExecutionContext wrong = dev.ctx;
    wrong.backend = dev.onGpu() ? clap::ExecutionBackend::CPU : clap::ExecutionBackend::CUDA;
    const auto mismatch = backend.supports(clap::DnnCapabilityQuery::forOperation(clap::DnnOperation::ActivationForward, d), wrong);
    report("Capability rejects a mismatched ExecutionContext", !mismatch, mismatch.reason);
}

}

int main()
{
    auto backend = clap::createDnnBackend();
    clap_example::Device dev(*backend);

    std::cout << "Backend: " << backend->name()
              << "  execution: " << clap::executionBackendName(backend->executionBackend())
              << "  caller stream: " << dev.ctx.stream << "\n\n";

    checkCapabilities(*backend, dev);

    for (auto type : {clap::DataType::Float32, clap::DataType::Float16, clap::DataType::BFloat16})
        checkSilu(*backend, dev, type);

    checkSoftmax(*backend, dev, clap::DataType::Float32, {2, 3, 4}, {}, -1, false);
    checkSoftmax(*backend, dev, clap::DataType::Float32, {2, 3, 4}, {}, 1, false);
    checkSoftmax(*backend, dev, clap::DataType::Float32, {2, 3, 4}, {}, 0, false);
    checkSoftmax(*backend, dev, clap::DataType::Float32, {2, 4, 3}, {12, 1, 4}, -1, false);
    checkSoftmax(*backend, dev, clap::DataType::Float32, {2, 3, 4}, {16, 5, 1}, -1, false);
    checkSoftmax(*backend, dev, clap::DataType::Float32, {4, 2, 3, 5}, {}, 2, false);
    checkSoftmax(*backend, dev, clap::DataType::Float32, {2, 3, 4}, {}, -1, true);
    checkSoftmax(*backend, dev, clap::DataType::BFloat16, {2, 3, 8}, {}, -1, false);
    checkSoftmax(*backend, dev, clap::DataType::Float16, {2, 3, 8}, {}, -1, false);
    checkSoftmaxBackward(*backend, dev, false);
    checkSoftmaxBackward(*backend, dev, true);
    checkLegacySoftmax(*backend, dev);

    checkRmsNorm(*backend, dev, clap::DataType::Float32, false);
    checkRmsNorm(*backend, dev, clap::DataType::Float32, true);
    checkRmsNorm(*backend, dev, clap::DataType::BFloat16, false);
    checkRmsNorm(*backend, dev, clap::DataType::Float16, false);

    checkLayerNorm(*backend, dev);

    for (auto type : {clap::DataType::Float32, clap::DataType::BFloat16, clap::DataType::Float16}) {
        checkSdpa(*backend, dev, type, 4, 4, 8, 8, false, false);
        checkSdpa(*backend, dev, type, 4, 4, 8, 8, true, false);
        checkSdpa(*backend, dev, type, 4, 4, 8, 12, false, true);
        checkSdpa(*backend, dev, type, 4, 2, 8, 8, true, false);
        checkSdpa(*backend, dev, type, 4, 1, 6, 10, false, false);
    }

    std::cout << "\nSummary: " << g_pass << " passed, " << g_fail << " failed, " << g_skip << " skipped (unsupported natively)\n";
    return g_fail == 0 ? 0 : 1;
}
