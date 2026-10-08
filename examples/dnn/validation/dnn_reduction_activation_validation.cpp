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

#include <dnn.hh>
#include "clap/examples/dnn_example_device.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
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
        if (std::isnan(got[i])) {
            detail = "NaN at " + std::to_string(i);
            return false;
        }
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

const char* modeName(clap::ActivationMode mode)
{
    return mode == clap::ActivationMode::ELU ? "ELU" : "ClippedReLU";
}

void refActivation(clap::ActivationMode mode, double alpha, const std::vector<float>& x, const std::vector<float>& dy,
                   std::vector<float>& y, std::vector<float>& dx)
{
    y.resize(x.size());
    dx.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double v = x[i];
        if (mode == clap::ActivationMode::ELU) {
            y[i] = static_cast<float>(v > 0.0 ? v : alpha * (std::exp(v) - 1.0));
            dx[i] = static_cast<float>(v > 0.0 ? dy[i] : dy[i] * alpha * std::exp(v));
        } else {
            y[i] = static_cast<float>(std::min(std::max(v, 0.0), alpha));
            dx[i] = (v > 0.0 && v < alpha) ? dy[i] : 0.0f;
        }
    }
}

void checkActivation(clap::IDnnBackend& backend, clap_example::Device& dev, clap::ActivationMode mode, double alpha, clap::DataType type)
{
    const std::string name = std::string(modeName(mode)) + " alpha=" + std::to_string(alpha).substr(0, 4) + " " + clap::dataTypeName(type) + " [2,3,7]";
    clap::TensorDesc d(type, {2, 3, 7});
    clap::ActivationDesc act;
    act.mode = mode;
    act.alpha = alpha;

    const auto s = backend.supports(clap::DnnCapabilityQuery::forActivation(clap::DnnOperation::ActivationForward, act, d), dev.ctx);
    if (!s) { skip(name, s.reason); return; }

    const auto x = quantize(pattern(42, 4.0f, 0.11f), type);
    const auto dy = quantize(pattern(42, 1.0f, 0.7f), type);
    std::vector<float> ry, rdx;
    refActivation(mode, alpha, x, dy, ry, rdx);

    void* X = dev.uploadVector(clap_example::encode(x, type));
    void* DY = dev.uploadVector(clap_example::encode(dy, type));
    void* Y = dev.allocate(d.bytes());
    void* DX = dev.allocate(d.bytes());

    try {
        backend.activationForward(act, d, X, d, Y, dev.ctx);
        std::string detail;
        report(name + " forward", close(clap_example::decode(dev.downloadVector<std::uint8_t>(Y, d.bytes()), type), ry, 4 * tolerance(type), detail), detail);
    } catch (const clap::DnnUnsupportedError& e) {
        skip(name + " forward", e.what());
    }

    try {
        backend.activationBackward(act, d, X, d, DY, d, DX, dev.ctx);
        std::string detail;
        report(name + " backward", close(clap_example::decode(dev.downloadVector<std::uint8_t>(DX, d.bytes()), type), rdx, 4 * tolerance(type), detail), detail);
    } catch (const clap::DnnUnsupportedError& e) {
        skip(name + " backward", e.what());
    }
}

void checkLegacyActivation(clap::IDnnBackend& backend, clap::ActivationMode mode, double alpha)
{
    const std::string name = std::string("Legacy float* ") + modeName(mode) + " [1,2,3,4]";
    clap::TensorDesc d({1, 2, 3, 4});
    clap::ActivationDesc act;
    act.mode = mode;
    act.alpha = alpha;
    const auto x = pattern(24, 3.0f, 0.23f);
    const std::vector<float> dy(24, 1.0f);
    std::vector<float> ry, rdx, y(24), dx(24);
    refActivation(mode, alpha, x, dy, ry, rdx);
    try {
        backend.activationForward(act, d, x.data(), d, y.data());
        backend.activationBackward(act, d, x.data(), d, dy.data(), d, dx.data());
        std::string d1, d2;
        const bool ok1 = close(y, ry, 1.0e-4, d1);
        const bool ok2 = close(dx, rdx, 1.0e-4, d2);
        report(name, ok1 && ok2, d1 + ", " + d2);
    } catch (const clap::DnnUnsupportedError& e) {
        skip(name, e.what());
    }
}

enum class Red { Max, Min, Mean, Norm1, Norm2 };

const char* redName(Red r)
{
    switch (r) {
        case Red::Max:   return "TensorReduceMax";
        case Red::Min:   return "TensorReduceMin";
        case Red::Mean:  return "TensorReduceMean";
        case Red::Norm1: return "TensorReduceNorm1";
        case Red::Norm2: return "TensorReduceNorm2";
    }
    return "";
}

clap::DnnOperation redOp(Red r)
{
    switch (r) {
        case Red::Max:   return clap::DnnOperation::TensorReduceMax;
        case Red::Min:   return clap::DnnOperation::TensorReduceMin;
        case Red::Mean:  return clap::DnnOperation::TensorReduceMean;
        case Red::Norm1: return clap::DnnOperation::TensorReduceNorm1;
        case Red::Norm2: return clap::DnnOperation::TensorReduceNorm2;
    }
    return clap::DnnOperation::TensorReduceSum;
}

std::vector<float> refReduce(Red r, const std::vector<float>& x, const std::vector<std::int64_t>& xd, const std::vector<std::int64_t>& yd)
{
    const std::size_t rank = xd.size();
    std::size_t ny = 1;
    for (auto v : yd) ny *= static_cast<std::size_t>(v);
    std::vector<double> acc(ny, 0.0);
    std::vector<std::size_t> cnt(ny, 0);
    std::vector<bool> seen(ny, false);
    std::vector<std::int64_t> idx(rank, 0);
    for (std::size_t n = 0; n < x.size(); ++n) {
        std::size_t o = 0;
        for (std::size_t k = 0; k < rank; ++k)
            o = o * static_cast<std::size_t>(yd[k]) + static_cast<std::size_t>(yd[k] == 1 ? 0 : idx[k]);
        const double v = x[n];
        switch (r) {
            case Red::Max:   acc[o] = seen[o] ? std::max(acc[o], v) : v; break;
            case Red::Min:   acc[o] = seen[o] ? std::min(acc[o], v) : v; break;
            case Red::Mean:  acc[o] += v; break;
            case Red::Norm1: acc[o] += std::fabs(v); break;
            case Red::Norm2: acc[o] += v * v; break;
        }
        seen[o] = true;
        cnt[o]++;
        for (std::size_t k = rank; k-- > 0;) {
            if (++idx[k] < xd[k]) break;
            idx[k] = 0;
        }
    }
    std::vector<float> out(ny);
    for (std::size_t i = 0; i < ny; ++i) {
        double v = acc[i];
        if (r == Red::Mean) v /= static_cast<double>(cnt[i]);
        if (r == Red::Norm2) v = std::sqrt(v);
        out[i] = static_cast<float>(v);
    }
    return out;
}

void runReduce(clap::IDnnBackend& backend, Red r, const clap::TensorDesc& xd, const void* x, const clap::TensorDesc& yd, void* y, const clap::ExecutionContext& ctx)
{
    switch (r) {
        case Red::Max:   backend.tensorReduceMax(xd, x, yd, y, ctx); break;
        case Red::Min:   backend.tensorReduceMin(xd, x, yd, y, ctx); break;
        case Red::Mean:  backend.tensorReduceMean(xd, x, yd, y, ctx); break;
        case Red::Norm1: backend.tensorReduceNorm1(xd, x, yd, y, ctx); break;
        case Red::Norm2: backend.tensorReduceNorm2(xd, x, yd, y, ctx); break;
    }
}

void runLegacyReduce(clap::IDnnBackend& backend, Red r, const clap::TensorDesc& xd, const float* x, const clap::TensorDesc& yd, float* y)
{
    switch (r) {
        case Red::Max:   backend.tensorReduceMax(xd, x, yd, y); break;
        case Red::Min:   backend.tensorReduceMin(xd, x, yd, y); break;
        case Red::Mean:  backend.tensorReduceMean(xd, x, yd, y); break;
        case Red::Norm1: backend.tensorReduceNorm1(xd, x, yd, y); break;
        case Red::Norm2: backend.tensorReduceNorm2(xd, x, yd, y); break;
    }
}

std::string shapeText(const std::vector<std::int64_t>& d)
{
    std::string s = "[";
    for (std::size_t i = 0; i < d.size(); ++i)
        s += (i ? "," : "") + std::to_string(d[i]);
    return s + "]";
}

void checkReduce(clap::IDnnBackend& backend, clap_example::Device& dev, Red r, clap::DataType type,
                 const std::vector<std::int64_t>& xdims, const std::vector<std::int64_t>& ydims)
{
    const std::string name = std::string(redName(r)) + " " + clap::dataTypeName(type) + " " + shapeText(xdims) + "->" + shapeText(ydims);
    clap::TensorDesc xd(type, xdims), yd(type, ydims);

    const auto s = backend.supports(clap::DnnCapabilityQuery::forReduction(redOp(r), xd, yd), dev.ctx);
    if (!s) { skip(name, s.reason); return; }

    std::size_t n = 1;
    for (auto v : xdims) n *= static_cast<std::size_t>(v);
    const auto x = quantize(pattern(n, 2.5f, 0.3f), type);
    const auto ref = refReduce(r, x, xdims, ydims);

    void* X = dev.uploadVector(clap_example::encode(x, type));
    void* Y = dev.allocate(yd.bytes());
    try {
        runReduce(backend, r, xd, X, yd, Y, dev.ctx);
        std::string detail;
        const double tol = type == clap::DataType::Float32 ? 1.0e-4 : 8 * tolerance(type);
        report(name, close(clap_example::decode(dev.downloadVector<std::uint8_t>(Y, yd.bytes()), type), ref, tol, detail), detail);
    } catch (const clap::DnnUnsupportedError& e) {
        skip(name, e.what());
    }
}

void checkLegacyReduce(clap::IDnnBackend& backend, Red r)
{
    const std::vector<std::int64_t> xdims = {2, 3, 4, 5}, ydims = {2, 1, 4, 1};
    const std::string name = std::string("Legacy float* ") + redName(r) + " " + shapeText(xdims) + "->" + shapeText(ydims);
    clap::TensorDesc xd(xdims), yd(ydims);
    const auto x = pattern(120, 2.0f, 0.4f);
    const auto ref = refReduce(r, x, xdims, ydims);
    std::vector<float> y(ref.size());
    try {
        runLegacyReduce(backend, r, xd, x.data(), yd, y.data());
        std::string detail;
        report(name, close(y, ref, 1.0e-4, detail), detail);
    } catch (const clap::DnnUnsupportedError& e) {
        skip(name, e.what());
    }
}

}

int main()
{
    auto backend = clap::createDnnBackend();
    clap_example::Device dev(*backend);

    std::cout << "Backend: " << backend->name()
              << "  execution: " << clap::executionBackendName(backend->executionBackend())
              << "  caller stream: " << dev.ctx.stream << "\n\n";

    for (auto type : {clap::DataType::Float32, clap::DataType::Float16, clap::DataType::BFloat16}) {
        checkActivation(*backend, dev, clap::ActivationMode::ELU, 1.0, type);
        checkActivation(*backend, dev, clap::ActivationMode::ClippedReLU, 2.5, type);
    }
    checkActivation(*backend, dev, clap::ActivationMode::ELU, 0.5, clap::DataType::Float32);
    checkActivation(*backend, dev, clap::ActivationMode::ClippedReLU, 6.0, clap::DataType::Float32);
    checkLegacyActivation(*backend, clap::ActivationMode::ELU, 1.0);
    checkLegacyActivation(*backend, clap::ActivationMode::ClippedReLU, 2.0);

    for (Red r : {Red::Max, Red::Min, Red::Mean, Red::Norm1, Red::Norm2}) {
        checkReduce(*backend, dev, r, clap::DataType::Float32, {2, 3, 4, 5}, {2, 1, 4, 1});
        checkReduce(*backend, dev, r, clap::DataType::Float32, {2, 3, 4, 5}, {1, 1, 1, 1});
        checkReduce(*backend, dev, r, clap::DataType::Float32, {2, 3, 4, 5}, {2, 3, 4, 1});
        checkReduce(*backend, dev, r, clap::DataType::Float32, {4, 6, 8, 2}, {1, 6, 1, 2});
        checkReduce(*backend, dev, r, clap::DataType::Float16, {2, 3, 4, 5}, {2, 1, 4, 1});
        checkReduce(*backend, dev, r, clap::DataType::BFloat16, {2, 3, 4, 5}, {2, 1, 4, 1});
        checkLegacyReduce(*backend, r);
    }

    std::cout << "\nSummary: " << g_pass << " passed, " << g_fail << " failed, " << g_skip << " skipped (unsupported natively)\n";
    return g_fail == 0 ? 0 : 1;
}
