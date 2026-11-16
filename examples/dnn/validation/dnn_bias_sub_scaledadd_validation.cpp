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
#include <functional>
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

void guarded(const std::string& name, const std::function<void()>& body)
{
    try {
        body();
    } catch (const clap::DnnUnsupportedError& e) {
        skip(name, e.what());
    } catch (const std::exception& e) {
        report(name, false, e.what());
    }
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

std::vector<std::size_t> offsets(const clap::TensorDesc& d)
{
    const auto st = d.effectiveStrides();
    std::vector<std::size_t> out;
    out.reserve(d.elements());
    std::vector<std::int64_t> idx(d.rank(), 0);
    for (std::size_t n = 0; n < d.elements(); ++n) {
        std::int64_t off = 0;
        for (std::size_t k = 0; k < d.rank(); ++k)
            off += idx[k] * st[k];
        out.push_back(static_cast<std::size_t>(off));
        for (std::size_t k = d.rank(); k-- > 0;) {
            if (++idx[k] < d.dims[k])
                break;
            idx[k] = 0;
        }
    }
    return out;
}

std::vector<std::int64_t> logical(const clap::TensorDesc& d, std::size_t n)
{
    std::vector<std::int64_t> idx(d.rank(), 0);
    for (std::size_t k = d.rank(); k-- > 0;) {
        idx[k] = static_cast<std::int64_t>(n % static_cast<std::size_t>(d.dims[k]));
        n /= static_cast<std::size_t>(d.dims[k]);
    }
    return idx;
}

std::size_t broadcastOffset(const clap::TensorDesc& b, const std::vector<std::int64_t>& idx)
{
    const auto st = b.effectiveStrides();
    if (b.rank() == 1)
        return static_cast<std::size_t>(idx[1] * st[0]);
    std::int64_t off = 0;
    for (std::size_t k = 0; k < b.rank(); ++k)
        off += (b.dims[k] == 1 ? 0 : idx[k]) * st[k];
    return static_cast<std::size_t>(off);
}

struct Buffer {
    std::vector<float> host;
    void* dev = nullptr;
};

Buffer place(clap_example::Device& dev, const clap::TensorDesc& d, float scale, float phase)
{
    Buffer b;
    b.host = quantize(pattern(d.spanElements(), scale, phase), d.type);
    const auto raw = clap_example::encode(b.host, d.type);
    b.dev = dev.allocate(raw.size());
    dev.upload(b.dev, raw.data(), raw.size());
    return b;
}

std::vector<float> fetch(clap_example::Device& dev, const clap::TensorDesc& d, void* p)
{
    std::vector<std::uint8_t> raw(d.bytes());
    dev.download(raw.data(), p, raw.size());
    return clap_example::decode(raw, d.type);
}

std::vector<float> gather(const clap::TensorDesc& d, const std::vector<float>& span)
{
    std::vector<float> out;
    for (std::size_t o : offsets(d))
        out.push_back(span[o]);
    return out;
}

enum class Kind { BiasAdd, Subtract, ScaledAdd };

const char* kindName(Kind k)
{
    return k == Kind::BiasAdd ? "BiasAdd" : k == Kind::Subtract ? "TensorSubtract" : "TensorScaledAdd";
}

clap::DnnOperation kindOp(Kind k)
{
    return k == Kind::BiasAdd ? clap::DnnOperation::BiasAdd
         : k == Kind::Subtract ? clap::DnnOperation::TensorSubtract
         : clap::DnnOperation::TensorScaledAdd;
}

struct Case {
    std::string label;
    std::vector<std::int64_t> dims;
    std::vector<std::int64_t> strides;
    std::vector<std::int64_t> b_dims;
    bool in_place = false;
    float alpha = 1.0f;
    float beta = 1.0f;
};

void run(clap::IDnnBackend& backend, clap_example::Device& dev, Kind kind, clap::DataType type, const Case& c)
{
    const std::string name = std::string(kindName(kind)) + " " + c.label + " " + clap::dataTypeName(type);
    clap::TensorDesc ad(type, c.dims, c.strides);
    clap::TensorDesc bd(type, c.b_dims.empty() ? c.dims : c.b_dims, c.b_dims.empty() ? c.strides : std::vector<std::int64_t>{});
    clap::TensorDesc yd = ad;

    const auto support = backend.supports(clap::DnnCapabilityQuery::forBinary(kindOp(kind), ad, bd, yd), dev.ctx);
    if (!support) {
        skip(name, support.reason);
        return;
    }

    guarded(name, [&] {
        Buffer a = place(dev, ad, 3.0f, 0.2f);
        Buffer b = place(dev, bd, 2.0f, 1.3f);
        void* y = c.in_place ? a.dev : dev.allocate(yd.bytes());

        std::vector<float> ref;
        const auto ao = offsets(ad);
        for (std::size_t n = 0; n < ao.size(); ++n) {
            const auto idx = logical(ad, n);
            const double av = a.host[ao[n]];
            const double bv = b.host[broadcastOffset(bd, idx)];
            double r = 0.0;
            if (kind == Kind::BiasAdd)
                r = av + bv;
            else if (kind == Kind::Subtract)
                r = av - bv;
            else
                r = static_cast<double>(c.alpha) * av + static_cast<double>(c.beta) * bv;
            ref.push_back(static_cast<float>(r));
        }

        if (kind == Kind::BiasAdd)
            backend.biasAdd(ad, a.dev, bd, b.dev, yd, y, dev.ctx);
        else if (kind == Kind::Subtract)
            backend.tensorSubtract(ad, a.dev, bd, b.dev, yd, y, dev.ctx);
        else
            backend.tensorScaledAdd(ad, a.dev, bd, b.dev, yd, y, c.alpha, c.beta, dev.ctx);

        const auto got = gather(yd, fetch(dev, yd, y));
        std::string detail;
        const bool ok = close(got, ref, tolerance(type), detail);
        report(name, ok, detail + ", " + support.reason);
    });
}

void checkLegacy(clap::IDnnBackend& backend, Kind kind)
{
    const std::string name = std::string("Legacy float* ") + kindName(kind);
    guarded(name, [&] {
        clap::TensorDesc ad({2, 3, 2, 2});
        clap::TensorDesc bd = kind == Kind::BiasAdd ? clap::TensorDesc({3}) : ad;
        const auto a = pattern(ad.elements(), 3.0f, 0.4f);
        const auto b = pattern(bd.elements(), 2.0f, 0.9f);
        std::vector<float> y(ad.elements()), ref(ad.elements());
        for (std::size_t i = 0; i < ref.size(); ++i) {
            if (kind == Kind::BiasAdd)
                ref[i] = a[i] + b[(i / 4) % 3];
            else if (kind == Kind::Subtract)
                ref[i] = a[i] - b[i];
            else
                ref[i] = 0.75f * a[i] - 1.5f * b[i];
        }
        if (kind == Kind::BiasAdd)
            backend.biasAdd(ad, a.data(), bd, b.data(), ad, y.data());
        else if (kind == Kind::Subtract)
            backend.tensorSubtract(ad, a.data(), bd, b.data(), ad, y.data());
        else
            backend.tensorScaledAdd(ad, a.data(), bd, b.data(), ad, y.data(), 0.75f, -1.5f);
        std::string detail;
        report(name, close(y, ref, 1.0e-5, detail), detail);
    });
}

void checkRejects(clap::IDnnBackend& backend, clap_example::Device& dev, const std::string& name, Kind kind,
                  const clap::TensorDesc& ad, const clap::TensorDesc& bd, const clap::TensorDesc& yd)
{
    const auto support = backend.supports(clap::DnnCapabilityQuery::forBinary(kindOp(kind), ad, bd, yd), dev.ctx);
    bool threw = false;
    std::string what;
    try {
        void* a = dev.allocate(ad.bytes());
        void* b = dev.allocate(bd.bytes());
        void* y = dev.allocate(yd.bytes());
        if (kind == Kind::BiasAdd)
            backend.biasAdd(ad, a, bd, b, yd, y, dev.ctx);
        else
            backend.tensorScaledAdd(ad, a, bd, b, yd, y, 1.0f, 1.0f, dev.ctx);
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    report(name, !support && threw, what);
}

}

int main()
{
    auto backend = clap::createDnnBackend();
    clap_example::Device dev(*backend);

    std::cout << "Backend: " << backend->name()
              << "  execution: " << clap::executionBackendName(backend->executionBackend())
              << "  caller stream: " << dev.ctx.stream << "\n\n";

    const std::vector<Case> bias_cases = {
        {"NCHW bias[C]", {2, 3, 4, 5}, {}, {3}},
        {"NCHW bias[1,C,1,1] in-place", {2, 3, 4, 5}, {}, {1, 3, 1, 1}, true},
        {"NHWC-strided bias[C]", {2, 3, 4, 5}, {60, 1, 15, 3}, {3}},
        {"2D [N,C] bias[C]", {4, 6}, {}, {6}},
    };
    const std::vector<Case> sub_cases = {
        {"packed", {2, 3, 4, 5}, {}, {}},
        {"strided", {2, 3, 4}, {24, 8, 1}, {}},
        {"broadcast b[1,C,1,1]", {2, 3, 4, 5}, {}, {1, 3, 1, 1}},
    };
    const std::vector<Case> scaled_cases = {
        {"packed alpha=2 beta=-0.5", {2, 3, 4, 5}, {}, {}, false, 2.0f, -0.5f},
        {"in-place y=a alpha=1 beta=0.25", {2, 3, 4, 5}, {}, {}, true, 1.0f, 0.25f},
        {"strided alpha=-1.5 beta=3", {2, 3, 4}, {24, 8, 1}, {}, false, -1.5f, 3.0f},
    };

    for (auto type : {clap::DataType::Float32, clap::DataType::Float16, clap::DataType::BFloat16}) {
        for (const auto& c : bias_cases)
            run(*backend, dev, Kind::BiasAdd, type, c);
        for (const auto& c : sub_cases)
            run(*backend, dev, Kind::Subtract, type, c);
        for (const auto& c : scaled_cases)
            run(*backend, dev, Kind::ScaledAdd, type, c);
    }

    checkLegacy(*backend, Kind::BiasAdd);
    checkLegacy(*backend, Kind::Subtract);
    checkLegacy(*backend, Kind::ScaledAdd);

    checkRejects(*backend, dev, "BiasAdd rejects bias length != C", Kind::BiasAdd,
                 clap::TensorDesc(clap::DataType::Float32, {2, 3, 4, 4}), clap::TensorDesc(clap::DataType::Float32, {4}),
                 clap::TensorDesc(clap::DataType::Float32, {2, 3, 4, 4}));
    checkRejects(*backend, dev, "TensorScaledAdd rejects mismatched shapes", Kind::ScaledAdd,
                 clap::TensorDesc(clap::DataType::Float32, {2, 3, 4}), clap::TensorDesc(clap::DataType::Float32, {1, 3, 1}),
                 clap::TensorDesc(clap::DataType::Float32, {2, 3, 4}));

    std::cout << "\nSummary: " << g_pass << " passed, " << g_fail << " failed, " << g_skip << " skipped (unsupported natively)\n";
    return g_fail == 0 ? 0 : 1;
}
