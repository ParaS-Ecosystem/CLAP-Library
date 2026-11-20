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

#include <array>
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

using Shape = std::array<std::int64_t, 4>;
using Strides = std::array<std::int64_t, 4>;

Strides nchwPacked(const Shape& s)
{
    return {s[1] * s[2] * s[3], s[2] * s[3], s[3], 1};
}

Strides nhwcPacked(const Shape& s)
{
    return {s[2] * s[3] * s[1], 1, s[3] * s[1], s[1]};
}

std::size_t span(const Shape& s, const Strides& st)
{
    std::size_t n = 1;
    for (int k = 0; k < 4; ++k)
        n += static_cast<std::size_t>(s[k] - 1) * static_cast<std::size_t>(st[k]);
    return n;
}

std::vector<float> pattern(std::size_t n, clap::DataType type)
{
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = 4.0f * std::sin(0.61f * static_cast<float>(i) + 0.3f) + 0.01f * static_cast<float>(i % 97);
    return clap_example::decode(clap_example::encode(v, type), type);
}

void* place(clap_example::Device& dev, const std::vector<float>& host, clap::DataType type)
{
    const auto raw = clap_example::encode(host, type);
    void* p = dev.allocate(raw.size());
    dev.upload(p, raw.data(), raw.size());
    return p;
}

std::vector<float> fetch(clap_example::Device& dev, void* p, std::size_t elements, clap::DataType type)
{
    std::vector<std::uint8_t> raw(elements * clap::dataTypeSize(type));
    dev.download(raw.data(), p, raw.size());
    return clap_example::decode(raw, type);
}

bool compareLogical(const Shape& s,
                    const std::vector<float>& x, const Strides& xs,
                    const std::vector<float>& y, const Strides& ys,
                    std::string& detail)
{
    std::size_t checked = 0;
    for (std::int64_t n = 0; n < s[0]; ++n)
        for (std::int64_t c = 0; c < s[1]; ++c)
            for (std::int64_t h = 0; h < s[2]; ++h)
                for (std::int64_t w = 0; w < s[3]; ++w) {
                    const std::size_t xo = static_cast<std::size_t>(n * xs[0] + c * xs[1] + h * xs[2] + w * xs[3]);
                    const std::size_t yo = static_cast<std::size_t>(n * ys[0] + c * ys[1] + h * ys[2] + w * ys[3]);
                    if (x[xo] != y[yo]) {
                        detail = "mismatch at n=" + std::to_string(n) + " c=" + std::to_string(c) +
                                 " h=" + std::to_string(h) + " w=" + std::to_string(w);
                        return false;
                    }
                    ++checked;
                }
    detail = std::to_string(checked) + " elements bit-exact";
    return true;
}

clap::TensorDesc nchwDesc(clap::DataType type, const Shape& s, const Strides& st)
{
    return clap::TensorDesc(type, {s[0], s[1], s[2], s[3]}, {st[0], st[1], st[2], st[3]});
}

clap::TensorDesc nhwcDesc(clap::DataType type, const Shape& s)
{
    clap::TensorDesc d(type, {s[0], s[2], s[3], s[1]});
    d.layout = clap::TensorLayout::NHWC;
    return d;
}

std::string shapeName(const Shape& s)
{
    return "N" + std::to_string(s[0]) + " C" + std::to_string(s[1]) + " H" + std::to_string(s[2]) + " W" + std::to_string(s[3]);
}

void runNchwToNhwc(clap::IDnnBackend& backend, clap_example::Device& dev, clap::DataType type, const Shape& s)
{
    const std::string name = "nchwToNhwc " + shapeName(s) + " " + clap::dataTypeName(type);
    const auto xd = nchwDesc(type, s, nchwPacked(s));
    const auto yd = nhwcDesc(type, s);
    const auto support = backend.supports(clap::DnnCapabilityQuery::forNchwToNhwc(xd, yd), dev.ctx);
    if (!support) {
        skip(name, support.reason);
        return;
    }
    guarded(name, [&] {
        const auto x = pattern(xd.elements(), type);
        void* dx = place(dev, x, type);
        void* dy = dev.allocate(yd.bytes());
        backend.nchwToNhwc(xd, dx, yd, dy, dev.ctx);
        const auto y = fetch(dev, dy, yd.elements(), type);
        std::string detail;
        report(name, compareLogical(s, x, nchwPacked(s), y, nhwcPacked(s), detail), detail + ", " + support.reason);
    });
}

void runNhwcToNchw(clap::IDnnBackend& backend, clap_example::Device& dev, clap::DataType type, const Shape& s)
{
    const std::string name = "nhwcToNchw " + shapeName(s) + " " + clap::dataTypeName(type);
    const auto xd = nhwcDesc(type, s);
    const auto yd = nchwDesc(type, s, nchwPacked(s));
    const auto support = backend.supports(clap::DnnCapabilityQuery::forNhwcToNchw(xd, yd), dev.ctx);
    if (!support) {
        skip(name, support.reason);
        return;
    }
    guarded(name, [&] {
        const auto x = pattern(xd.elements(), type);
        void* dx = place(dev, x, type);
        void* dy = dev.allocate(yd.bytes());
        backend.nhwcToNchw(xd, dx, yd, dy, dev.ctx);
        const auto y = fetch(dev, dy, yd.elements(), type);
        std::string detail;
        report(name, compareLogical(s, x, nhwcPacked(s), y, nchwPacked(s), detail), detail + ", " + support.reason);
    });
}

void runRoundTrip(clap::IDnnBackend& backend, clap_example::Device& dev, clap::DataType type, const Shape& s)
{
    const std::string name = "round trip NCHW->NHWC->NCHW " + shapeName(s) + " " + clap::dataTypeName(type);
    const auto xd = nchwDesc(type, s, nchwPacked(s));
    const auto md = nhwcDesc(type, s);
    const auto support = backend.supports(clap::DnnCapabilityQuery::forNchwToNhwc(xd, md), dev.ctx);
    if (!support) {
        skip(name, support.reason);
        return;
    }
    guarded(name, [&] {
        const auto x = pattern(xd.elements(), type);
        void* dx = place(dev, x, type);
        void* dm = dev.allocate(md.bytes());
        void* dz = dev.allocate(xd.bytes());
        backend.nchwToNhwc(xd, dx, md, dm, dev.ctx);
        backend.nhwcToNchw(md, dm, xd, dz, dev.ctx);
        const auto z = fetch(dev, dz, xd.elements(), type);
        bool same = z.size() == x.size();
        for (std::size_t i = 0; same && i < x.size(); ++i)
            same = z[i] == x[i];
        report(name, same, same ? "bit-exact" : "round trip changed the data");
    });
}

void runGeneral(clap::IDnnBackend& backend, clap_example::Device& dev, clap::DataType type,
                const std::string& label, const Shape& s, const Strides& xs, const Strides& ys)
{
    const std::string name = "tensorLayoutTransform " + label + " " + shapeName(s) + " " + clap::dataTypeName(type);
    const auto xd = nchwDesc(type, s, xs);
    const auto yd = nchwDesc(type, s, ys);
    const auto support = backend.supports(clap::DnnCapabilityQuery::forLayoutTransform(xd, yd), dev.ctx);
    if (!support) {
        skip(name, support.reason);
        return;
    }
    guarded(name, [&] {
        const auto x = pattern(span(s, xs), type);
        void* dx = place(dev, x, type);
        void* dy = dev.allocate(yd.bytes());
        backend.tensorLayoutTransform(xd, dx, yd, dy, dev.ctx);
        const auto y = fetch(dev, dy, span(s, ys), type);
        std::string detail;
        report(name, compareLogical(s, x, xs, y, ys, detail), detail + ", " + support.reason);
    });
}

void runTaggedGeneral(clap::IDnnBackend& backend, clap_example::Device& dev, clap::DataType type, const Shape& s)
{
    const std::string name = "tensorLayoutTransform NHWC-tagged x -> NCHW " + shapeName(s) + " " + clap::dataTypeName(type);
    const auto xd = nhwcDesc(type, s);
    const auto yd = nchwDesc(type, s, nchwPacked(s));
    const auto support = backend.supports(clap::DnnCapabilityQuery::forLayoutTransform(xd, yd), dev.ctx);
    if (!support) {
        skip(name, support.reason);
        return;
    }
    guarded(name, [&] {
        const auto x = pattern(xd.elements(), type);
        void* dx = place(dev, x, type);
        void* dy = dev.allocate(yd.bytes());
        backend.tensorLayoutTransform(xd, dx, yd, dy, dev.ctx);
        const auto y = fetch(dev, dy, yd.elements(), type);
        std::string detail;
        report(name, compareLogical(s, x, nhwcPacked(s), y, nchwPacked(s), detail), detail + ", " + support.reason);
    });
}

void checkLegacy(clap::IDnnBackend& backend)
{
    guarded("Legacy float* nchwToNhwc + nhwcToNchw", [&] {
        const Shape s = {2, 3, 4, 5};
        const auto xd = nchwDesc(clap::DataType::Float32, s, nchwPacked(s));
        const auto md = nhwcDesc(clap::DataType::Float32, s);
        const auto x = pattern(xd.elements(), clap::DataType::Float32);
        std::vector<float> m(md.elements()), z(xd.elements());
        backend.nchwToNhwc(xd, x.data(), md, m.data());
        backend.nhwcToNchw(md, m.data(), xd, z.data());
        std::string detail;
        bool ok = compareLogical(s, x, nchwPacked(s), m, nhwcPacked(s), detail) && z == x;
        report("Legacy float* nchwToNhwc + nhwcToNchw", ok, detail);
    });
    guarded("Legacy float* tensorLayoutTransform", [&] {
        const Shape s = {1, 3, 2, 5};
        const Strides ys = nhwcPacked(s);
        const auto xd = nchwDesc(clap::DataType::Float32, s, nchwPacked(s));
        const auto yd = nchwDesc(clap::DataType::Float32, s, ys);
        const auto x = pattern(xd.elements(), clap::DataType::Float32);
        std::vector<float> y(yd.spanElements());
        backend.tensorLayoutTransform(xd, x.data(), yd, y.data());
        std::string detail;
        report("Legacy float* tensorLayoutTransform", compareLogical(s, x, nchwPacked(s), y, ys, detail), detail);
    });
}

void checkRejects(clap::IDnnBackend& backend, clap_example::Device& dev, const std::string& name,
                  const clap::TensorDesc& xd, const clap::TensorDesc& yd, int routine)
{
    const auto query = routine == 1 ? clap::DnnCapabilityQuery::forNchwToNhwc(xd, yd)
                     : routine == 2 ? clap::DnnCapabilityQuery::forNhwcToNchw(xd, yd)
                     : clap::DnnCapabilityQuery::forLayoutTransform(xd, yd);
    const auto support = backend.supports(query, dev.ctx);
    bool threw = false;
    std::string what;
    try {
        void* x = dev.allocate(xd.bytes());
        void* y = dev.allocate(yd.bytes());
        if (routine == 1)
            backend.nchwToNhwc(xd, x, yd, y, dev.ctx);
        else if (routine == 2)
            backend.nhwcToNchw(xd, x, yd, y, dev.ctx);
        else
            backend.tensorLayoutTransform(xd, x, yd, y, dev.ctx);
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    report(name, !support && threw, what);
}

void checkSharedBuffer(clap::IDnnBackend& backend, clap_example::Device& dev)
{
    const Shape s = {1, 3, 4, 5};
    const auto xd = nchwDesc(clap::DataType::Float32, s, nchwPacked(s));
    const auto yd = nhwcDesc(clap::DataType::Float32, s);
    bool threw = false;
    std::string what;
    try {
        void* x = dev.allocate(xd.bytes());
        backend.nchwToNhwc(xd, x, yd, x, dev.ctx);
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    report("nchwToNhwc rejects x and y sharing one buffer", threw, what);
}

}

int main()
{
    auto backend = clap::createDnnBackend();
    clap_example::Device dev(*backend);

    std::cout << "Backend: " << backend->name()
              << "  execution: " << clap::executionBackendName(backend->executionBackend())
              << "  caller stream: " << dev.ctx.stream << "\n\n";

    const std::vector<Shape> shapes = {
        {2, 3, 5, 7},
        {1, 1, 1, 1},
        {3, 16, 4, 9},
        {2, 64, 7, 3},
    };

    for (auto type : {clap::DataType::Float32, clap::DataType::Float16, clap::DataType::BFloat16}) {
        for (const auto& s : shapes) {
            runNchwToNhwc(*backend, dev, type, s);
            runNhwcToNchw(*backend, dev, type, s);
            runRoundTrip(*backend, dev, type, s);
        }
        const Shape g = {2, 3, 4, 5};
        runGeneral(*backend, dev, type, "NCHW -> NHWC strides", g, nchwPacked(g), nhwcPacked(g));
        runGeneral(*backend, dev, type, "NCHW -> padded rows", g, nchwPacked(g), {3 * 4 * 8, 4 * 8, 8, 1});
        runGeneral(*backend, dev, type, "CHWN -> NCHW", g, {1, 4 * 5 * 2, 5 * 2, 2}, nchwPacked(g));
        runTaggedGeneral(*backend, dev, type, {2, 3, 5, 7});
    }

    checkLegacy(*backend);

    const clap::DataType f32 = clap::DataType::Float32;
    checkRejects(*backend, dev, "nchwToNhwc rejects rank-3 tensors",
                 clap::TensorDesc(f32, {3, 4, 5}), clap::TensorDesc(f32, {4, 5, 3}), 1);
    checkRejects(*backend, dev, "nchwToNhwc rejects y given in NCHW order (H != W)",
                 nchwDesc(f32, {2, 3, 4, 5}, nchwPacked({2, 3, 4, 5})), clap::TensorDesc(f32, {2, 3, 4, 5}), 1);
    checkRejects(*backend, dev, "nhwcToNchw rejects mismatched batch",
                 nhwcDesc(f32, {2, 3, 4, 5}), nchwDesc(f32, {3, 3, 4, 5}, nchwPacked({3, 3, 4, 5})), 2);
    checkRejects(*backend, dev, "tensorLayoutTransform rejects datatype change",
                 nchwDesc(f32, {1, 3, 4, 5}, nchwPacked({1, 3, 4, 5})),
                 nchwDesc(clap::DataType::Float16, {1, 3, 4, 5}, nhwcPacked({1, 3, 4, 5})), 0);
    checkRejects(*backend, dev, "tensorLayoutTransform rejects overlapping output strides",
                 nchwDesc(f32, {1, 3, 4, 5}, nchwPacked({1, 3, 4, 5})),
                 nchwDesc(f32, {1, 3, 4, 5}, {60, 1, 5, 1}), 0);
    checkSharedBuffer(*backend, dev);

    std::cout << "\nSummary: " << g_pass << " passed, " << g_fail << " failed, " << g_skip << " skipped (unsupported natively)\n";
    return g_fail == 0 ? 0 : 1;
}
