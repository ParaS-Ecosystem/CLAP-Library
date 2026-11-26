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

struct Case {
    std::string label;
    std::int64_t n, c, h, w, k, groups, r, s;
    std::int64_t pad, stride, dilation;

    std::int64_t oh() const { return (h + 2 * pad - dilation * (r - 1) - 1) / stride + 1; }
    std::int64_t ow() const { return (w + 2 * pad - dilation * (s - 1) - 1) / stride + 1; }
    std::int64_t cg() const { return c / groups; }
    std::int64_t kg() const { return k / groups; }

    clap::ConvolutionDesc conv() const
    {
        clap::ConvolutionDesc d;
        d.pad_h = d.pad_w = pad;
        d.stride_h = d.stride_w = stride;
        d.dilation_h = d.dilation_w = dilation;
        d.groups = groups;
        return d;
    }

    std::string name() const
    {
        return label + " N" + std::to_string(n) + " C" + std::to_string(c) + " K" + std::to_string(k) +
               " G" + std::to_string(groups) + " " + std::to_string(r) + "x" + std::to_string(s) +
               " pad" + std::to_string(pad) + " stride" + std::to_string(stride) + " dil" + std::to_string(dilation);
    }
};

std::vector<float> pattern(std::size_t n, float phase, clap::DataType type)
{
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = std::sin(0.37f * static_cast<float>(i) + phase) + 0.25f * std::cos(0.11f * static_cast<float>(i));
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

enum class Direction { Forward, BackwardData, BackwardWeights };

std::vector<float> reference(const Case& t, Direction dir, const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>& dy)
{
    const std::int64_t OH = t.oh(), OW = t.ow(), CG = t.cg(), KG = t.kg();
    std::vector<double> acc;
    if (dir == Direction::Forward)
        acc.assign(static_cast<std::size_t>(t.n * t.k * OH * OW), 0.0);
    else if (dir == Direction::BackwardData)
        acc.assign(static_cast<std::size_t>(t.n * t.c * t.h * t.w), 0.0);
    else
        acc.assign(static_cast<std::size_t>(t.k * CG * t.r * t.s), 0.0);

    for (std::int64_t n = 0; n < t.n; ++n)
        for (std::int64_t k = 0; k < t.k; ++k) {
            const std::int64_t g = k / KG;
            for (std::int64_t oh = 0; oh < OH; ++oh)
                for (std::int64_t ow = 0; ow < OW; ++ow) {
                    const std::size_t yo = static_cast<std::size_t>(((n * t.k + k) * OH + oh) * OW + ow);
                    for (std::int64_t cc = 0; cc < CG; ++cc) {
                        const std::int64_t ci = g * CG + cc;
                        for (std::int64_t r = 0; r < t.r; ++r) {
                            const std::int64_t ih = oh * t.stride - t.pad + r * t.dilation;
                            if (ih < 0 || ih >= t.h)
                                continue;
                            for (std::int64_t s = 0; s < t.s; ++s) {
                                const std::int64_t iw = ow * t.stride - t.pad + s * t.dilation;
                                if (iw < 0 || iw >= t.w)
                                    continue;
                                const std::size_t xo = static_cast<std::size_t>(((n * t.c + ci) * t.h + ih) * t.w + iw);
                                const std::size_t wo = static_cast<std::size_t>(((k * CG + cc) * t.r + r) * t.s + s);
                                if (dir == Direction::Forward)
                                    acc[yo] += static_cast<double>(x[xo]) * w[wo];
                                else if (dir == Direction::BackwardData)
                                    acc[xo] += static_cast<double>(dy[yo]) * w[wo];
                                else
                                    acc[wo] += static_cast<double>(dy[yo]) * x[xo];
                            }
                        }
                    }
                }
        }
    return std::vector<float>(acc.begin(), acc.end());
}

double tolerance(clap::DataType type)
{
    if (type == clap::DataType::Float16)
        return 1.0e-2;
    if (type == clap::DataType::BFloat16)
        return 5.0e-2;
    return 1.0e-4;
}

bool close(const std::vector<float>& got, const std::vector<float>& want, clap::DataType type, std::string& detail)
{
    double scale = 1.0e-6, err = 0.0;
    for (float v : want)
        scale = std::max(scale, static_cast<double>(std::fabs(v)));
    for (std::size_t i = 0; i < want.size(); ++i)
        err = std::max(err, std::fabs(static_cast<double>(got[i]) - want[i]));
    detail = "max rel err " + std::to_string(err / scale);
    return got.size() == want.size() && err / scale <= tolerance(type);
}

clap::DnnCapabilityQuery query(clap::DnnOperation op, const Case& t, const clap::TensorDesc& xd, const clap::TensorDesc& wd, const clap::TensorDesc& yd)
{
    if (t.groups == t.c && t.groups > 1)
        return clap::DnnCapabilityQuery::forDepthwiseConvolution(op, t.conv(), xd, wd, yd);
    return clap::DnnCapabilityQuery::forConvolution(op, t.conv(), xd, wd, yd);
}

void run(clap::IDnnBackend& backend, clap_example::Device& dev, clap::DataType type, const Case& t, Direction dir)
{
    static const char* const dir_name[] = {"forward", "backward data", "backward weights"};
    static const clap::DnnOperation ops[] = {
        clap::DnnOperation::ConvolutionForward,
        clap::DnnOperation::ConvolutionBackwardData,
        clap::DnnOperation::ConvolutionBackwardWeights,
    };
    const int d = static_cast<int>(dir);
    const std::string name = t.name() + " " + clap::dataTypeName(type) + " " + dir_name[d];

    const clap::TensorDesc xd(type, {t.n, t.c, t.h, t.w});
    const clap::TensorDesc wd(type, {t.k, t.cg(), t.r, t.s});
    const clap::TensorDesc yd(type, {t.n, t.k, t.oh(), t.ow()});

    const auto support = backend.supports(query(ops[d], t, xd, wd, yd), dev.ctx);
    if (!support) {
        skip(name, support.reason);
        return;
    }

    guarded(name, [&] {
        const auto x = pattern(xd.elements(), 0.3f, type);
        const auto w = pattern(wd.elements(), 1.1f, type);
        const auto dy = pattern(yd.elements(), 2.7f, type);
        const auto want = reference(t, dir, x, w, dy);
        std::vector<float> got;
        if (dir == Direction::Forward) {
            void* X = place(dev, x, type);
            void* W = place(dev, w, type);
            void* Y = dev.allocate(yd.bytes());
            backend.convolutionForward(xd, X, wd, W, t.conv(), yd, Y, dev.ctx);
            got = fetch(dev, Y, yd.elements(), type);
        } else if (dir == Direction::BackwardData) {
            void* DY = place(dev, dy, type);
            void* W = place(dev, w, type);
            void* DX = dev.allocate(xd.bytes());
            backend.convolutionBackwardData(yd, DY, wd, W, t.conv(), xd, DX, dev.ctx);
            got = fetch(dev, DX, xd.elements(), type);
        } else {
            void* X = place(dev, x, type);
            void* DY = place(dev, dy, type);
            void* DW = dev.allocate(wd.bytes());
            backend.convolutionBackwardWeights(xd, X, yd, DY, t.conv(), wd, DW, dev.ctx);
            got = fetch(dev, DW, wd.elements(), type);
        }
        std::string detail;
        report(name, close(got, want, type, detail), detail + ", " + support.reason);
    });
}

void checkLegacy(clap::IDnnBackend& backend)
{
    const Case t{"legacy float* depthwise", 1, 4, 6, 6, 4, 4, 3, 3, 1, 1, 1};
    const std::string name = "Legacy float* " + t.name() + " forward + backward weights";
    guarded(name, [&] {
        const clap::DataType f32 = clap::DataType::Float32;
        const clap::TensorDesc xd(f32, {t.n, t.c, t.h, t.w});
        const clap::TensorDesc wd(f32, {t.k, t.cg(), t.r, t.s});
        const clap::TensorDesc yd(f32, {t.n, t.k, t.oh(), t.ow()});
        const auto x = pattern(xd.elements(), 0.3f, f32);
        const auto w = pattern(wd.elements(), 1.1f, f32);
        const auto dy = pattern(yd.elements(), 2.7f, f32);
        std::vector<float> y(yd.elements()), dw(wd.elements());
        backend.convolutionForward(xd, x.data(), wd, w.data(), t.conv(), yd, y.data());
        backend.convolutionBackwardWeights(xd, x.data(), yd, dy.data(), t.conv(), wd, dw.data());
        std::string a, b;
        const bool ok = close(y, reference(t, Direction::Forward, x, w, dy), f32, a) &&
                        close(dw, reference(t, Direction::BackwardWeights, x, w, dy), f32, b);
        report(name, ok, "y " + a + (b.empty() ? "" : ", dw " + b));
    });
}

void checkRejects(clap::IDnnBackend& backend, clap_example::Device& dev, const std::string& name,
                  const clap::TensorDesc& xd, const clap::TensorDesc& wd, const clap::TensorDesc& yd, std::int64_t groups)
{
    clap::ConvolutionDesc c;
    c.pad_h = c.pad_w = 1;
    c.groups = groups;
    const auto support = backend.supports(
        clap::DnnCapabilityQuery::forConvolution(clap::DnnOperation::ConvolutionForward, c, xd, wd, yd), dev.ctx);
    bool threw = false;
    std::string what;
    try {
        void* X = dev.allocate(xd.bytes());
        void* W = dev.allocate(wd.bytes());
        void* Y = dev.allocate(yd.bytes());
        backend.convolutionForward(xd, X, wd, W, c, yd, Y, dev.ctx);
    } catch (const clap::DnnUnsupportedError& e) {
        what = std::string("unsupported instead of invalid: ") + e.what();
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    report(name, threw && !support, what);
}

void checkTransposeRejectsGroups(clap::IDnnBackend& backend, clap_example::Device& dev)
{
    const clap::DataType f32 = clap::DataType::Float32;
    const clap::TensorDesc xd(f32, {1, 4, 5, 5});
    const clap::TensorDesc wd(f32, {4, 2, 3, 3});
    const clap::TensorDesc yd(f32, {1, 4, 5, 5});
    clap::ConvolutionDesc c;
    c.pad_h = c.pad_w = 1;
    c.groups = 2;
    const auto support = backend.supports(
        clap::DnnCapabilityQuery::forConvolution(clap::DnnOperation::ConvolutionTransposeForward, c, xd, wd, yd), dev.ctx);
    bool unsupported = false;
    std::string what;
    try {
        void* X = dev.allocate(xd.bytes());
        void* W = dev.allocate(wd.bytes());
        void* Y = dev.allocate(yd.bytes());
        backend.convolutionTransposeForward(xd, X, wd, W, c, yd, Y, dev.ctx);
    } catch (const clap::DnnUnsupportedError& e) {
        unsupported = true;
        what = e.what();
    } catch (const std::exception& e) {
        what = e.what();
    }
    report("convolutionTransposeForward reports groups > 1 as unsupported", unsupported && !support, what);
}

}

int main()
{
    auto backend = clap::createDnnBackend();
    clap_example::Device dev(*backend);

    std::cout << "Backend: " << backend->name()
              << "  execution: " << clap::executionBackendName(backend->executionBackend())
              << "  caller stream: " << dev.ctx.stream << "\n\n";

    const std::vector<Case> cases = {
        {"grouped", 2, 4, 7, 6, 6, 2, 3, 3, 1, 1, 1},
        {"grouped", 1, 6, 9, 8, 9, 3, 3, 3, 1, 2, 1},
        {"depthwise", 2, 8, 6, 5, 8, 8, 3, 3, 1, 1, 1},
        {"depthwise x2", 1, 4, 7, 7, 8, 4, 5, 5, 2, 1, 1},
        {"depthwise dilated", 1, 3, 9, 9, 3, 3, 3, 3, 2, 1, 2},
        {"ungrouped", 1, 3, 5, 5, 4, 1, 3, 3, 1, 1, 1},
    };

    for (auto type : {clap::DataType::Float32, clap::DataType::Float16, clap::DataType::BFloat16})
        for (const auto& t : cases)
            for (auto dir : {Direction::Forward, Direction::BackwardData, Direction::BackwardWeights})
                run(*backend, dev, type, t, dir);

    checkLegacy(*backend);

    const clap::DataType f32 = clap::DataType::Float32;
    checkRejects(*backend, dev, "rejects groups = 0",
                 clap::TensorDesc(f32, {1, 4, 5, 5}), clap::TensorDesc(f32, {4, 4, 3, 3}), clap::TensorDesc(f32, {1, 4, 5, 5}), 0);
    checkRejects(*backend, dev, "rejects input channels not divisible by groups",
                 clap::TensorDesc(f32, {1, 5, 5, 5}), clap::TensorDesc(f32, {4, 2, 3, 3}), clap::TensorDesc(f32, {1, 4, 5, 5}), 2);
    checkRejects(*backend, dev, "rejects output channels not divisible by groups",
                 clap::TensorDesc(f32, {1, 4, 5, 5}), clap::TensorDesc(f32, {3, 2, 3, 3}), clap::TensorDesc(f32, {1, 3, 5, 5}), 2);
    checkRejects(*backend, dev, "rejects weights not shaped [K, C / groups, R, S]",
                 clap::TensorDesc(f32, {1, 4, 5, 5}), clap::TensorDesc(f32, {4, 4, 3, 3}), clap::TensorDesc(f32, {1, 4, 5, 5}), 2);
    checkTransposeRejectsGroups(*backend, dev);

    std::cout << "\nSummary: " << g_pass << " passed, " << g_fail << " failed, " << g_skip << " skipped (unsupported natively)\n";
    return g_fail == 0 ? 0 : 1;
}
