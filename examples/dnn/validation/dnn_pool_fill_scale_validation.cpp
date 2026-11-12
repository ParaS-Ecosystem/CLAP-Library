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

const char* poolName(clap::PoolingMode mode)
{
    return mode == clap::PoolingMode::Average ? "AvgPool(include padding)" : "AvgPool(exclude padding)";
}

struct PoolGeom {
    std::int64_t n, c, h, w, oh, ow;
};

std::int64_t divisor(const clap::PoolingDesc& p, const PoolGeom& g, std::int64_t oy, std::int64_t ox)
{
    if (p.mode == clap::PoolingMode::Average)
        return p.window_h * p.window_w;
    const std::int64_t y0 = oy * p.stride_h - p.pad_h, x0 = ox * p.stride_w - p.pad_w;
    const std::int64_t rows = std::min(y0 + p.window_h, g.h) - std::max<std::int64_t>(y0, 0);
    const std::int64_t cols = std::min(x0 + p.window_w, g.w) - std::max<std::int64_t>(x0, 0);
    return rows * cols;
}

void refPool(const clap::PoolingDesc& p, const PoolGeom& g, const std::vector<float>& x, const std::vector<float>& dy,
             std::vector<float>& y, std::vector<float>& dx)
{
    y.assign(static_cast<std::size_t>(g.n * g.c * g.oh * g.ow), 0.0f);
    dx.assign(x.size(), 0.0f);
    for (std::int64_t nc = 0; nc < g.n * g.c; ++nc) {
        for (std::int64_t oy = 0; oy < g.oh; ++oy) {
            for (std::int64_t ox = 0; ox < g.ow; ++ox) {
                const double div = static_cast<double>(divisor(p, g, oy, ox));
                const std::size_t o = static_cast<std::size_t>((nc * g.oh + oy) * g.ow + ox);
                double sum = 0.0;
                for (std::int64_t ky = 0; ky < p.window_h; ++ky) {
                    for (std::int64_t kx = 0; kx < p.window_w; ++kx) {
                        const std::int64_t iy = oy * p.stride_h - p.pad_h + ky, ix = ox * p.stride_w - p.pad_w + kx;
                        if (iy < 0 || ix < 0 || iy >= g.h || ix >= g.w)
                            continue;
                        const std::size_t i = static_cast<std::size_t>((nc * g.h + iy) * g.w + ix);
                        sum += x[i];
                        dx[i] += static_cast<float>(dy[o] / div);
                    }
                }
                y[o] = static_cast<float>(sum / div);
            }
        }
    }
}

void checkPooling(clap::IDnnBackend& backend, clap_example::Device& dev, clap::PoolingMode mode, clap::DataType type)
{
    const std::string name = std::string(poolName(mode)) + " forward+backward " + clap::dataTypeName(type);
    guarded(name, [&] {
        clap::PoolingDesc p;
        p.mode = mode;
        p.window_h = 3;
        p.window_w = 3;
        p.pad_h = 1;
        p.pad_w = 1;
        p.stride_h = 2;
        p.stride_w = 2;
        const PoolGeom g{2, 3, 5, 5, 3, 3};
        clap::TensorDesc xd(type, {g.n, g.c, g.h, g.w}), yd(type, {g.n, g.c, g.oh, g.ow});

        const auto x = quantize(pattern(xd.elements(), 2.0f, 0.3f), type);
        const auto dy = quantize(pattern(yd.elements(), 1.0f, 1.1f), type);
        std::vector<float> ref_y, ref_dx;
        refPool(p, g, x, dy, ref_y, ref_dx);

        void* d_x = dev.allocate(xd.bytes());
        void* d_y = dev.allocate(yd.bytes());
        void* d_dy = dev.allocate(yd.bytes());
        void* d_dx = dev.allocate(xd.bytes());
        const auto ex = clap_example::encode(x, type);
        const auto edy = clap_example::encode(dy, type);
        dev.upload(d_x, ex.data(), ex.size());
        dev.upload(d_dy, edy.data(), edy.size());

        backend.poolingForward(p, xd, d_x, yd, d_y, dev.ctx);
        backend.poolingBackward(p, xd, d_x, yd, d_y, yd, d_dy, xd, d_dx, dev.ctx);

        std::vector<std::uint8_t> ry(yd.bytes()), rdx(xd.bytes());
        dev.download(ry.data(), d_y, ry.size());
        dev.download(rdx.data(), d_dx, rdx.size());

        std::string dy_detail, dx_detail;
        const bool ok_y = close(clap_example::decode(ry, type), ref_y, tolerance(type), dy_detail);
        const bool ok_dx = close(clap_example::decode(rdx, type), ref_dx, tolerance(type), dx_detail);
        report(name, ok_y && ok_dx, "y " + dy_detail + ", dx " + dx_detail);
    });
}

void checkLegacyPooling(clap::IDnnBackend& backend, clap::PoolingMode mode)
{
    const std::string name = std::string("Legacy float* ") + poolName(mode);
    guarded(name, [&] {
        clap::PoolingDesc p;
        p.mode = mode;
        p.window_h = 3;
        p.window_w = 3;
        p.pad_h = 1;
        p.pad_w = 1;
        p.stride_h = 1;
        p.stride_w = 1;
        const PoolGeom g{1, 2, 4, 4, 4, 4};
        clap::TensorDesc xd({g.n, g.c, g.h, g.w}), yd({g.n, g.c, g.oh, g.ow});
        const auto x = pattern(xd.elements(), 3.0f, 0.1f);
        const std::vector<float> dy(yd.elements(), 0.0f);
        std::vector<float> ref_y, ref_dx, y(yd.elements());
        refPool(p, g, x, dy, ref_y, ref_dx);
        backend.poolingForward(p, xd, x.data(), yd, y.data());
        std::string detail;
        report(name, close(y, ref_y, 1.0e-5, detail), detail);
    });
}

const char* elementName(clap::DnnOperation op)
{
    return op == clap::DnnOperation::TensorFill ? "TensorFill" : "TensorScale";
}

void checkElement(clap::IDnnBackend& backend, clap_example::Device& dev, clap::DnnOperation op, clap::DataType type, bool strided)
{
    const std::string name = std::string(elementName(op)) + (strided ? " strided " : " packed ") + clap::dataTypeName(type);
    const std::vector<std::int64_t> dims = {2, 3, 4};
    const std::vector<std::int64_t> strides = strided ? std::vector<std::int64_t>{24, 8, 1} : std::vector<std::int64_t>{};
    clap::TensorDesc yd(type, dims, strides);

    const auto support = backend.supports(clap::DnnCapabilityQuery::forOperation(op, yd), dev.ctx);
    if (!support) {
        skip(name, support.reason);
        return;
    }

    guarded(name, [&] {
        const bool fill = op == clap::DnnOperation::TensorFill;
        const float value = fill ? -1.75f : 2.5f;
        const std::size_t span = yd.spanElements();
        const float sentinel = 9.0f;

        std::vector<float> init(span, sentinel);
        const auto vals = pattern(span, 4.0f, 0.2f);
        const auto st = yd.effectiveStrides();
        std::vector<bool> inside(span, false);
        for (std::int64_t a = 0; a < dims[0]; ++a)
            for (std::int64_t b = 0; b < dims[1]; ++b)
                for (std::int64_t c = 0; c < dims[2]; ++c) {
                    const std::size_t i = static_cast<std::size_t>(a * st[0] + b * st[1] + c * st[2]);
                    inside[i] = true;
                    init[i] = fill ? std::numeric_limits<float>::quiet_NaN() : vals[i];
                }
        init = quantize(init, type);

        std::vector<float> ref(span);
        const float q = quantize({value}, type)[0];
        for (std::size_t i = 0; i < span; ++i)
            ref[i] = !inside[i] ? init[i] : (fill ? q : init[i] * value);

        const auto raw = clap_example::encode(init, type);
        void* d_y = dev.allocate(raw.size());
        dev.upload(d_y, raw.data(), raw.size());

        if (fill)
            backend.tensorFill(yd, d_y, value, dev.ctx);
        else
            backend.tensorScale(yd, d_y, value, dev.ctx);

        std::vector<std::uint8_t> out(raw.size());
        dev.download(out.data(), d_y, out.size());
        std::string detail;
        const bool ok = close(clap_example::decode(out, type), ref, tolerance(type), detail);
        report(name, ok, detail + ", " + support.reason);
    });
}

void checkLegacyElement(clap::IDnnBackend& backend, clap::DnnOperation op)
{
    const std::string name = std::string("Legacy float* ") + elementName(op);
    guarded(name, [&] {
        clap::TensorDesc yd({2, 3, 4});
        auto y = pattern(yd.elements(), 3.0f, 0.5f);
        std::vector<float> ref(y.size());
        if (op == clap::DnnOperation::TensorFill) {
            std::fill(ref.begin(), ref.end(), 0.25f);
            backend.tensorFill(yd, y.data(), 0.25f);
        } else {
            for (std::size_t i = 0; i < y.size(); ++i)
                ref[i] = y[i] * -0.5f;
            backend.tensorScale(yd, y.data(), -0.5f);
        }
        std::string detail;
        report(name, close(y, ref, 1.0e-6, detail), detail);
    });
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
        checkPooling(*backend, dev, clap::PoolingMode::Average, type);
        checkPooling(*backend, dev, clap::PoolingMode::AverageExcludePadding, type);
    }
    checkLegacyPooling(*backend, clap::PoolingMode::Average);
    checkLegacyPooling(*backend, clap::PoolingMode::AverageExcludePadding);

    for (auto op : {clap::DnnOperation::TensorFill, clap::DnnOperation::TensorScale}) {
        for (auto type : {clap::DataType::Float32, clap::DataType::Float16, clap::DataType::BFloat16}) {
            checkElement(*backend, dev, op, type, false);
            checkElement(*backend, dev, op, type, true);
        }
        checkLegacyElement(*backend, op);
    }

    std::cout << "\nSummary: " << g_pass << " passed, " << g_fail << " failed, " << g_skip << " skipped (unsupported natively)\n";
    return g_fail == 0 ? 0 : 1;
}
