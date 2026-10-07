#include <dnn.hh>
#include "clap/examples/dnn_example_device.hh"

#include <cmath>
#include <iostream>
#include <vector>

// Scaled dot-product attention with causal masking and grouped-query
// attention (4 query heads sharing 2 key/value heads).  The datatype is the
// first of Float32 / BFloat16 / Float16 that the selected backend supports
// natively; unsupported configurations are reported, never emulated.
int main() {
    auto backend = clap::createDnnBackend();
    clap_example::Device device(*backend);

    const std::int64_t B = 1, Hq = 4, Hkv = 2, S = 8, D = 16;

    clap::AttentionDesc attention;
    attention.causal = true;
    attention.num_query_heads = static_cast<int>(Hq);
    attention.num_kv_heads = static_cast<int>(Hkv);

    std::cout << "Backend: " << backend->name() << "\n";

    for (auto type : {clap::DataType::Float32, clap::DataType::BFloat16, clap::DataType::Float16}) {
        clap::TensorDesc q_desc(type, {B, Hq, S, D});
        clap::TensorDesc kv_desc(type, {B, Hkv, S, D});
        clap::TensorDesc o_desc(type, {B, Hq, S, D});

        const auto support = backend->supports(
            clap::DnnCapabilityQuery::forAttention(attention, q_desc, kv_desc, kv_desc, o_desc),
            device.ctx);
        std::cout << clap::dataTypeName(type) << ": " << (support ? "supported" : "unsupported")
                  << " (" << support.reason << ")\n";
        if (!support)
            continue;

        std::vector<float> q(B * Hq * S * D), k(B * Hkv * S * D), v(B * Hkv * S * D);
        for (std::size_t i = 0; i < q.size(); ++i) q[i] = std::sin(0.1f * static_cast<float>(i));
        for (std::size_t i = 0; i < k.size(); ++i) k[i] = std::cos(0.07f * static_cast<float>(i));
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.01f * static_cast<float>(i % 50);

        void* d_q = device.uploadVector(clap_example::encode(q, type));
        void* d_k = device.uploadVector(clap_example::encode(k, type));
        void* d_v = device.uploadVector(clap_example::encode(v, type));
        void* d_o = device.allocate(o_desc.bytes());

        backend->scaledDotProductAttentionForward(
            attention,
            q_desc, d_q,
            kv_desc, d_k,
            kv_desc, d_v,
            nullptr, nullptr,
            o_desc, d_o,
            device.ctx);

        const auto raw = device.downloadVector<std::uint8_t>(d_o, o_desc.bytes());
        const auto o = clap_example::decode(raw, type);

        std::cout << "O[head 0, query 0..1, 0..3]: ";
        for (int s = 0; s < 2; ++s)
            for (int j = 0; j < 4; ++j)
                std::cout << o[static_cast<std::size_t>(s * D + j)] << ' ';
        std::cout << '\n';
        return 0;
    }
}
