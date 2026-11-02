#include "gemm_select.hpp"

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <string>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace clap {
namespace rocblas_gemm_select {
namespace {

// -----------------------------------------------------------------------------
// Private hipBLASLt ABI definitions.
//
// These definitions match the ROCm 7.2.2 declarations verified in
// /opt/rocm-7.2.2/include/hipblaslt/hipblaslt.h.
// No hipBLASLt header or link-time dependency is required.
// -----------------------------------------------------------------------------

using LtHandle = void*;
using LtMatmulDesc = void*;
using LtMatrixLayout = void*;
using LtPreference = void*;

struct LtMatmulAlgo {
    uint8_t data[16];
    size_t max_workspace_bytes;
};

struct LtHeuristicResult {
    LtMatmulAlgo algo;
    size_t workspaceSize;
    int32_t state;
    float wavesCount;
    int32_t reserved[4];
};

static_assert(sizeof(LtMatmulAlgo) == 24,
              "Unexpected hipBLASLt algorithm ABI size");

static_assert(sizeof(LtHeuristicResult) == 56,
              "Unexpected hipBLASLt heuristic-result ABI size");

// ROCm 7.2.2 enum values verified from installed headers.
constexpr int kHipblasLtSuccess = 0;

constexpr int kHipDataTypeF32  = 0;
constexpr int kHipDataTypeBF16 = 14;

constexpr int kHipComputeF32 = 2;

constexpr int kHipOperationN = 111;
constexpr int kHipOperationT = 112;

constexpr int kLtDescTransA = 0;
constexpr int kLtDescTransB = 1;
constexpr int kLtDescPointerMode = 13;

constexpr int kLtPointerModeHost = 0;
constexpr int kLtPreferenceMaxWorkspace = 1;

constexpr int kHipMemcpyDeviceToDevice = 3;

// -----------------------------------------------------------------------------
// Dynamically loaded HIP runtime and hipBLASLt functions.
// -----------------------------------------------------------------------------

struct DynamicApi {

    void* hip_library = nullptr;
    void* lt_library = nullptr;

    std::mutex load_mutex;
    bool attempted = false;
    bool loaded = false;
    std::string error;

    // HIP runtime functions.
    using HipMallocFn =
        int (*)(void**, size_t);

    using HipFreeFn =
        int (*)(void*);

    using HipMemcpyAsyncFn =
        int (*)(void*, const void*, size_t, int, void*);

    using HipMemsetAsyncFn =
        int (*)(void*, int, size_t, void*);

    using HipStreamIsCapturingFn =
        int (*)(void*, int*);

    using HipStreamSynchronizeFn =
        int (*)(void*);

    using HipEventCreateFn =
        int (*)(void**);

    using HipEventRecordFn =
        int (*)(void*, void*);

    using HipEventSynchronizeFn =
        int (*)(void*);

    using HipEventElapsedTimeFn =
        int (*)(float*, void*, void*);

    using HipEventDestroyFn =
        int (*)(void*);

    HipMallocFn hip_malloc = nullptr;
    HipFreeFn hip_free = nullptr;
    HipMemcpyAsyncFn hip_memcpy_async = nullptr;
    HipMemsetAsyncFn hip_memset_async = nullptr;
    HipStreamIsCapturingFn hip_stream_is_capturing = nullptr;
    HipStreamSynchronizeFn hip_stream_synchronize = nullptr;
    HipEventCreateFn hip_event_create = nullptr;
    HipEventRecordFn hip_event_record = nullptr;
    HipEventSynchronizeFn hip_event_synchronize = nullptr;
    HipEventElapsedTimeFn hip_event_elapsed_time = nullptr;
    HipEventDestroyFn hip_event_destroy = nullptr;

    // hipBLASLt functions.
    using LtCreateFn =
        int (*)(LtHandle*);

    using LtDestroyFn =
        int (*)(LtHandle);

    using LtMatmulDescCreateFn =
        int (*)(LtMatmulDesc*, int, int);

    using LtMatmulDescDestroyFn =
        int (*)(LtMatmulDesc);

    using LtMatmulDescSetAttributeFn =
        int (*)(LtMatmulDesc, int, const void*, size_t);

    using LtMatrixLayoutCreateFn =
        int (*)(LtMatrixLayout*, int, uint64_t, uint64_t, int64_t);

    using LtMatrixLayoutDestroyFn =
        int (*)(LtMatrixLayout);

    using LtPreferenceCreateFn =
        int (*)(LtPreference*);

    using LtPreferenceDestroyFn =
        int (*)(LtPreference);

    using LtPreferenceSetAttributeFn =
        int (*)(LtPreference, int, const void*, size_t);

    using LtAlgoGetHeuristicFn =
        int (*)(LtHandle, LtMatmulDesc,
                LtMatrixLayout, LtMatrixLayout,
                LtMatrixLayout, LtMatrixLayout,
                LtPreference, int,
                LtHeuristicResult*, int*);

    using LtMatmulFn =
        int (*)(LtHandle, LtMatmulDesc,
                const void*,
                const void*, LtMatrixLayout,
                const void*, LtMatrixLayout,
                const void*,
                const void*, LtMatrixLayout,
                void*, LtMatrixLayout,
                const LtMatmulAlgo*,
                void*, size_t, void*);

    LtCreateFn lt_create = nullptr;
    LtDestroyFn lt_destroy = nullptr;

    LtMatmulDescCreateFn lt_desc_create = nullptr;
    LtMatmulDescDestroyFn lt_desc_destroy = nullptr;
    LtMatmulDescSetAttributeFn lt_desc_set_attribute = nullptr;

    LtMatrixLayoutCreateFn lt_layout_create = nullptr;
    LtMatrixLayoutDestroyFn lt_layout_destroy = nullptr;

    LtPreferenceCreateFn lt_preference_create = nullptr;
    LtPreferenceDestroyFn lt_preference_destroy = nullptr;
    LtPreferenceSetAttributeFn lt_preference_set_attribute = nullptr;

    LtAlgoGetHeuristicFn lt_algo_get_heuristic = nullptr;
    LtMatmulFn lt_matmul = nullptr;

    template <typename T>
    bool load_symbol(void* library, T& function, const char* name) {
        function = reinterpret_cast<T>(dlsym(library, name));

        if (!function) {
            error = std::string("Missing dynamic symbol: ") + name;
            return false;
        }

        return true;
    }

    void* open_library(const char* primary, const char* fallback) {
        void* handle = dlopen(primary, RTLD_NOW | RTLD_LOCAL);

        if (!handle && fallback) {
            handle = dlopen(fallback, RTLD_NOW | RTLD_LOCAL);
        }

        return handle;
    }

    bool load() {

        std::lock_guard<std::mutex> lock(load_mutex);

        if (attempted) {
            return loaded;
        }

        attempted = true;

        hip_library = open_library(
            "libamdhip64.so",
            "libamdhip64.so.7");

        if (!hip_library) {
            error = "Unable to load libamdhip64.so";
            return false;
        }

        lt_library = open_library(
            "libhipblaslt.so",
            "libhipblaslt.so.1");

        if (!lt_library) {
            error = "Unable to load libhipblaslt.so";
            return false;
        }

#define LOAD_HIP(member, symbol) \
        if (!load_symbol(hip_library, member, symbol)) return false

#define LOAD_LT(member, symbol) \
        if (!load_symbol(lt_library, member, symbol)) return false

        // HIP runtime.
        LOAD_HIP(hip_malloc, "hipMalloc");
        LOAD_HIP(hip_free, "hipFree");
        LOAD_HIP(hip_memcpy_async, "hipMemcpyAsync");
        LOAD_HIP(hip_memset_async, "hipMemsetAsync");
        LOAD_HIP(hip_stream_is_capturing, "hipStreamIsCapturing");
        LOAD_HIP(hip_stream_synchronize, "hipStreamSynchronize");
        LOAD_HIP(hip_event_create, "hipEventCreate");
        LOAD_HIP(hip_event_record, "hipEventRecord");
        LOAD_HIP(hip_event_synchronize, "hipEventSynchronize");
        LOAD_HIP(hip_event_elapsed_time, "hipEventElapsedTime");
        LOAD_HIP(hip_event_destroy, "hipEventDestroy");

        // hipBLASLt.
        LOAD_LT(lt_create, "hipblasLtCreate");
        LOAD_LT(lt_destroy, "hipblasLtDestroy");

        LOAD_LT(lt_desc_create, "hipblasLtMatmulDescCreate");
        LOAD_LT(lt_desc_destroy, "hipblasLtMatmulDescDestroy");
        LOAD_LT(lt_desc_set_attribute, "hipblasLtMatmulDescSetAttribute");

        LOAD_LT(lt_layout_create, "hipblasLtMatrixLayoutCreate");
        LOAD_LT(lt_layout_destroy, "hipblasLtMatrixLayoutDestroy");

        LOAD_LT(lt_preference_create, "hipblasLtMatmulPreferenceCreate");
        LOAD_LT(lt_preference_destroy, "hipblasLtMatmulPreferenceDestroy");
        LOAD_LT(lt_preference_set_attribute,
                "hipblasLtMatmulPreferenceSetAttribute");

        LOAD_LT(lt_algo_get_heuristic, "hipblasLtMatmulAlgoGetHeuristic");
        LOAD_LT(lt_matmul, "hipblasLtMatmul");

#undef LOAD_HIP
#undef LOAD_LT

        loaded = true;
        return true;
    }
};

// One process-wide dynamic API instance.
// Libraries remain loaded for the lifetime of the process.

DynamicApi& dynamic_api() {
    static DynamicApi api;
    return api;
}


// -----------------------------------------------------------------------------
// GEMM backend selection configuration.
// -----------------------------------------------------------------------------

enum class GemmPrecision : uint8_t {
    FP32,
    BF16
};

enum class GemmMode : uint8_t {
    Auto,
    ForceRocblas,
    ForceHipblaslt
};

GemmMode get_gemm_mode() {
    const char* value = std::getenv("CLAP_ROCM_GEMM");

    if (!value || !*value) {
        return GemmMode::Auto;
    }

    std::string mode(value);

    std::transform(
        mode.begin(), mode.end(), mode.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (mode == "rocblas") {
        return GemmMode::ForceRocblas;
    }

    if (mode == "hipblaslt") {
        return GemmMode::ForceHipblaslt;
    }

    return GemmMode::Auto;
}

bool debug_enabled() {
    const char* value = std::getenv("CLAP_ROCM_GEMM_DEBUG");

    if (!value) {
        return false;
    }

    return std::string(value) == "1" ||
           std::string(value) == "true" ||
           std::string(value) == "yes" ||
           std::string(value) == "on";
}

void debug_log(const char* message) {
    if (debug_enabled()) {
        std::fprintf(stderr, "[CLAP ROCm GEMM] %s\n", message);
    }
}

// -----------------------------------------------------------------------------
// Shape classification.
//
// Dimensions up to 16 are classified exactly.
// Larger dimensions use two classes per octave:
//   [2^p, 2^(p+0.5))
//   [2^(p+0.5), 2^(p+1))
//
// k remains exact in the key because it affects algorithm suitability.
// -----------------------------------------------------------------------------

int shape_bucket(int dimension) {
    if (dimension <= 16) {
        return dimension;
    }

    const int bucket =
        static_cast<int>(std::floor(
            2.0 * std::log2(static_cast<double>(dimension))));

    // Keep bucket IDs separate from exact dimensions 1 through 16.
    return 32 + bucket;
}

struct ShapeKey {
    GemmPrecision precision;
    int m_bucket;
    int n_bucket;
    int k;
    int trans_a;
    int trans_b;

    bool operator==(const ShapeKey& other) const {
        return precision == other.precision &&
               m_bucket == other.m_bucket &&
               n_bucket == other.n_bucket &&
               k == other.k &&
               trans_a == other.trans_a &&
               trans_b == other.trans_b;
    }
};

struct ShapeKeyHash {
    size_t operator()(const ShapeKey& key) const {
        size_t seed = 0;

        const auto combine = [&seed](size_t value) {
            seed ^= value + 0x9e3779b9U + (seed << 6) + (seed >> 2);
        };

        combine(std::hash<int>{}(static_cast<int>(key.precision)));
        combine(std::hash<int>{}(key.m_bucket));
        combine(std::hash<int>{}(key.n_bucket));
        combine(std::hash<int>{}(key.k));
        combine(std::hash<int>{}(key.trans_a));
        combine(std::hash<int>{}(key.trans_b));

        return seed;
    }
};

ShapeKey make_shape_key(
    GemmPrecision precision,
    int m,
    int n,
    int k,
    int trans_a,
    int trans_b) {

    return ShapeKey{
        precision,
        shape_bucket(m),
        shape_bucket(n),
        k,
        trans_a,
        trans_b
    };
}

// -----------------------------------------------------------------------------
// Per-shape-class decision cache.
//
// The heuristic algorithm and measured winner will be stored here later.
// A cache entry is created only after a valid calibration/selection.
// -----------------------------------------------------------------------------

struct ShapeDecision {
    bool ready = false;
    bool calibrated = false;
    bool use_hipblaslt = false;
    LtMatmulAlgo algo{};
    size_t workspace_size = 0;
};

std::mutex& decision_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<ShapeKey, ShapeDecision, ShapeKeyHash>& decision_cache() {
    static std::unordered_map<ShapeKey, ShapeDecision, ShapeKeyHash> cache;
    return cache;
}


// -----------------------------------------------------------------------------
// Normalized GEMM parameters.
//
// hipBLASLt uses column-major matrix layouts. Row-major CLAP calls are
// converted to the equivalent transposed column-major operation.
// -----------------------------------------------------------------------------

struct NormalizedGemm {
    int m, n, k;
    int lda, ldb, ldc;
    rocblas_operation trans_a, trans_b;
    const void* A;
    const void* B;
};

NormalizedGemm normalize_gemm(
    Layout layout,
    Transpose transA,
    Transpose transB,
    int m, int n, int k,
    const void* A, int lda,
    const void* B, int ldb,
    int ldc) {

    NormalizedGemm p{
        m, n, k, lda, ldb, ldc,
        to_rocblas_trans(transA),
        to_rocblas_trans(transB),
        A, B
    };

    if (layout == Layout::RowMajor) {
        std::swap(p.m, p.n);
        std::swap(p.trans_a, p.trans_b);
        std::swap(p.lda, p.ldb);
        std::swap(p.A, p.B);
    }

    return p;
}

int to_lt_operation(rocblas_operation op) {
    return op == rocblas_operation_none
        ? kHipOperationN
        : kHipOperationT;
}

// -----------------------------------------------------------------------------
// hipBLASLt handle.
//
// Keep the handle alive for the lifetime of the process.
// -----------------------------------------------------------------------------

LtHandle get_lt_handle(DynamicApi& api) {
    static std::mutex mutex;
    static LtHandle handle = nullptr;

    std::lock_guard<std::mutex> lock(mutex);

    if (!handle) {
        const int status = api.lt_create(&handle);

        if (status != kHipblasLtSuccess || !handle) {
            throw std::runtime_error(
                "hipblasLtCreate failed, status=" +
                std::to_string(status));
        }
    }

    return handle;
}

// -----------------------------------------------------------------------------
// RAII management of hipBLASLt descriptors.
// -----------------------------------------------------------------------------

struct LtResources {
    DynamicApi* api = nullptr;

    LtMatmulDesc desc = nullptr;
    LtMatrixLayout layout_a = nullptr;
    LtMatrixLayout layout_b = nullptr;
    LtMatrixLayout layout_c = nullptr;
    LtMatrixLayout layout_d = nullptr;
    LtPreference preference = nullptr;

    explicit LtResources(DynamicApi& dynamic_api)
        : api(&dynamic_api) {}

    LtResources(const LtResources&) = delete;
    LtResources& operator=(const LtResources&) = delete;

    ~LtResources() {
        if (!api) {
            return;
        }

        if (preference) {
            api->lt_preference_destroy(preference);
        }

        if (layout_d) {
            api->lt_layout_destroy(layout_d);
        }

        if (layout_c) {
            api->lt_layout_destroy(layout_c);
        }

        if (layout_b) {
            api->lt_layout_destroy(layout_b);
        }

        if (layout_a) {
            api->lt_layout_destroy(layout_a);
        }

        if (desc) {
            api->lt_desc_destroy(desc);
        }
    }
};

bool create_lt_resources(
    DynamicApi& api,
    const NormalizedGemm& p,
    bool bf16,
    LtResources& resources,
    std::string& error) {

    const int input_type =
        bf16 ? kHipDataTypeBF16 : kHipDataTypeF32;

    const int output_type = input_type;

    int status = api.lt_desc_create(
        &resources.desc,
        kHipComputeF32,
        kHipDataTypeF32);

    if (status != kHipblasLtSuccess) {
        error = "hipblasLtMatmulDescCreate failed, status=" +
                std::to_string(status);
        return false;
    }

    const int trans_a = to_lt_operation(p.trans_a);
    const int trans_b = to_lt_operation(p.trans_b);
    const int pointer_mode = kLtPointerModeHost;

    status = api.lt_desc_set_attribute(
        resources.desc, kLtDescTransA,
        &trans_a, sizeof(trans_a));

    if (status != kHipblasLtSuccess) {
        error = "Setting hipBLASLt TRANSA failed, status=" +
                std::to_string(status);
        return false;
    }

    status = api.lt_desc_set_attribute(
        resources.desc, kLtDescTransB,
        &trans_b, sizeof(trans_b));

    if (status != kHipblasLtSuccess) {
        error = "Setting hipBLASLt TRANSB failed, status=" +
                std::to_string(status);
        return false;
    }

    status = api.lt_desc_set_attribute(
        resources.desc, kLtDescPointerMode,
        &pointer_mode, sizeof(pointer_mode));

    if (status != kHipblasLtSuccess) {
        error = "Setting hipBLASLt pointer mode failed, status=" +
                std::to_string(status);
        return false;
    }

    const uint64_t a_rows =
        p.trans_a == rocblas_operation_none ? p.m : p.k;
    const uint64_t a_cols =
        p.trans_a == rocblas_operation_none ? p.k : p.m;

    const uint64_t b_rows =
        p.trans_b == rocblas_operation_none ? p.k : p.n;
    const uint64_t b_cols =
        p.trans_b == rocblas_operation_none ? p.n : p.k;

    status = api.lt_layout_create(
        &resources.layout_a, input_type,
        a_rows, a_cols, p.lda);

    if (status != kHipblasLtSuccess) {
        error = "Creating hipBLASLt A layout failed, status=" +
                std::to_string(status);
        return false;
    }

    status = api.lt_layout_create(
        &resources.layout_b, input_type,
        b_rows, b_cols, p.ldb);

    if (status != kHipblasLtSuccess) {
        error = "Creating hipBLASLt B layout failed, status=" +
                std::to_string(status);
        return false;
    }

    status = api.lt_layout_create(
        &resources.layout_c, output_type,
        p.m, p.n, p.ldc);

    if (status != kHipblasLtSuccess) {
        error = "Creating hipBLASLt C layout failed, status=" +
                std::to_string(status);
        return false;
    }

    status = api.lt_layout_create(
        &resources.layout_d, output_type,
        p.m, p.n, p.ldc);

    if (status != kHipblasLtSuccess) {
        error = "Creating hipBLASLt D layout failed, status=" +
                std::to_string(status);
        return false;
    }

    status = api.lt_preference_create(&resources.preference);

    if (status != kHipblasLtSuccess) {
        error = "hipblasLtMatmulPreferenceCreate failed, status=" +
                std::to_string(status);
        return false;
    }

    // Initial implementation uses zero workspace. This avoids introducing
    // temporary workspace allocation into every GEMM call.
    const size_t max_workspace_bytes = 0;

    status = api.lt_preference_set_attribute(
        resources.preference,
        kLtPreferenceMaxWorkspace,
        &max_workspace_bytes,
        sizeof(max_workspace_bytes));

    if (status != kHipblasLtSuccess) {
        error = "Setting hipBLASLt workspace preference failed, status=" +
                std::to_string(status);
        return false;
    }

    return true;
}

// -----------------------------------------------------------------------------
// Obtain one hipBLASLt algorithm for a normalized GEMM shape.
// -----------------------------------------------------------------------------

bool query_lt_algorithm(
    DynamicApi& api,
    const NormalizedGemm& p,
    bool bf16,
    ShapeDecision& decision,
    std::string& error) {

    LtResources resources(api);

    if (!create_lt_resources(api, p, bf16, resources, error)) {
        return false;
    }

    LtHeuristicResult result{};
    int returned_algorithms = 0;

    const int status = api.lt_algo_get_heuristic(
        get_lt_handle(api),
        resources.desc,
        resources.layout_a,
        resources.layout_b,
        resources.layout_c,
        resources.layout_d,
        resources.preference,
        1,
        &result,
        &returned_algorithms);

    if (status != kHipblasLtSuccess) {
        error = "hipblasLtMatmulAlgoGetHeuristic failed, status=" +
                std::to_string(status);
        return false;
    }

    if (returned_algorithms < 1 ||
        result.state != kHipblasLtSuccess) {
        error = "hipBLASLt returned no usable GEMM algorithm";
        return false;
    }

    if (result.workspaceSize != 0) {
        error = "hipBLASLt selected an algorithm requiring workspace";
        return false;
    }

    decision.ready = true;
    decision.use_hipblaslt = true;
    decision.algo = result.algo;
    decision.workspace_size = result.workspaceSize;

    return true;
}


// -----------------------------------------------------------------------------
// rocBLAS FP32 execution.
// -----------------------------------------------------------------------------

void run_rocblas_fp32(
    rocblas_handle handle,
    const NormalizedGemm& p,
    float alpha,
    float beta,
    float* output) {

    auto fn = clap::dyn::p_rocblas_sgemm;

    if (!fn) {
        throw std::runtime_error("rocBLAS sgemm is unavailable");
    }

    const rocblas_status status = fn(
        handle, p.trans_a, p.trans_b,
        p.m, p.n, p.k,
        &alpha,
        static_cast<const float*>(p.A), p.lda,
        static_cast<const float*>(p.B), p.ldb,
        &beta, output, p.ldc);

    if (status != rocblas_status_success) {
        throw std::runtime_error(
            "rocBLAS sgemm failed, status=" +
            std::to_string(static_cast<int>(status)));
    }
}

// -----------------------------------------------------------------------------
// rocBLAS BF16 execution.
// -----------------------------------------------------------------------------

// The rocBLAS datatype and algorithm parameters are C enum values.
// Use int here, matching the existing CLAP BF16 dynamic function pointer
// and avoiding a compile-time dependency on native rocBLAS enum headers.
using RocblasGemmExFn = rocblas_status (*)(
    rocblas_handle,
    rocblas_operation, rocblas_operation,
    int, int, int,
    const void*,
    const void*, int, int,
    const void*, int, int,
    const void*,
    const void*, int, int,
    void*, int, int,
    int, int, int, unsigned);

RocblasGemmExFn get_rocblas_gemm_ex() {

    static void* library = []() -> void* {
        void* handle = dlopen("librocblas.so", RTLD_NOW | RTLD_LOCAL);

        if (!handle) {
            handle = dlopen("librocblas.so.5", RTLD_NOW | RTLD_LOCAL);
        }

        if (!handle) {
            const char* error = dlerror();

            throw std::runtime_error(
                std::string("Unable to load rocBLAS for BF16 GEMM: ") +
                (error ? error : "unknown dlopen error"));
        }

        return handle;
    }();

    static RocblasGemmExFn function =
        reinterpret_cast<RocblasGemmExFn>(
            dlsym(library, "rocblas_gemm_ex"));

    if (!function) {
        throw std::runtime_error(
            "rocblas_gemm_ex symbol is unavailable");
    }

    return function;
}

void run_rocblas_bf16(
    rocblas_handle handle,
    const NormalizedGemm& p,
    float alpha,
    float beta,
    void* output) {

    constexpr int kRocblasBF16 = 168;
    constexpr int kRocblasF32 = 151;

    auto fn = get_rocblas_gemm_ex();

    constexpr int bf16_type = kRocblasBF16;
    constexpr int f32_type = kRocblasF32;
    constexpr int default_algo = 0;

    const rocblas_status status = fn(
        handle, p.trans_a, p.trans_b,
        p.m, p.n, p.k,
        &alpha,
        p.A, bf16_type, p.lda,
        p.B, bf16_type, p.ldb,
        &beta,
        output, bf16_type, p.ldc,
        output, bf16_type, p.ldc,
        f32_type, default_algo, 0, 0);

    if (status != rocblas_status_success) {
        throw std::runtime_error(
            "rocBLAS BF16 gemm_ex failed, status=" +
            std::to_string(static_cast<int>(status)));
    }
}

// -----------------------------------------------------------------------------
// hipBLASLt execution.
// -----------------------------------------------------------------------------

void run_hipblaslt(
    DynamicApi& api,
    LtHandle lt_handle,
    const LtResources& resources,
    const ShapeDecision& decision,
    const NormalizedGemm& p,
    float alpha,
    float beta,
    const void* input_c,
    void* output,
    void* stream) {

    const int status = api.lt_matmul(
        lt_handle,
        resources.desc,
        &alpha,
        p.A, resources.layout_a,
        p.B, resources.layout_b,
        &beta,
        input_c, resources.layout_c,
        output, resources.layout_d,
        &decision.algo,
        nullptr, 0,
        stream);

    if (status != kHipblasLtSuccess) {
        throw std::runtime_error(
            "hipblasLtMatmul failed, status=" +
            std::to_string(status));
    }
}

// -----------------------------------------------------------------------------
// HIP runtime error handling and scratch-output preparation.
// -----------------------------------------------------------------------------

void check_hip(int status, const char* operation) {
    if (status != 0) {
        throw std::runtime_error(
            std::string(operation) +
            " failed, HIP status=" + std::to_string(status));
    }
}

size_t gemm_output_bytes(const NormalizedGemm& p, bool bf16) {

    const size_t element_size = bf16 ? 2u : 4u;

    if (p.ldc < p.m || p.n < 0) {
        throw std::invalid_argument(
            "Invalid normalized GEMM output dimensions");
    }

    const size_t leading_dimension =
        static_cast<size_t>(p.ldc);

    const size_t columns =
        static_cast<size_t>(p.n);

    if (leading_dimension != 0 &&
        columns > std::numeric_limits<size_t>::max() /
                  leading_dimension) {
        throw std::overflow_error(
            "GEMM scratch matrix size overflow");
    }

    const size_t elements = leading_dimension * columns;

    if (elements >
        std::numeric_limits<size_t>::max() / element_size) {
        throw std::overflow_error(
            "GEMM scratch byte size overflow");
    }

    return elements * element_size;
}

void prepare_scratch_output(
    DynamicApi& api,
    void* scratch,
    const void* original_output,
    size_t bytes,
    float beta,
    void* stream) {

    if (beta == 0.0f) {
        return;
    }

    check_hip(
        api.hip_memcpy_async(
            scratch,
            original_output,
            bytes,
            kHipMemcpyDeviceToDevice,
            stream),
        "hipMemcpyAsync while preparing GEMM scratch output");
}



// -----------------------------------------------------------------------------
// RAII wrappers for temporary GPU buffers and HIP events.
// -----------------------------------------------------------------------------

struct HipDeviceBuffer {
    DynamicApi* api = nullptr;
    void* pointer = nullptr;

    HipDeviceBuffer(DynamicApi& dynamic_api, size_t bytes)
        : api(&dynamic_api) {

        check_hip(
            api->hip_malloc(&pointer, bytes),
            "hipMalloc for GEMM calibration");
    }

    HipDeviceBuffer(const HipDeviceBuffer&) = delete;
    HipDeviceBuffer& operator=(const HipDeviceBuffer&) = delete;

    ~HipDeviceBuffer() {
        if (api && pointer) {
            api->hip_free(pointer);
        }
    }
};

struct HipEventPair {
    DynamicApi* api = nullptr;
    void* start = nullptr;
    void* stop = nullptr;

    explicit HipEventPair(DynamicApi& dynamic_api)
        : api(&dynamic_api) {

        check_hip(
            api->hip_event_create(&start),
            "hipEventCreate(start)");

        try {
            check_hip(
                api->hip_event_create(&stop),
                "hipEventCreate(stop)");
        } catch (...) {
            api->hip_event_destroy(start);
            start = nullptr;
            throw;
        }
    }

    HipEventPair(const HipEventPair&) = delete;
    HipEventPair& operator=(const HipEventPair&) = delete;

    ~HipEventPair() {
        if (api && stop) {
            api->hip_event_destroy(stop);
        }

        if (api && start) {
            api->hip_event_destroy(start);
        }
    }
};

// -----------------------------------------------------------------------------
// Measure one rocBLAS execution using HIP events.
//
// The scratch output is restored before each measurement if beta != 0.
// -----------------------------------------------------------------------------

template <typename RocblasRun>
float measure_rocblas(
    DynamicApi& api,
    HipEventPair& events,
    const NormalizedGemm& p,
    const void* original_output,
    size_t output_bytes,
    float beta,
    void* scratch_output,
    void* stream,
    RocblasRun&& run_rocblas) {

    prepare_scratch_output(
        api, scratch_output, original_output,
        output_bytes, beta, stream);

    check_hip(
        api.hip_event_record(events.start, stream),
        "hipEventRecord(rocBLAS start)");

    run_rocblas(scratch_output);

    check_hip(
        api.hip_event_record(events.stop, stream),
        "hipEventRecord(rocBLAS stop)");

    check_hip(
        api.hip_event_synchronize(events.stop),
        "hipEventSynchronize(rocBLAS)");

    float elapsed_ms = 0.0f;

    check_hip(
        api.hip_event_elapsed_time(
            &elapsed_ms, events.start, events.stop),
        "hipEventElapsedTime(rocBLAS)");

    return elapsed_ms;
}

// -----------------------------------------------------------------------------
// Measure one hipBLASLt execution using HIP events.
// -----------------------------------------------------------------------------

float measure_hipblaslt(
    DynamicApi& api,
    HipEventPair& events,
    LtHandle lt_handle,
    const LtResources& resources,
    const ShapeDecision& decision,
    const NormalizedGemm& p,
    float alpha,
    float beta,
    const void* original_output,
    size_t output_bytes,
    void* scratch_c,
    void* scratch_d,
    void* stream) {

    prepare_scratch_output(
        api, scratch_c, original_output,
        output_bytes, beta, stream);

    check_hip(
        api.hip_event_record(events.start, stream),
        "hipEventRecord(hipBLASLt start)");

    run_hipblaslt(
        api, lt_handle, resources, decision, p,
        alpha, beta, scratch_c, scratch_d, stream);

    check_hip(
        api.hip_event_record(events.stop, stream),
        "hipEventRecord(hipBLASLt stop)");

    check_hip(
        api.hip_event_synchronize(events.stop),
        "hipEventSynchronize(hipBLASLt)");

    float elapsed_ms = 0.0f;

    check_hip(
        api.hip_event_elapsed_time(
            &elapsed_ms, events.start, events.stop),
        "hipEventElapsedTime(hipBLASLt)");

    return elapsed_ms;
}

// -----------------------------------------------------------------------------
// First-use calibration for one GEMM shape class.
//
// Both candidates are warmed up. Then each is measured three times, with
// alternating measurement order. The best sample is used for the decision.
//
// hipBLASLt is selected only when it is more than 3% faster.
// -----------------------------------------------------------------------------

template <typename RocblasRun>
bool calibrate_shape(
    DynamicApi& api,
    LtHandle lt_handle,
    const NormalizedGemm& p,
    bool bf16,
    float alpha,
    float beta,
    const void* original_output,
    void* stream,
    RocblasRun&& run_rocblas,
    ShapeDecision& decision,
    std::string& error) {

    if (!decision.ready &&
        !query_lt_algorithm(api, p, bf16, decision, error)) {
        return false;
    }

    LtResources resources(api);

    if (!create_lt_resources(
            api, p, bf16, resources, error)) {
        decision.ready = false;
        return false;
    }

    const size_t output_bytes = gemm_output_bytes(p, bf16);

    HipDeviceBuffer scratch_c(api, output_bytes);
    HipDeviceBuffer scratch_d(api, output_bytes);

    HipEventPair events(api);

    // Warm up rocBLAS.
    prepare_scratch_output(
        api, scratch_d.pointer, original_output,
        output_bytes, beta, stream);

    run_rocblas(scratch_d.pointer);

    // Warm up hipBLASLt. C input and D output are separate buffers.
    prepare_scratch_output(
        api, scratch_c.pointer, original_output,
        output_bytes, beta, stream);

    run_hipblaslt(
        api, lt_handle, resources, decision, p,
        alpha, beta, scratch_c.pointer,
        scratch_d.pointer, stream);

    // Ensure both warm-up executions have completed before timing.
    check_hip(
        api.hip_stream_synchronize(stream),
        "hipStreamSynchronize after GEMM warm-up");

    float best_rocblas_ms =
        std::numeric_limits<float>::infinity();

    float best_hipblaslt_ms =
        std::numeric_limits<float>::infinity();

    for (int round = 0; round < 3; ++round) {

        float rocblas_ms = 0.0f;
        float hipblaslt_ms = 0.0f;

        if ((round % 2) == 0) {

            rocblas_ms = measure_rocblas(
                api, events, p,
                original_output, output_bytes,
                beta, scratch_d.pointer, stream,
                run_rocblas);

            hipblaslt_ms = measure_hipblaslt(
                api, events, lt_handle, resources,
                decision, p, alpha, beta,
                original_output, output_bytes,
                scratch_c.pointer, scratch_d.pointer,
                stream);

        } else {

            hipblaslt_ms = measure_hipblaslt(
                api, events, lt_handle, resources,
                decision, p, alpha, beta,
                original_output, output_bytes,
                scratch_c.pointer, scratch_d.pointer,
                stream);

            rocblas_ms = measure_rocblas(
                api, events, p,
                original_output, output_bytes,
                beta, scratch_d.pointer, stream,
                run_rocblas);
        }

        best_rocblas_ms =
            std::min(best_rocblas_ms, rocblas_ms);

        best_hipblaslt_ms =
            std::min(best_hipblaslt_ms, hipblaslt_ms);
    }

    constexpr float kRequiredSpeedup = 0.03f;

    decision.ready = true;
    decision.calibrated = true;
    decision.use_hipblaslt =
        best_hipblaslt_ms <
        best_rocblas_ms * (1.0f - kRequiredSpeedup);

    if (debug_enabled()) {
        std::fprintf(
            stderr,
            "[CLAP ROCm GEMM] calibration "
            "precision=%s m=%d n=%d k=%d "
            "rocBLAS_best=%.3f us "
            "hipBLASLt_best=%.3f us "
            "selected=%s\n",
            bf16 ? "BF16" : "FP32",
            p.m, p.n, p.k,
            best_rocblas_ms * 1000.0f,
            best_hipblaslt_ms * 1000.0f,
            decision.use_hipblaslt ? "hipblaslt" : "rocblas");
    }

    return true;
}


// -----------------------------------------------------------------------------
// Query graph-capture status.
//
// Any nonzero status (Active or Invalidated) is treated conservatively as
// capture. Calibration is skipped in that case.
// -----------------------------------------------------------------------------

bool query_stream_capture(
    DynamicApi& api,
    void* stream,
    bool& capturing,
    std::string& error) {

    int capture_status = 0;

    const int status =
        api.hip_stream_is_capturing(stream, &capture_status);

    if (status != 0) {
        error = "hipStreamIsCapturing failed, HIP status=" +
                std::to_string(status);
        return false;
    }

    capturing = (capture_status != 0);
    return true;
}

// -----------------------------------------------------------------------------
// Execute hipBLASLt for the current dimensions.
//
// For normal GEMM execution C and D refer to the caller's output matrix.
// The calibration path uses separate scratch C and D buffers.
// -----------------------------------------------------------------------------

void execute_hipblaslt(
    DynamicApi& api,
    const NormalizedGemm& p,
    bool bf16,
    const ShapeDecision& decision,
    float alpha,
    float beta,
    void* C,
    void* stream) {

    LtResources resources(api);
    std::string error;

    if (!create_lt_resources(
            api, p, bf16, resources, error)) {
        throw std::runtime_error(error);
    }

    run_hipblaslt(
        api,
        get_lt_handle(api),
        resources,
        decision,
        p,
        alpha,
        beta,
        C,
        C,
        stream);
}

// -----------------------------------------------------------------------------
// Common backend dispatcher for FP32 and BF16.
// -----------------------------------------------------------------------------

template <typename RocblasRun>
void dispatch_gemm(
    GemmPrecision precision,
    bool bf16,
    const NormalizedGemm& p,
    float alpha,
    float beta,
    const void* original_output,
    void* output,
    void* stream,
    RocblasRun&& run_rocblas) {

    const GemmMode mode = get_gemm_mode();

    // Explicit rocBLAS mode avoids loading HIP or hipBLASLt altogether.
    if (mode == GemmMode::ForceRocblas) {
        run_rocblas(output);
        return;
    }

    DynamicApi& api = dynamic_api();

    if (!api.load()) {
        if (mode == GemmMode::ForceHipblaslt) {
            throw std::runtime_error(
                "CLAP_ROCM_GEMM=hipblaslt requested, but dynamic "
                "hipBLASLt/HIP loading failed: " + api.error);
        }

        debug_log(
            "hipBLASLt/HIP runtime unavailable; falling back to rocBLAS");

        run_rocblas(output);
        return;
    }

    const ShapeKey key = make_shape_key(
        precision,
        p.m,
        p.n,
        p.k,
        static_cast<int>(p.trans_a),
        static_cast<int>(p.trans_b));

    // Forced hipBLASLt mode uses the heuristic without performance timing.
    if (mode == GemmMode::ForceHipblaslt) {

        ShapeDecision decision;

        {
            std::lock_guard<std::mutex> lock(decision_mutex());

            auto& cache = decision_cache();
            auto it = cache.find(key);

            if (it == cache.end() || !it->second.ready) {

                ShapeDecision candidate;
                std::string error;

                if (!query_lt_algorithm(
                        api, p, bf16, candidate, error)) {
                    throw std::runtime_error(
                        "Forced hipBLASLt algorithm selection failed: " +
                        error);
                }

                cache[key] = candidate;
                it = cache.find(key);
            }

            decision = it->second;
        }

        // Forced mode uses the heuristic even if an earlier auto
        // calibration found rocBLAS to be faster.
        decision.use_hipblaslt = true;

        execute_hipblaslt(
            api, p, bf16, decision,
            alpha, beta, output, stream);

        return;
    }

    // Auto mode: detect graph capture before attempting calibration.
    bool capturing = false;
    std::string capture_error;

    if (!query_stream_capture(
            api, stream, capturing, capture_error)) {

        debug_log(capture_error.c_str());
        debug_log("Using rocBLAS because capture status is unavailable");

        run_rocblas(output);
        return;
    }

    ShapeDecision decision;
    bool fallback_to_rocblas = false;
    std::string calibration_error;

    {
        // Serialize first-use calibration and cache insertion.
        std::lock_guard<std::mutex> lock(decision_mutex());

        auto& cache = decision_cache();
        auto it = cache.find(key);

        if (it != cache.end() && it->second.calibrated) {

            decision = it->second;

        } else if (capturing) {

            // No timing and no cache insertion during graph capture.
            fallback_to_rocblas = true;

        } else {

            ShapeDecision candidate;

            if (it != cache.end()) {
                // Reuse a previously obtained heuristic when available.
                candidate = it->second;
            }

            try {
                if (calibrate_shape(
                        api,
                        get_lt_handle(api),
                        p,
                        bf16,
                        alpha,
                        beta,
                        original_output,
                        stream,
                        run_rocblas,
                        candidate,
                        calibration_error)) {

                    candidate.calibrated = true;
                    cache[key] = candidate;
                    decision = candidate;

                } else {
                    fallback_to_rocblas = true;
                }

            } catch (const std::exception& e) {

                calibration_error = e.what();

                // Best-effort synchronization before using the fallback.
                (void)api.hip_stream_synchronize(stream);

                fallback_to_rocblas = true;
            }
        }
    }

    if (fallback_to_rocblas) {

        if (!calibration_error.empty()) {
            debug_log(calibration_error.c_str());
        }

        if (capturing) {
            debug_log(
                "Graph capture active; skipping calibration for this call");
        } else {
            debug_log(
                "Calibration unavailable; using rocBLAS for this call");
        }

        run_rocblas(output);
        return;
    }

    if (!decision.use_hipblaslt) {
        run_rocblas(output);
        return;
    }

    // An algorithm may become unsuitable for a particular shape within
    // a bucket. In auto mode, fall back to rocBLAS if Lt rejects it.
    try {

        execute_hipblaslt(
            api, p, bf16, decision,
            alpha, beta, output, stream);

    } catch (const std::exception& e) {

        debug_log(e.what());
        debug_log("hipBLASLt execution failed; falling back to rocBLAS");

        {
            std::lock_guard<std::mutex> lock(decision_mutex());

            auto& cache = decision_cache();
            auto it = cache.find(key);

            if (it != cache.end()) {
                it->second.use_hipblaslt = false;
            }
        }

        run_rocblas(output);
    }
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// Public FP32 device GEMM selector.
// -----------------------------------------------------------------------------

void gemm_fp32(
    rocblas_handle handle,
    Layout layout,
    Transpose transA,
    Transpose transB,
    int m, int n, int k,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc,
    void* stream) {

    const NormalizedGemm p = normalize_gemm(
        layout, transA, transB,
        m, n, k,
        A, lda, B, ldb, ldc);

    const auto run_rocblas = [&](void* output) {
        run_rocblas_fp32(
            handle, p, alpha, beta,
            static_cast<float*>(output));
    };

    dispatch_gemm(
        GemmPrecision::FP32,
        false,
        p,
        alpha,
        beta,
        C,
        C,
        stream,
        run_rocblas);
}

// -----------------------------------------------------------------------------
// Public BF16 device GEMM selector.
// -----------------------------------------------------------------------------

void gemm_bf16(
    rocblas_handle handle,
    Layout layout,
    Transpose transA,
    Transpose transB,
    int m, int n, int k,
    float alpha,
    const void* A, int lda,
    const void* B, int ldb,
    float beta,
    void* C, int ldc,
    void* stream) {

    const NormalizedGemm p = normalize_gemm(
        layout, transA, transB,
        m, n, k,
        A, lda, B, ldb, ldc);

    const auto run_rocblas = [&](void* output) {
        run_rocblas_bf16(
            handle, p, alpha, beta, output);
    };

    dispatch_gemm(
        GemmPrecision::BF16,
        true,
        p,
        alpha,
        beta,
        C,
        C,
        stream,
        run_rocblas);
}

} // namespace rocblas_gemm_select
} // namespace clap
