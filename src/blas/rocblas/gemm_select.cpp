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
#include <initializer_list>
#include <memory>
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
constexpr int kHipDataTypeF16 = 2;
constexpr int kHipDataTypeE4M3 = 28;
constexpr int kHipDataTypeE5M2 = 29;
constexpr int kHipDataTypeE4M3Fnuz = 1000;
constexpr int kHipDataTypeE5M2Fnuz = 1001;

constexpr int kLtDescAScalePointer = 5;
constexpr int kLtDescBScalePointer = 6;
constexpr int kLtDescAScaleMode = 31;
constexpr int kLtDescBScaleMode = 32;
constexpr int kLtMatrixScaleScalar32F = 0;

constexpr int kHipComputeF32 = 2;

constexpr int kHipOperationN = 111;
constexpr int kHipOperationT = 112;

constexpr int kLtDescTransA = 0;
constexpr int kLtDescTransB = 1;
constexpr int kLtDescPointerMode = 13;

constexpr int kLtPointerModeHost = 0;
constexpr int kLtPreferenceMaxWorkspace = 1;

// hipblasLtMatrixLayoutAttribute_t (ROCm 7.2.2).
constexpr int kLtMatrixLayoutBatchCount = 0;
constexpr int kLtMatrixLayoutStridedBatchOffset = 1;

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

    using HipGetDeviceFn =
        int (*)(int*);

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
    HipGetDeviceFn hip_get_device = nullptr;
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

    using LtMatrixLayoutSetAttributeFn =
        int (*)(LtMatrixLayout, int, const void*, size_t);

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
    LtMatrixLayoutSetAttributeFn lt_layout_set_attribute = nullptr;
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

    void* open_library(std::initializer_list<const char*> names) {
        for (const char* name : names) {
            if (void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
                return handle;
            }
        }

        return nullptr;
    }

    bool load() {

        std::lock_guard<std::mutex> lock(load_mutex);

        if (attempted) {
            return loaded;
        }

        attempted = true;

        hip_library = open_library({
            "libamdhip64.so",
            "libamdhip64.so.7",
            "libamdhip64.so.6"
        });

        if (!hip_library) {
            error = "Unable to load libamdhip64.so";
            return false;
        }

        lt_library = open_library({
            "libhipblaslt.so",
            "libhipblaslt.so.1"
        });

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

        // Optional: used only by the per-device FP8 handle cache.
        hip_get_device = reinterpret_cast<HipGetDeviceFn>(
            dlsym(hip_library, "hipGetDevice"));

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
        LOAD_LT(lt_layout_set_attribute,
                "hipblasLtMatrixLayoutSetAttribute");
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

void debug_log(const std::string& message) {
    debug_log(message.c_str());
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
    int batch_count;
    int64_t stride_a;
    int64_t stride_b;
    int64_t stride_c;
    bool strided_batched;

    bool operator==(const ShapeKey& other) const {
        return precision == other.precision &&
               m_bucket == other.m_bucket &&
               n_bucket == other.n_bucket &&
               k == other.k &&
               trans_a == other.trans_a &&
               trans_b == other.trans_b &&
               batch_count == other.batch_count &&
               stride_a == other.stride_a &&
               stride_b == other.stride_b &&
               stride_c == other.stride_c &&
               strided_batched == other.strided_batched;
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
        combine(std::hash<int>{}(key.batch_count));
        combine(std::hash<int64_t>{}(key.stride_a));
        combine(std::hash<int64_t>{}(key.stride_b));
        combine(std::hash<int64_t>{}(key.stride_c));
        combine(std::hash<bool>{}(key.strided_batched));

        return seed;
    }
};

ShapeKey make_shape_key(
    GemmPrecision precision,
    int m,
    int n,
    int k,
    int trans_a,
    int trans_b,
    int batch_count,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c,
    bool strided_batched) {

    return ShapeKey{
        precision,
        shape_bucket(m),
        shape_bucket(n),
        k,
        trans_a,
        trans_b,
        batch_count,
        stride_a,
        stride_b,
        stride_c,
        strided_batched
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

    int batch_count = 1;
    int64_t stride_a = 0;
    int64_t stride_b = 0;
    int64_t stride_c = 0;
    bool strided_batched = false;
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

NormalizedGemm normalize_strided_batched_gemm(
    Layout layout,
    Transpose transA,
    Transpose transB,
    int m, int n, int k,
    const void* A, int lda, int64_t strideA,
    const void* B, int ldb, int64_t strideB,
    int ldc, int64_t strideC,
    int batch_count) {

    NormalizedGemm p = normalize_gemm(
        layout, transA, transB,
        m, n, k, A, lda, B, ldb, ldc);

    p.batch_count = batch_count;
    p.stride_c = strideC;
    p.strided_batched = true;

    if (layout == Layout::RowMajor) {
        p.stride_a = strideB;
        p.stride_b = strideA;
    } else {
        p.stride_a = strideA;
        p.stride_b = strideB;
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

// FP8 handles are cached separately, one per HIP device.
bool get_current_hip_device(DynamicApi& api, int& device) {
    if (!api.hip_get_device) {
        return false;
    }

    device = -1;
    return api.hip_get_device(&device) == 0 && device >= 0;
}

LtHandle get_fp8_lt_handle(DynamicApi& api, int device) {
    static std::mutex mutex;
    static std::unordered_map<int, LtHandle> handles;

    std::lock_guard<std::mutex> lock(mutex);

    const auto found = handles.find(device);
    if (found != handles.end()) {
        return found->second;
    }

    LtHandle handle = nullptr;
    const int status = api.lt_create(&handle);

    if (status != kHipblasLtSuccess || !handle) {
        debug_log("hipblasLtCreate failed for FP8 device handle");
        return nullptr;
    }

    handles.emplace(device, handle);
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

// FP8 plans use exact dimensions and datatype combinations. Unlike the
// existing FP32/BF16 cache, these plans retain their descriptors and layouts.
struct Fp8PlanKey {
    int device;
    int m, n, k;
    int lda, ldb, ldc;
    int trans_a, trans_b;
    int type_a, type_b, type_c;

    bool operator==(const Fp8PlanKey& other) const {
        return device == other.device &&
               m == other.m && n == other.n && k == other.k &&
               lda == other.lda && ldb == other.ldb && ldc == other.ldc &&
               trans_a == other.trans_a && trans_b == other.trans_b &&
               type_a == other.type_a && type_b == other.type_b &&
               type_c == other.type_c;
    }
};

struct Fp8PlanKeyHash {
    size_t operator()(const Fp8PlanKey& key) const {
        size_t seed = 0;
        const auto combine = [&seed](size_t value) {
            seed ^= value + 0x9e3779b9U + (seed << 6) + (seed >> 2);
        };

        combine(std::hash<int>{}(key.device));
        combine(std::hash<int>{}(key.m));
        combine(std::hash<int>{}(key.n));
        combine(std::hash<int>{}(key.k));
        combine(std::hash<int>{}(key.lda));
        combine(std::hash<int>{}(key.ldb));
        combine(std::hash<int>{}(key.ldc));
        combine(std::hash<int>{}(key.trans_a));
        combine(std::hash<int>{}(key.trans_b));
        combine(std::hash<int>{}(key.type_a));
        combine(std::hash<int>{}(key.type_b));
        combine(std::hash<int>{}(key.type_c));
        return seed;
    }
};

struct Fp8Plan {
    std::unique_ptr<LtResources> resources;
    LtMatmulAlgo algo{};
    std::mutex mutex;
};

std::mutex& fp8_plan_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<
    Fp8PlanKey,
    std::shared_ptr<Fp8Plan>,
    Fp8PlanKeyHash>& fp8_plan_cache() {
    static std::unordered_map<
        Fp8PlanKey,
        std::shared_ptr<Fp8Plan>,
        Fp8PlanKeyHash> cache;
    return cache;
}

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

    // Configure strided-batched execution on all four matrix layouts.
    // hipBLASLt expects batch count as int32_t and strides in elements
    // as int64_t, matching the ROCm 7.2.2 public header.
    if (p.strided_batched) {
        const int32_t batch_count =
            static_cast<int32_t>(p.batch_count);

        const auto set_batch_attributes =
            [&](LtMatrixLayout layout,
                int64_t stride,
                const char* matrix_name) -> bool {

            int attr_status = api.lt_layout_set_attribute(
                layout,
                kLtMatrixLayoutBatchCount,
                &batch_count,
                sizeof(batch_count));

            if (attr_status != kHipblasLtSuccess) {
                error = std::string("Setting hipBLASLt ") +
                        matrix_name +
                        " batch count failed, status=" +
                        std::to_string(attr_status);
                return false;
            }

            attr_status = api.lt_layout_set_attribute(
                layout,
                kLtMatrixLayoutStridedBatchOffset,
                &stride,
                sizeof(stride));

            if (attr_status != kHipblasLtSuccess) {
                error = std::string("Setting hipBLASLt ") +
                        matrix_name +
                        " batch stride failed, status=" +
                        std::to_string(attr_status);
                return false;
            }

            return true;
        };

        if (!set_batch_attributes(
                resources.layout_a, p.stride_a, "A") ||
            !set_batch_attributes(
                resources.layout_b, p.stride_b, "B") ||
            !set_batch_attributes(
                resources.layout_c, p.stride_c, "C") ||
            !set_batch_attributes(
                resources.layout_d, p.stride_c, "D")) {
            return false;
        }
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

// Map CLAP FP8 input formats to ROCm HIP datatypes.
bool fp8_hip_data_type(BlasDataType type, int& result) {
    switch (type) {
        case BlasDataType::Float8E4M3:
            result = kHipDataTypeE4M3;
            return true;
        case BlasDataType::Float8E5M2:
            result = kHipDataTypeE5M2;
            return true;
        case BlasDataType::Float8E4M3Fnuz:
            result = kHipDataTypeE4M3Fnuz;
            return true;
        case BlasDataType::Float8E5M2Fnuz:
            result = kHipDataTypeE5M2Fnuz;
            return true;
        default:
            return false;
    }
}

bool fp8_output_hip_data_type(BlasDataType type, int& result) {
    switch (type) {
        case BlasDataType::Float16:
            result = kHipDataTypeF16;
            return true;
        case BlasDataType::BFloat16:
            result = kHipDataTypeBF16;
            return true;
        case BlasDataType::Float32:
            result = kHipDataTypeF32;
            return true;
        default:
            return false;
    }
}

// Build and retain the descriptor/layouts and selected algorithm for one
// exact FP8 GEMM shape. Unsupported combinations return false.
bool create_fp8_plan(
    DynamicApi& api,
    LtHandle handle,
    const NormalizedGemm& p,
    BlasDataType type_a,
    BlasDataType type_b,
    BlasDataType type_c,
    std::shared_ptr<Fp8Plan>& result,
    std::string& error) {

    int input_type_a = 0;
    int input_type_b = 0;
    int output_type = 0;

    if (!fp8_hip_data_type(type_a, input_type_a) ||
        !fp8_hip_data_type(type_b, input_type_b) ||
        !fp8_output_hip_data_type(type_c, output_type)) {
        error = "Unsupported FP8 input or output datatype";
        return false;
    }

    auto plan = std::make_shared<Fp8Plan>();
    plan->resources = std::make_unique<LtResources>(api);
    LtResources& resources = *plan->resources;

    int status = api.lt_desc_create(
        &resources.desc, kHipComputeF32, kHipDataTypeF32);

    if (status != kHipblasLtSuccess) {
        error = "FP8 hipBLASLt descriptor creation failed, status=" +
                std::to_string(status);
        return false;
    }

    const int trans_a = to_lt_operation(p.trans_a);
    const int trans_b = to_lt_operation(p.trans_b);
    const int pointer_mode = kLtPointerModeHost;

    const auto set_desc_attribute =
        [&](int attribute, const void* value, size_t size,
            const char* name) -> bool {
            const int attr_status = api.lt_desc_set_attribute(
                resources.desc, attribute, value, size);
            if (attr_status != kHipblasLtSuccess) {
                error = std::string("Setting FP8 ") + name +
                        " failed, status=" + std::to_string(attr_status);
                return false;
            }
            return true;
        };

    if (!set_desc_attribute(
            kLtDescTransA, &trans_a, sizeof(trans_a), "TRANSA") ||
        !set_desc_attribute(
            kLtDescTransB, &trans_b, sizeof(trans_b), "TRANSB") ||
        !set_desc_attribute(
            kLtDescPointerMode, &pointer_mode,
            sizeof(pointer_mode), "pointer mode")) {
        return false;
    }

    const int scale_mode = kLtMatrixScaleScalar32F;

    if (!set_desc_attribute(
            kLtDescAScaleMode, &scale_mode,
            sizeof(scale_mode), "A scale mode") ||
        !set_desc_attribute(
            kLtDescBScaleMode, &scale_mode,
            sizeof(scale_mode), "B scale mode")) {
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
        &resources.layout_a, input_type_a,
        a_rows, a_cols, p.lda);
    if (status != kHipblasLtSuccess) {
        error = "Creating FP8 A layout failed, status=" +
                std::to_string(status);
        return false;
    }

    status = api.lt_layout_create(
        &resources.layout_b, input_type_b,
        b_rows, b_cols, p.ldb);
    if (status != kHipblasLtSuccess) {
        error = "Creating FP8 B layout failed, status=" +
                std::to_string(status);
        return false;
    }

    status = api.lt_layout_create(
        &resources.layout_c, output_type, p.m, p.n, p.ldc);
    if (status != kHipblasLtSuccess) {
        error = "Creating FP8 C layout failed, status=" +
                std::to_string(status);
        return false;
    }

    status = api.lt_layout_create(
        &resources.layout_d, output_type, p.m, p.n, p.ldc);
    if (status != kHipblasLtSuccess) {
        error = "Creating FP8 D layout failed, status=" +
                std::to_string(status);
        return false;
    }

    status = api.lt_preference_create(&resources.preference);
    if (status != kHipblasLtSuccess) {
        error = "Creating FP8 heuristic preference failed, status=" +
                std::to_string(status);
        return false;
    }

    const size_t max_workspace_bytes = 0;
    status = api.lt_preference_set_attribute(
        resources.preference,
        kLtPreferenceMaxWorkspace,
        &max_workspace_bytes,
        sizeof(max_workspace_bytes));

    if (status != kHipblasLtSuccess) {
        error = "Setting FP8 workspace preference failed, status=" +
                std::to_string(status);
        return false;
    }

    constexpr int kRequestedAlgorithms = 8;
    LtHeuristicResult candidates[kRequestedAlgorithms]{};
    int returned = 0;

    status = api.lt_algo_get_heuristic(
        handle,
        resources.desc,
        resources.layout_a,
        resources.layout_b,
        resources.layout_c,
        resources.layout_d,
        resources.preference,
        kRequestedAlgorithms,
        candidates,
        &returned);

    if (status != kHipblasLtSuccess || returned <= 0 ||
        returned > kRequestedAlgorithms) {
        error = "No FP8 hipBLASLt heuristic result is available";
        return false;
    }

    int selected = -1;
    for (int i = 0; i < returned; ++i) {
        if (candidates[i].state == kHipblasLtSuccess &&
            candidates[i].workspaceSize == 0) {
            selected = i;
            break;
        }
    }

    if (selected < 0) {
        error = "No usable zero-workspace FP8 algorithm";
        return false;
    }

    plan->algo = candidates[selected].algo;
    result = std::move(plan);
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

    constexpr int kRequestedAlgorithms = 8;

    LtHeuristicResult results[kRequestedAlgorithms]{};
    int returned_algorithms = 0;

    const int status = api.lt_algo_get_heuristic(
        get_lt_handle(api),
        resources.desc,
        resources.layout_a,
        resources.layout_b,
        resources.layout_c,
        resources.layout_d,
        resources.preference,
        kRequestedAlgorithms,
        results,
        &returned_algorithms);

    if (status != kHipblasLtSuccess) {
        error = "hipblasLtMatmulAlgoGetHeuristic failed, status=" +
                std::to_string(status);
        return false;
    }

    if (returned_algorithms < 0 ||
        returned_algorithms > kRequestedAlgorithms) {
        error = "hipBLASLt returned an invalid algorithm count";
        return false;
    }

    if (returned_algorithms == 0) {
        error = "hipBLASLt returned no GEMM algorithms";
        return false;
    }

    int selected = -1;

    for (int i = 0; i < returned_algorithms; ++i) {
        const auto& candidate = results[i];

        if (debug_enabled()) {
            std::fprintf(
                stderr,
                "[CLAP ROCm GEMM] Lt candidate %d/%d "
                "state=%d workspace=%zu waves=%.3f\n",
                i,
                returned_algorithms,
                candidate.state,
                candidate.workspaceSize,
                candidate.wavesCount);
        }

        if (candidate.state == kHipblasLtSuccess &&
            candidate.workspaceSize == 0) {
            selected = i;
            break;
        }
    }

    if (selected < 0) {
        error = "hipBLASLt returned no usable zero-workspace algorithm";
        return false;
    }

    decision.ready = true;
    decision.use_hipblaslt = true;
    decision.algo = results[selected].algo;
    decision.workspace_size = results[selected].workspaceSize;

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
            handle = dlopen("librocblas.so.4", RTLD_NOW | RTLD_LOCAL);
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

// -----------------------------------------------------------------------------
// rocBLAS FP32 strided-batched GEMM execution.
// -----------------------------------------------------------------------------

using RocblasSgemmStridedBatchedFn = rocblas_status (*)(
    rocblas_handle,
    rocblas_operation, rocblas_operation,
    int, int, int,
    const float*,
    const float*, int, int64_t,
    const float*, int, int64_t,
    const float*,
    float*, int, int64_t,
    int);

RocblasSgemmStridedBatchedFn get_rocblas_sgemm_strided_batched() {

    static void* library = []() -> void* {
        void* handle = dlopen("librocblas.so", RTLD_NOW | RTLD_LOCAL);

        if (!handle) {
            handle = dlopen("librocblas.so.5", RTLD_NOW | RTLD_LOCAL);
        }

        if (!handle) {
            handle = dlopen("librocblas.so.4", RTLD_NOW | RTLD_LOCAL);
        }

        if (!handle) {
            const char* error = dlerror();
            throw std::runtime_error(
                std::string("Unable to load rocBLAS for strided-batched "
                            "FP32 GEMM: ") +
                (error ? error : "unknown dlopen error"));
        }

        return handle;
    }();

    static RocblasSgemmStridedBatchedFn function =
        reinterpret_cast<RocblasSgemmStridedBatchedFn>(
            dlsym(library, "rocblas_sgemm_strided_batched"));

    if (!function) {
        throw std::runtime_error(
            "rocblas_sgemm_strided_batched symbol is unavailable");
    }

    return function;
}

void run_rocblas_fp32_strided_batched(
    rocblas_handle handle,
    const NormalizedGemm& p,
    float alpha,
    float beta,
    float* output) {

    auto fn = get_rocblas_sgemm_strided_batched();

    const rocblas_status status = fn(
        handle,
        p.trans_a, p.trans_b,
        p.m, p.n, p.k,
        &alpha,
        static_cast<const float*>(p.A), p.lda, p.stride_a,
        static_cast<const float*>(p.B), p.ldb, p.stride_b,
        &beta,
        output, p.ldc, p.stride_c,
        p.batch_count);

    if (status != rocblas_status_success) {
        throw std::runtime_error(
            "rocBLAS FP32 strided-batched GEMM failed, status=" +
            std::to_string(static_cast<int>(status)));
    }
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

    if (p.ldc < p.m || p.n < 0 ||
        p.batch_count < 1 || p.stride_c < 0) {
        throw std::invalid_argument(
            "Invalid normalized GEMM output dimensions or batch metadata");
    }

    const size_t leading_dimension = static_cast<size_t>(p.ldc);
    const size_t columns = static_cast<size_t>(p.n);
    const size_t max_size = std::numeric_limits<size_t>::max();

    if (leading_dimension != 0 &&
        columns > max_size / leading_dimension) {
        throw std::overflow_error("GEMM scratch matrix size overflow");
    }

    const size_t matrix_elements = leading_dimension * columns;
    const size_t preceding_batches =
        static_cast<size_t>(p.batch_count - 1);
    const size_t stride =
        static_cast<size_t>(p.stride_c);

    size_t batch_offset = 0;

    if (stride != 0 && preceding_batches != 0) {
        if (preceding_batches > max_size / stride) {
            throw std::overflow_error(
                "GEMM batched scratch offset overflow");
        }

        batch_offset = preceding_batches * stride;
    }

    if (matrix_elements > max_size - batch_offset) {
        throw std::overflow_error(
            "GEMM batched scratch element count overflow");
    }

    const size_t total_elements = batch_offset + matrix_elements;

    if (total_elements > max_size / element_size) {
        throw std::overflow_error(
            "GEMM batched scratch byte size overflow");
    }

    return total_elements * element_size;
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

    check_hip(
        api.hip_memset_async(
            scratch_c.pointer, 0, output_bytes, stream),
        "hipMemsetAsync for GEMM scratch C");

    check_hip(
        api.hip_memset_async(
            scratch_d.pointer, 0, output_bytes, stream),
        "hipMemsetAsync for GEMM scratch D");

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
        static_cast<int>(p.trans_b),
        p.batch_count,
        p.stride_a,
        p.stride_b,
        p.stride_c,
        p.strided_batched);

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
// -----------------------------------------------------------------------------
// Public FP32 strided-batched device GEMM selector.
// -----------------------------------------------------------------------------

void gemm_fp32_strided_batched(
    rocblas_handle handle,
    Layout layout,
    Transpose transA,
    Transpose transB,
    int m, int n, int k,
    float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta,
    float* C, int ldc, long long strideC,
    int batch_count,
    void* stream) {

    if (batch_count < 0 || strideA < 0 ||
        strideB < 0 || strideC < 0) {
        throw std::invalid_argument(
            "Invalid FP32 strided-batched GEMM count or stride");
    }

    if (batch_count == 0 || m == 0 || n == 0) {
        return;
    }

    const NormalizedGemm p = normalize_strided_batched_gemm(
        layout, transA, transB,
        m, n, k,
        A, lda, static_cast<int64_t>(strideA),
        B, ldb, static_cast<int64_t>(strideB),
        ldc, static_cast<int64_t>(strideC),
        batch_count);

    const auto run_rocblas = [&](void* output) {
        run_rocblas_fp32_strided_batched(
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


// -----------------------------------------------------------------------------
// Public scaled FP8 device GEMM selector.
// -----------------------------------------------------------------------------

bool gemm_scaled_fp8(
    Layout layout,
    Transpose transA,
    Transpose transB,
    int m, int n, int k,
    float alpha,
    const void* A, BlasDataType typeA, int lda,
    const float* scaleA,
    const void* B, BlasDataType typeB, int ldb,
    const float* scaleB,
    float beta,
    void* C, BlasDataType typeC, int ldc,
    void* stream) {

    try {
        if (m < 0 || n < 0 || k < 0) {
            return false;
        }

        if (m == 0 || n == 0) {
            return true;
        }

        if (k == 0 || !A || !B || !C || !scaleA || !scaleB) {
            return false;
        }

        int ignored_type = 0;
        if (!fp8_hip_data_type(typeA, ignored_type) ||
            !fp8_hip_data_type(typeB, ignored_type) ||
            !fp8_output_hip_data_type(typeC, ignored_type)) {
            return false;
        }

        DynamicApi& api = dynamic_api();
        if (!api.load()) {
            debug_log(api.error);
            return false;
        }

        int device = -1;
        if (!get_current_hip_device(api, device)) {
            debug_log("Unable to determine current HIP device for FP8 GEMM");
            return false;
        }

        LtHandle handle = get_fp8_lt_handle(api, device);
        if (!handle) {
            return false;
        }

        const NormalizedGemm p = normalize_gemm(
            layout, transA, transB,
            m, n, k, A, lda, B, ldb, ldc);

        // Row-major normalization swaps the two input operands.
        BlasDataType normalized_type_a = typeA;
        BlasDataType normalized_type_b = typeB;
        const float* normalized_scale_a = scaleA;
        const float* normalized_scale_b = scaleB;

        if (layout == Layout::RowMajor) {
            std::swap(normalized_type_a, normalized_type_b);
            std::swap(normalized_scale_a, normalized_scale_b);
        }

        const Fp8PlanKey key{
            device,
            p.m, p.n, p.k,
            p.lda, p.ldb, p.ldc,
            static_cast<int>(p.trans_a),
            static_cast<int>(p.trans_b),
            static_cast<int>(normalized_type_a),
            static_cast<int>(normalized_type_b),
            static_cast<int>(typeC)
        };

        std::shared_ptr<Fp8Plan> plan;

        {
            std::lock_guard<std::mutex> lock(fp8_plan_cache_mutex());
            auto& cache = fp8_plan_cache();
            const auto found = cache.find(key);

            if (found != cache.end()) {
                plan = found->second;
            }
        }

        if (!plan) {
            std::shared_ptr<Fp8Plan> candidate;
            std::string error;

            if (!create_fp8_plan(
                    api, handle, p,
                    normalized_type_a,
                    normalized_type_b,
                    typeC,
                    candidate, error)) {
                debug_log(error);
                return false;
            }

            std::lock_guard<std::mutex> lock(fp8_plan_cache_mutex());
            auto& cache = fp8_plan_cache();
            const auto inserted = cache.emplace(key, candidate);
            plan = inserted.first->second;
        }

        // Scale attributes belong to the retained descriptor. Protect their
        // update and the corresponding matmul as one operation.
        std::lock_guard<std::mutex> plan_lock(plan->mutex);

        const void* scale_a_pointer = normalized_scale_a;
        const void* scale_b_pointer = normalized_scale_b;

        int status = api.lt_desc_set_attribute(
            plan->resources->desc,
            kLtDescAScalePointer,
            &scale_a_pointer,
            sizeof(scale_a_pointer));

        if (status != kHipblasLtSuccess) {
            debug_log(
                "Setting FP8 A scale pointer failed, status=" +
                std::to_string(status));
            return false;
        }

        status = api.lt_desc_set_attribute(
            plan->resources->desc,
            kLtDescBScalePointer,
            &scale_b_pointer,
            sizeof(scale_b_pointer));

        if (status != kHipblasLtSuccess) {
            debug_log(
                "Setting FP8 B scale pointer failed, status=" +
                std::to_string(status));
            return false;
        }

        status = api.lt_matmul(
            handle,
            plan->resources->desc,
            &alpha,
            p.A, plan->resources->layout_a,
            p.B, plan->resources->layout_b,
            &beta,
            C, plan->resources->layout_c,
            C, plan->resources->layout_d,
            &plan->algo,
            nullptr, 0,
            stream);

        if (status != kHipblasLtSuccess) {
            debug_log(
                "FP8 hipblasLtMatmul failed, status=" +
                std::to_string(status));
            return false;
        }

        return true;

    } catch (const std::exception& error) {
        debug_log(
            std::string("Scaled FP8 GEMM failed: ") + error.what());
        return false;
    } catch (...) {
        debug_log("Scaled FP8 GEMM failed with an unknown error");
        return false;
    }
}

} // namespace rocblas_gemm_select
} // namespace clap
