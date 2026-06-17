#include "skippy.h"
#include "skippy/devices.h"
#include "skippy-signals.h"

#include <mutex>
#include "gguf.h"
#include "llama-arch.h"
#include "llama-context.h"
#include "llama-ext.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <cstdio>
#include <limits>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

struct skippy_model {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_context * mtp_ctx = nullptr;
    skippy_runtime_config config = {};
    bool executable = true;
    uint32_t lane_count = 1;
    std::vector<bool> lane_in_use;
    std::vector<std::vector<llama_token>> lane_resident_prefix_tokens;
};

struct skippy_session {
    skippy_model * stage_model = nullptr;
    llama_context * ctx = nullptr;
    int32_t seq_id = 0;
    int32_t checkpoint_seq_id = 1;
    int32_t n_past = 0;
    bool borrowed_sequence = false;
    int32_t borrowed_prefix_tokens = 0;
    bool preserve_prefix_on_free = false;
    int32_t preserve_prefix_tokens = 0;
    bool checkpoint_valid = false;
    int32_t checkpoint_n_past = 0;
    size_t checkpoint_token_history_size = 0;
    size_t checkpoint_signal_history_size = 0;
    std::vector<llama_token> token_history;
    std::vector<skippy_token_signal> signal_history;
    llama_sampler * sampling_chain = nullptr;
    llama_sampler * grammar_sampler = nullptr;
    skippy_sampling_config sampling_config = {};
    bool sampling_config_valid = false;
    std::string chat_sampling_metadata;
    uint64_t grammar_generated_start = 0;
    size_t sampling_accepted_token_count = 0;
    int32_t mtp_next_pos = 0;
    bool mtp_has_pending_h = false;
    std::vector<float> mtp_pending_h;
    bool mtp_has_pending_draft = false;
    llama_pos mtp_pending_draft_pos = 0;
    llama_token mtp_pending_draft_token = -1;
};

struct skippy_tensor_meta {
    std::string name;
    int32_t layer_index = -1;
    skippy_tensor_role role = SKIPPY_TENSOR_ROLE_UNKNOWN;
    ggml_type type = GGML_TYPE_COUNT;
    std::vector<int64_t> ne;
    uint64_t offset = 0;
    uint64_t size = 0;
};

static uint64_t skippy_tensor_element_count(const skippy_tensor_meta & tensor) {
    uint64_t count = 1;
    for (const int64_t dim : tensor.ne) {
        if (dim <= 0 || count > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
            return 0;
        }
        count *= static_cast<uint64_t>(dim);
    }
    return tensor.ne.empty() ? 0 : count;
}

struct skippy_slice_range {
    int32_t stage_index = -1;
    int32_t layer_start = 0;
    int32_t layer_end = 0;
    bool include_embeddings = false;
    bool include_output = false;
};

struct skippy_slice_plan {
    std::vector<skippy_slice_range> ranges;
};

struct skippy_source_tensor {
    const skippy_model_info * source = nullptr;
    const skippy_tensor_meta * tensor = nullptr;
};

struct skippy_model_info {
    gguf_context * ctx = nullptr;
    std::string path;
    std::vector<skippy_tensor_meta> tensors;
};

static skippy_error * skippy_make_error(enum skippy_status status, const char * message) {
    skippy_error * error = new skippy_error{};
    error->status = status;

    if (message == nullptr) {
        error->message = nullptr;
        return error;
    }

    const size_t len = std::strlen(message);
    char * owned = static_cast<char *>(std::malloc(len + 1));
    if (owned == nullptr) {
        error->status = SKIPPY_STATUS_ERROR;
        error->message = nullptr;
        return error;
    }

    std::memcpy(owned, message, len + 1);
    error->message = owned;
    return error;
}

static void skippy_set_error(
        skippy_error ** out_error,
        enum skippy_status status,
        const char * message) {
    if (out_error != nullptr) {
        *out_error = skippy_make_error(status, message);
    }
}

static void skippy_disable_chat_grammar_after_exception(
        skippy_session * session,
        const char * operation,
        const char * message) {
    if (session == nullptr || session->grammar_sampler == nullptr) {
        return;
    }
    fprintf(stderr, "skippy: disabling chat grammar after %s failed: %s\n",
            operation != nullptr ? operation : "grammar operation",
            message != nullptr ? message : "unknown native exception");
    llama_sampler_free(session->grammar_sampler);
    session->grammar_sampler = nullptr;
}

static enum skippy_status skippy_success(skippy_error ** out_error) {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    return SKIPPY_STATUS_OK;
}

static int32_t skippy_stage_layer_count(const llama_model * model) {
    if (model == nullptr) {
        return 0;
    }
    const auto & hparams = model->hparams;
    if (hparams.n_layer_nextn > 0) {
        return static_cast<int32_t>(hparams.n_layer_all);
    }
    return llama_model_n_layer(model);
}

static enum skippy_backend_device_type skippy_backend_device_type_from_ggml(
        enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return SKIPPY_BACKEND_DEVICE_TYPE_CPU;
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return SKIPPY_BACKEND_DEVICE_TYPE_GPU;
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return SKIPPY_BACKEND_DEVICE_TYPE_IGPU;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return SKIPPY_BACKEND_DEVICE_TYPE_ACCEL;
        case GGML_BACKEND_DEVICE_TYPE_META:
            return SKIPPY_BACKEND_DEVICE_TYPE_META;
    }
    return SKIPPY_BACKEND_DEVICE_TYPE_ACCEL;
}

static uint64_t skippy_backend_device_caps_from_ggml(const ggml_backend_dev_caps & caps) {
    uint64_t out = 0;
    if (caps.async) {
        out |= SKIPPY_BACKEND_DEVICE_CAP_ASYNC;
    }
    if (caps.host_buffer) {
        out |= SKIPPY_BACKEND_DEVICE_CAP_HOST_BUFFER;
    }
    if (caps.buffer_from_host_ptr) {
        out |= SKIPPY_BACKEND_DEVICE_CAP_BUFFER_FROM_HOST_PTR;
    }
    if (caps.events) {
        out |= SKIPPY_BACKEND_DEVICE_CAP_EVENTS;
    }
    return out;
}

static uint64_t skippy_runtime_event_timestamp_mono_ns() {
    const int64_t now_us = ggml_time_us();
    return now_us > 0 ? static_cast<uint64_t>(now_us) * UINT64_C(1000) : 0;
}

static uint32_t skippy_runtime_event_failure_code_from_status(enum skippy_status status) {
    switch (status) {
        case SKIPPY_STATUS_OK:
            return SKIPPY_RUNTIME_EVENT_FAILURE_NONE;
        case SKIPPY_STATUS_INVALID_ARGUMENT:
            return SKIPPY_RUNTIME_EVENT_FAILURE_INVALID_ARGUMENT;
        case SKIPPY_STATUS_IO_ERROR:
            return SKIPPY_RUNTIME_EVENT_FAILURE_IO_ERROR;
        case SKIPPY_STATUS_MODEL_ERROR:
            return SKIPPY_RUNTIME_EVENT_FAILURE_MODEL_ERROR;
        case SKIPPY_STATUS_RUNTIME_ERROR:
        case SKIPPY_STATUS_UNSUPPORTED:
            return SKIPPY_RUNTIME_EVENT_FAILURE_RUNTIME_ERROR;
        case SKIPPY_STATUS_ERROR:
        case SKIPPY_STATUS_BUFFER_TOO_SMALL:
            return SKIPPY_RUNTIME_EVENT_FAILURE_INTERNAL_ERROR;
    }

    return SKIPPY_RUNTIME_EVENT_FAILURE_INTERNAL_ERROR;
}

struct skippy_runtime_event_scope {
    skippy_runtime_event_reporter_v1 reporter = {};
    uint64_t sequence = 0;
    uint64_t stage_id = 0;
    uint64_t model_id = 0;
    uint64_t last_progress_step = 0;
    bool active = false;
    bool backend_emitted = false;

    skippy_runtime_event_scope(
            const skippy_runtime_config * config,
            const skippy_runtime_event_reporter_v1 * reporter_in) {
        if (config != nullptr && config->stage_index >= 0) {
            stage_id = static_cast<uint64_t>(config->stage_index);
        }

        if (reporter_in == nullptr || reporter_in->callback == nullptr) {
            return;
        }

        reporter = *reporter_in;
        active = true;
    }

    ~skippy_runtime_event_scope() {
        clear();
    }

    void clear() {
        active = false;
        reporter = {};
        sequence = 0;
        stage_id = 0;
        model_id = 0;
        last_progress_step = 0;
        backend_emitted = false;
    }

    bool is_enabled() const {
        return active && reporter.callback != nullptr;
    }

    void emit(
            uint32_t category,
            uint32_t kind,
            uint32_t emitter,
            uint64_t progress_current,
            uint64_t progress_total,
            uint32_t progress_unit,
            uint32_t failure_code,
            enum skippy_status status,
            const char * detail_ptr,
            uint64_t detail_len) {
        if (!is_enabled()) {
            return;
        }

        skippy_runtime_event_v1 event = {
            SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION,
            static_cast<uint32_t>(sizeof(skippy_runtime_event_v1)),
            category,
            kind,
            emitter,
            0,
            ++sequence,
            skippy_runtime_event_timestamp_mono_ns(),
            model_id,
            stage_id,
            0,
            progress_current,
            progress_total,
            progress_unit,
            failure_code,
            static_cast<int32_t>(status),
            0,
            detail_ptr,
            detail_len,
        };
        reporter.callback(&event, reporter.user_data);
    }

    void emit_detail(
            uint32_t category,
            uint32_t kind,
            uint32_t emitter,
            enum skippy_status status,
            const char * detail_ptr) {
        emit(
                category,
                kind,
                emitter,
                0,
                0,
                SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_NONE,
                SKIPPY_RUNTIME_EVENT_FAILURE_NONE,
                status,
                detail_ptr,
                detail_ptr != nullptr ? static_cast<uint64_t>(std::strlen(detail_ptr)) : 0);
    }

    void emit_started(const char * detail_ptr) {
        emit_detail(
                SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN,
                SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_STARTED,
                SKIPPY_RUNTIME_EVENT_EMITTER_OPEN_THREAD,
                SKIPPY_STATUS_OK,
                detail_ptr);
    }

    void emit_progress(float progress) {
        if (!is_enabled()) {
            return;
        }

        progress = std::max(0.0f, std::min(progress, 1.0f));
        const uint64_t scaled = static_cast<uint64_t>(std::llround(progress * 1000.0f));
        if (scaled <= last_progress_step) {
            return;
        }

        last_progress_step = scaled;
        emit(
                SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN,
                SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_PROGRESS,
                SKIPPY_RUNTIME_EVENT_EMITTER_WORKER_THREAD,
                scaled,
                1000,
                SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_STEPS,
                SKIPPY_RUNTIME_EVENT_FAILURE_NONE,
                SKIPPY_STATUS_OK,
                nullptr,
                0);
    }

    void emit_backend_selected(const char * detail_ptr) {
        if (!is_enabled() || backend_emitted || detail_ptr == nullptr || detail_ptr[0] == '\0') {
            return;
        }

        backend_emitted = true;
        emit_detail(
                SKIPPY_RUNTIME_EVENT_CATEGORY_BACKEND,
                SKIPPY_RUNTIME_EVENT_KIND_BACKEND_DEVICE_SELECTED,
                SKIPPY_RUNTIME_EVENT_EMITTER_OPEN_THREAD,
                SKIPPY_STATUS_OK,
                detail_ptr);
    }

    void emit_finished() {
        emit(
                SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN,
                SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_FINISHED,
                SKIPPY_RUNTIME_EVENT_EMITTER_OPEN_THREAD,
                0,
                0,
                SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_NONE,
                SKIPPY_RUNTIME_EVENT_FAILURE_NONE,
                SKIPPY_STATUS_OK,
                nullptr,
                0);
    }

    void emit_failure(enum skippy_status status, const char * detail_ptr) {
        emit(
                SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN,
                SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_FAILED_HANDLED,
                SKIPPY_RUNTIME_EVENT_EMITTER_OPEN_THREAD,
                0,
                0,
                SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_NONE,
                skippy_runtime_event_failure_code_from_status(status),
                status,
                detail_ptr,
                detail_ptr != nullptr ? static_cast<uint64_t>(std::strlen(detail_ptr)) : 0);
    }
};

static enum skippy_status skippy_validate_runtime_event_reporter(
        const struct skippy_runtime_event_reporter_v1 * reporter,
        struct skippy_error ** out_error) {
    if (reporter == nullptr) {
        return SKIPPY_STATUS_OK;
    }
    if (reporter->struct_size < sizeof(struct skippy_runtime_event_reporter_v1)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "runtime event reporter struct is too small");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (reporter->callback == nullptr) {
        return SKIPPY_STATUS_OK;
    }
    if (reporter->abi_version != SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "unsupported runtime event reporter ABI version");
        return SKIPPY_STATUS_UNSUPPORTED;
    }
    return SKIPPY_STATUS_OK;
}

static bool skippy_model_open_progress_callback(float progress, void * user_data) {
    skippy_runtime_event_scope * scope = static_cast<skippy_runtime_event_scope *>(user_data);
    if (scope != nullptr) {
        scope->emit_progress(progress);
    }
    return true;
}

static int32_t skippy_layer_from_name(const char * name) {
    if (name == nullptr) {
        return -1;
    }

    static const std::regex blk_pattern("^blk\\.([0-9]+)\\.");
    static const std::regex cache_pattern("^cache_.*_l([0-9]+)$");

    std::cmatch match;
    if (std::regex_search(name, match, blk_pattern)) {
        return static_cast<int32_t>(std::stol(match[1].str()));
    }
    if (std::regex_search(name, match, cache_pattern)) {
        return static_cast<int32_t>(std::stol(match[1].str()));
    }

    return -1;
}

static enum skippy_tensor_role skippy_role_from_name(const char * name, int32_t layer_index) {
    if (name == nullptr) {
        return SKIPPY_TENSOR_ROLE_UNKNOWN;
    }

    const std::string tensor_name(name);
    if (layer_index >= 0) {
        return SKIPPY_TENSOR_ROLE_LAYER;
    }
    if (tensor_name == "token_embd.weight" || tensor_name == "per_layer_token_embd.weight") {
        return SKIPPY_TENSOR_ROLE_EMBEDDING;
    }
    if (tensor_name == "output.weight" || tensor_name == "output.bias") {
        return SKIPPY_TENSOR_ROLE_OUTPUT;
    }
    if (tensor_name == "output_norm.weight" || tensor_name == "output_norm.bias") {
        return SKIPPY_TENSOR_ROLE_FINAL_NORM;
    }

    return SKIPPY_TENSOR_ROLE_METADATA;
}

template <typename T>
static bool skippy_read(FILE * file, T & out) {
    return std::fread(&out, 1, sizeof(T), file) == sizeof(T);
}

static bool skippy_skip(FILE * file, uint64_t bytes) {
    return std::fseek(file, static_cast<long>(bytes), SEEK_CUR) == 0;
}

static bool skippy_read_string(FILE * file, std::string & out) {
    uint64_t len = 0;
    if (!skippy_read(file, len)) {
        return false;
    }
    out.resize(static_cast<size_t>(len));
    return len == 0 || std::fread(out.data(), 1, static_cast<size_t>(len), file) == len;
}

static bool skippy_gguf_type_size(gguf_type type, size_t & out_size) {
    switch (type) {
        case GGUF_TYPE_UINT8:   out_size = sizeof(uint8_t);  return true;
        case GGUF_TYPE_INT8:    out_size = sizeof(int8_t);   return true;
        case GGUF_TYPE_UINT16:  out_size = sizeof(uint16_t); return true;
        case GGUF_TYPE_INT16:   out_size = sizeof(int16_t);  return true;
        case GGUF_TYPE_UINT32:  out_size = sizeof(uint32_t); return true;
        case GGUF_TYPE_INT32:   out_size = sizeof(int32_t);  return true;
        case GGUF_TYPE_FLOAT32: out_size = sizeof(float);    return true;
        case GGUF_TYPE_BOOL:    out_size = sizeof(int8_t);   return true;
        case GGUF_TYPE_UINT64:  out_size = sizeof(uint64_t); return true;
        case GGUF_TYPE_INT64:   out_size = sizeof(int64_t);  return true;
        case GGUF_TYPE_FLOAT64: out_size = sizeof(double);   return true;
        case GGUF_TYPE_STRING:
        case GGUF_TYPE_ARRAY:
        case GGUF_TYPE_COUNT:
            return false;
    }

    return false;
}

static bool skippy_skip_gguf_value(FILE * file, gguf_type type);

static bool skippy_skip_gguf_array(FILE * file) {
    int32_t raw_type = 0;
    uint64_t count = 0;
    if (!skippy_read(file, raw_type) || !skippy_read(file, count)) {
        return false;
    }

    const gguf_type type = static_cast<gguf_type>(raw_type);
    if (type == GGUF_TYPE_STRING) {
        for (uint64_t i = 0; i < count; ++i) {
            std::string ignored;
            if (!skippy_read_string(file, ignored)) {
                return false;
            }
        }
        return true;
    }

    size_t element_size = 0;
    if (!skippy_gguf_type_size(type, element_size)) {
        return false;
    }
    if (count > UINT64_MAX / element_size) {
        return false;
    }
    return skippy_skip(file, count * element_size);
}

static bool skippy_skip_gguf_value(FILE * file, gguf_type type) {
    if (type == GGUF_TYPE_STRING) {
        std::string ignored;
        return skippy_read_string(file, ignored);
    }
    if (type == GGUF_TYPE_ARRAY) {
        return skippy_skip_gguf_array(file);
    }

    size_t value_size = 0;
    if (!skippy_gguf_type_size(type, value_size)) {
        return false;
    }
    return skippy_skip(file, value_size);
}

static bool skippy_parse_tensor_metadata(skippy_model_info * info) {
    FILE * file = ggml_fopen(info->path.c_str(), "rb");
    if (file == nullptr) {
        return false;
    }

    char magic[4] = {};
    uint32_t version = 0;
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    bool ok = std::fread(magic, 1, sizeof(magic), file) == sizeof(magic) &&
            std::memcmp(magic, GGUF_MAGIC, sizeof(magic)) == 0 &&
            skippy_read(file, version) &&
            version == GGUF_VERSION &&
            skippy_read(file, n_tensors) &&
            skippy_read(file, n_kv) &&
            n_tensors >= 0 &&
            n_kv >= 0;

    for (int64_t i = 0; ok && i < n_kv; ++i) {
        std::string key;
        int32_t raw_type = 0;
        ok = skippy_read_string(file, key) &&
                skippy_read(file, raw_type) &&
                skippy_skip_gguf_value(file, static_cast<gguf_type>(raw_type));
    }

    info->tensors.clear();
    info->tensors.reserve(static_cast<size_t>(std::max<int64_t>(n_tensors, 0)));
    for (int64_t i = 0; ok && i < n_tensors; ++i) {
        skippy_tensor_meta meta;
        uint32_t n_dims = 0;
        int32_t raw_type = 0;
        ok = skippy_read_string(file, meta.name) && skippy_read(file, n_dims);
        if (!ok || n_dims > GGML_MAX_DIMS) {
            ok = false;
            break;
        }

        meta.ne.resize(n_dims);
        for (uint32_t dim = 0; ok && dim < n_dims; ++dim) {
            ok = skippy_read(file, meta.ne[dim]);
        }
        ok = ok && skippy_read(file, raw_type) && skippy_read(file, meta.offset);
        if (!ok) {
            break;
        }

        meta.type = static_cast<ggml_type>(raw_type);
        meta.layer_index = skippy_layer_from_name(meta.name.c_str());
        meta.role = skippy_role_from_name(meta.name.c_str(), meta.layer_index);
        const int64_t tensor_id = gguf_find_tensor(info->ctx, meta.name.c_str());
        if (tensor_id < 0) {
            ok = false;
            break;
        }
        meta.size = static_cast<uint64_t>(gguf_get_tensor_size(info->ctx, tensor_id));
        info->tensors.push_back(std::move(meta));
    }

    std::fclose(file);
    return ok && info->tensors.size() == static_cast<size_t>(n_tensors);
}

static bool skippy_tensor_selected(const skippy_tensor_meta & tensor, const skippy_slice_range & range) {
    if (tensor.layer_index >= 0) {
        return tensor.layer_index >= range.layer_start && tensor.layer_index < range.layer_end;
    }

    switch (tensor.role) {
        case SKIPPY_TENSOR_ROLE_EMBEDDING:
            return range.include_embeddings;
        case SKIPPY_TENSOR_ROLE_FINAL_NORM:
        case SKIPPY_TENSOR_ROLE_OUTPUT:
            return range.include_output;
        case SKIPPY_TENSOR_ROLE_METADATA:
        case SKIPPY_TENSOR_ROLE_TOKENIZER:
        case SKIPPY_TENSOR_ROLE_UNKNOWN:
            return true;
        case SKIPPY_TENSOR_ROLE_LAYER:
            return false;
    }

    return false;
}

static size_t skippy_align(size_t value, size_t alignment) {
    return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

static void skippy_clear_split_metadata(gguf_context * ctx) {
    gguf_remove_key(ctx, "split.no");
    gguf_remove_key(ctx, "split.count");
    gguf_remove_key(ctx, "split.tensors.count");
}

static skippy_model_info * skippy_model_info_open_owned(const char * path) {
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * ctx = gguf_init_from_file(path, params);
    if (ctx == nullptr) {
        return nullptr;
    }

    skippy_model_info * info = new skippy_model_info{};
    info->ctx = ctx;
    info->path = path;
    if (!skippy_parse_tensor_metadata(info)) {
        gguf_free(ctx);
        delete info;
        return nullptr;
    }
    return info;
}

static void skippy_model_info_delete(skippy_model_info * info) {
    if (info != nullptr) {
        gguf_free(info->ctx);
        delete info;
    }
}

static enum skippy_status skippy_add_tensor_to_context(
        ggml_context * ggml_ctx,
        gguf_context * out_ctx,
        const skippy_tensor_meta & meta,
        struct skippy_error ** out_error) {
    int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
    for (size_t dim = 0; dim < meta.ne.size() && dim < GGML_MAX_DIMS; ++dim) {
        ne[dim] = meta.ne[dim];
    }
    ggml_tensor * tensor = ggml_new_tensor(
            ggml_ctx,
            meta.type,
            static_cast<int>(meta.ne.size()),
            ne);
    if (tensor == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_ERROR, "failed to allocate output tensor metadata");
        return SKIPPY_STATUS_ERROR;
    }
    ggml_set_name(tensor, meta.name.c_str());
    gguf_add_tensor(out_ctx, tensor);
    return SKIPPY_STATUS_OK;
}

static enum skippy_status skippy_copy_source_tensors(
        const std::vector<skippy_source_tensor> & selected,
        const char * output_path,
        gguf_context * out_ctx,
        struct skippy_error ** out_error) {
    if (!gguf_write_to_file(out_ctx, output_path, true)) {
        skippy_set_error(out_error, SKIPPY_STATUS_IO_ERROR, "failed to write output GGUF metadata");
        return SKIPPY_STATUS_IO_ERROR;
    }

    FILE * output = ggml_fopen(output_path, "ab");
    if (output == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_IO_ERROR, "failed to open output GGUF for tensor copy");
        return SKIPPY_STATUS_IO_ERROR;
    }

    const size_t output_alignment = gguf_get_alignment(out_ctx);
    std::vector<uint8_t> buffer(1024 * 1024);
    std::vector<uint8_t> zeroes(output_alignment, 0);
    bool ok = true;

    for (const skippy_source_tensor & item : selected) {
        FILE * input = ggml_fopen(item.source->path.c_str(), "rb");
        if (input == nullptr) {
            ok = false;
            break;
        }

        const size_t source_data_offset = gguf_get_data_offset(item.source->ctx);
        if (std::fseek(input, static_cast<long>(source_data_offset + item.tensor->offset), SEEK_SET) != 0) {
            std::fclose(input);
            ok = false;
            break;
        }

        uint64_t remaining = item.tensor->size;
        while (remaining > 0) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
            if (std::fread(buffer.data(), 1, chunk, input) != chunk ||
                    std::fwrite(buffer.data(), 1, chunk, output) != chunk) {
                ok = false;
                break;
            }
            remaining -= chunk;
        }

        if (std::fclose(input) != 0) {
            ok = false;
        }
        if (!ok) {
            break;
        }

        const size_t padded = skippy_align(static_cast<size_t>(item.tensor->size), output_alignment);
        const size_t padding = padded - static_cast<size_t>(item.tensor->size);
        if (padding > 0 && std::fwrite(zeroes.data(), 1, padding, output) != padding) {
            ok = false;
            break;
        }
    }

    if (std::fclose(output) != 0) {
        ok = false;
    }

    if (!ok) {
        skippy_set_error(out_error, SKIPPY_STATUS_IO_ERROR, "failed to copy selected GGUF tensor data");
        return SKIPPY_STATUS_IO_ERROR;
    }

    return SKIPPY_STATUS_OK;
}

static bool skippy_is_full_model_config(const struct skippy_runtime_config * config) {
    if (config == nullptr) {
        return true;
    }

    return !config->filter_tensors_on_load && config->layer_start == 0;
}

static int32_t skippy_default_thread_count() {
#ifdef __linux__
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu = 0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings(
                "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break;
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t physical_cores = 0;
    size_t len = sizeof(physical_cores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &physical_cores, &len, NULL, 0) == 0 && physical_cores > 0) {
        return physical_cores;
    }
    if (sysctlbyname("hw.physicalcpu", &physical_cores, &len, NULL, 0) == 0 && physical_cores > 0) {
        return physical_cores;
    }
#endif
    const unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? static_cast<int32_t>(n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

static enum skippy_status skippy_apply_selected_backend_device(
        const struct skippy_runtime_config * config,
        llama_model_params & params,
        std::vector<ggml_backend_dev_t> & selected_devices,
        struct skippy_error ** out_error) {
    if (config == nullptr || config->selected_backend_device == nullptr || config->selected_backend_device[0] == '\0') {
        return SKIPPY_STATUS_OK;
    }

    const std::string device_name(config->selected_backend_device);
    if (device_name == "CPU" || device_name == "CPU0") {
        params.devices = nullptr;
        params.main_gpu = -1;
        params.n_gpu_layers = 0;
        return SKIPPY_STATUS_OK;
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_name(device_name.c_str());
    if (dev == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, ("unknown selected backend device: " + device_name).c_str());
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, ("selected backend device is CPU; use CPU or CPU0: " + device_name).c_str());
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    selected_devices.clear();
    selected_devices.push_back(dev);
    selected_devices.push_back(nullptr);
    params.devices = selected_devices.data();
    params.main_gpu = 0;
    return SKIPPY_STATUS_OK;
}

static void skippy_emit_observable_backend_device(
        skippy_runtime_event_scope * event_scope,
        const struct skippy_runtime_config * config,
        const llama_model * model) {
    if (event_scope == nullptr || !event_scope->is_enabled()) {
        return;
    }
    if (config != nullptr && config->selected_backend_device != nullptr && config->selected_backend_device[0] != '\0') {
        event_scope->emit_backend_selected(config->selected_backend_device);
        return;
    }
    if (model != nullptr && !model->devices.empty()) {
        event_scope->emit_backend_selected(ggml_backend_dev_name(model->devices.front().dev));
    }
}

struct skippy_filter_scope {
    explicit skippy_filter_scope(const skippy_runtime_config * config) {
        if (config != nullptr && config->filter_tensors_on_load) {
            llama_model_loader_stage_filter filter;
            filter.enabled = true;
            filter.layer_start = config->layer_start;
            filter.layer_end = config->layer_end;
            if (config->include_output && filter.layer_end < std::numeric_limits<int32_t>::max()) {
                filter.layer_end += 1;
            }
            filter.include_embeddings = config->include_embeddings;
            filter.include_output = config->include_output;
            llama_model_loader_set_stage_filter(filter);
            enabled = true;
        }
    }

    ~skippy_filter_scope() {
        if (enabled) {
            llama_model_loader_clear_stage_filter();
        }
    }

    bool enabled = false;
};

struct skippy_graph_filter_scope {
    explicit skippy_graph_filter_scope(const skippy_runtime_config * config) {
        if (config != nullptr && config->filter_tensors_on_load) {
            skippy_graph_filter filter;
            filter.enabled = true;
            filter.layer_start = config->layer_start;
            filter.layer_end = config->layer_end;
            filter.include_embeddings = config->include_embeddings;
            filter.include_output = config->include_output;
            skippy_graph_set_filter(filter);
            enabled = true;
        }
    }

    ~skippy_graph_filter_scope() {
        if (enabled) {
            skippy_graph_clear_filter();
        }
    }

    bool enabled = false;
};

struct skippy_activation_tokens_scope {
    skippy_activation_tokens_scope(const llama_token * tokens, size_t token_count) {
        if (tokens != nullptr && token_count > 0 && token_count <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            skippy_activation_tokens stage_tokens;
            stage_tokens.tokens = tokens;
            stage_tokens.token_count = static_cast<uint32_t>(token_count);
            skippy_graph_set_activation_tokens(stage_tokens);
            enabled = true;
        }
    }

    ~skippy_activation_tokens_scope() {
        if (enabled) {
            skippy_graph_clear_activation_tokens();
        }
    }

    bool enabled = false;
};

struct skippy_rwkv7_v_first_scope {
    skippy_rwkv7_v_first_scope(
            const skippy_activation_desc * desc,
            const void * payload,
            size_t hidden_bytes,
            int32_t n_embd) {
        if (desc != nullptr &&
            payload != nullptr &&
            (desc->flags & SKIPPY_ACTIVATION_FLAG_RWKV7_V_FIRST) != 0 &&
            desc->payload_bytes >= hidden_bytes) {
            skippy_activation_rwkv7_v_first sideband;
            sideband.values = reinterpret_cast<const float *>(static_cast<const uint8_t *>(payload) + hidden_bytes);
            sideband.token_count = desc->token_count;
            sideband.n_embd = static_cast<uint32_t>(n_embd);
            skippy_graph_set_rwkv7_v_first(sideband);
            enabled = true;
        }
    }

    ~skippy_rwkv7_v_first_scope() {
        if (enabled) {
            skippy_graph_clear_rwkv7_v_first();
        }
    }

    bool enabled = false;
};

struct skippy_gemma3n_altup_scope {
    skippy_gemma3n_altup_scope(
            const skippy_activation_desc * desc,
            const void * payload,
            int32_t n_embd,
            int32_t n_altup) {
        if (desc != nullptr &&
            payload != nullptr &&
            (desc->flags & SKIPPY_ACTIVATION_FLAG_GEMMA3N_ALTUP) != 0) {
            skippy_activation_gemma3n_altup sideband;
            sideband.values = reinterpret_cast<const float *>(payload);
            sideband.token_count = desc->token_count;
            sideband.n_embd = static_cast<uint32_t>(n_embd);
            sideband.n_altup = static_cast<uint32_t>(n_altup);
            skippy_graph_set_gemma3n_altup(sideband);
            enabled = true;
        }
    }

    ~skippy_gemma3n_altup_scope() {
        if (enabled) {
            skippy_graph_clear_gemma3n_altup();
        }
    }

    bool enabled = false;
};

static bool skippy_is_filtered(const skippy_session * session) {
    return session != nullptr &&
           session->stage_model != nullptr &&
           session->stage_model->config.filter_tensors_on_load;
}

static bool skippy_emits_activation_frame(const skippy_session * session) {
    return skippy_is_filtered(session) && !session->stage_model->config.include_output;
}

static bool skippy_is_rwkv7_activation_model(const skippy_session * session) {
    if (session == nullptr || session->stage_model == nullptr || session->stage_model->model == nullptr) {
        return false;
    }
    const llm_arch arch = session->stage_model->model->arch;
    return arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7;
}

static bool skippy_is_gemma3n_activation_model(const skippy_session * session) {
    if (session == nullptr || session->stage_model == nullptr || session->stage_model->model == nullptr) {
        return false;
    }
    return session->stage_model->model->arch == LLM_ARCH_GEMMA3N;
}

static size_t skippy_activation_hidden_bytes(const skippy_session * session, size_t token_count) {
    if (session == nullptr || session->stage_model == nullptr || session->stage_model->model == nullptr) {
        return 0;
    }

    return token_count *
           static_cast<size_t>(llama_model_n_embd(session->stage_model->model)) *
           sizeof(float);
}

static size_t skippy_gemma3n_altup_bytes(const skippy_session * session, size_t token_count) {
    if (session == nullptr || session->stage_model == nullptr || session->stage_model->model == nullptr) {
        return 0;
    }
    const llama_hparams & hparams = session->stage_model->model->hparams;
    return token_count *
           static_cast<size_t>(hparams.n_embd) *
           static_cast<size_t>(hparams.n_altup) *
           sizeof(float);
}

static uint64_t skippy_output_activation_flags(
        const skippy_session * session,
        const skippy_activation_desc * input_desc) {
    if (skippy_emits_activation_frame(session) && skippy_is_gemma3n_activation_model(session)) {
        return SKIPPY_ACTIVATION_FLAG_GEMMA3N_ALTUP;
    }
    if (!skippy_emits_activation_frame(session) || !skippy_is_rwkv7_activation_model(session)) {
        return 0;
    }

    const skippy_runtime_config & config = session->stage_model->config;
    if (config.layer_start == 0 && config.layer_end > 0) {
        return SKIPPY_ACTIVATION_FLAG_RWKV7_V_FIRST;
    }
    if (input_desc != nullptr && (input_desc->flags & SKIPPY_ACTIVATION_FLAG_RWKV7_V_FIRST) != 0) {
        return SKIPPY_ACTIVATION_FLAG_RWKV7_V_FIRST;
    }
    return 0;
}

static size_t skippy_activation_payload_bytes(
        const skippy_session * session,
        size_t token_count,
        uint64_t flags) {
    const size_t hidden_bytes = skippy_activation_hidden_bytes(session, token_count);
    if ((flags & SKIPPY_ACTIVATION_FLAG_GEMMA3N_ALTUP) != 0) {
        return skippy_gemma3n_altup_bytes(session, token_count);
    }
    size_t payload_bytes = hidden_bytes;
    if ((flags & SKIPPY_ACTIVATION_FLAG_RWKV7_V_FIRST) != 0) {
        payload_bytes += hidden_bytes;
    }
    return payload_bytes;
}

static bool skippy_has_activation_payload(const skippy_activation_desc * desc, const void * payload) {
    return desc != nullptr && desc->payload_bytes > 0 && payload != nullptr;
}

static enum skippy_status skippy_validate_frame_input(
        skippy_session * session,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        size_t expected_token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const skippy_runtime_config & config = session->stage_model->config;
    if (!config.filter_tensors_on_load || config.layer_start == 0) {
        if (skippy_has_activation_payload(input_desc, input_payload)) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "this stage expects token input, not activation input");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        return SKIPPY_STATUS_OK;
    }

    if (input_desc == nullptr || input_payload == nullptr || input_desc->payload_bytes == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "non-first runtime slices require an activation frame input");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (input_desc->version != 1 ||
        input_desc->dtype != SKIPPY_ACTIVATION_DTYPE_F32 ||
        input_desc->layout != SKIPPY_ACTIVATION_LAYOUT_TOKEN_MAJOR) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame must be version 1 F32 token-major");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (input_desc->sequence_count != 1 || input_desc->token_count != expected_token_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame token or sequence count does not match request");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (input_desc->layer_end != config.layer_start) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame layer_end must match this stage layer_start");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const uint64_t supported_flags = SKIPPY_ACTIVATION_FLAG_RWKV7_V_FIRST | SKIPPY_ACTIVATION_FLAG_GEMMA3N_ALTUP;
    if ((input_desc->flags & ~supported_flags) != 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame has unsupported sideband flags");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if ((input_desc->flags & SKIPPY_ACTIVATION_FLAG_GEMMA3N_ALTUP) != 0 && !skippy_is_gemma3n_activation_model(session)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "Gemma3n AltUp activation payload is only valid for Gemma3n stages");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_gemma3n_activation_model(session) && (input_desc->flags & SKIPPY_ACTIVATION_FLAG_GEMMA3N_ALTUP) == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "non-first Gemma3n runtime slices require AltUp activation payload");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if ((input_desc->flags & SKIPPY_ACTIVATION_FLAG_GEMMA3N_ALTUP) != 0 &&
        (input_desc->flags & SKIPPY_ACTIVATION_FLAG_RWKV7_V_FIRST) != 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "Gemma3n AltUp and RWKV7 v_first activation flags cannot be combined");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if ((input_desc->flags & SKIPPY_ACTIVATION_FLAG_RWKV7_V_FIRST) != 0 && !skippy_is_rwkv7_activation_model(session)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "RWKV7 v_first sideband is only valid for RWKV7 stages");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_rwkv7_activation_model(session) && (input_desc->flags & SKIPPY_ACTIVATION_FLAG_RWKV7_V_FIRST) == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "non-first RWKV7 runtime slices require v_first activation sideband");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const size_t expected_bytes = skippy_activation_payload_bytes(session, expected_token_count, input_desc->flags);
    if (input_desc->payload_bytes != expected_bytes) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame payload size does not match model hidden size");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    return SKIPPY_STATUS_OK;
}

static enum skippy_status skippy_prepare_output_activation_frame(
        skippy_session * session,
        size_t token_count,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        skippy_activation_desc * output_desc,
        const skippy_activation_desc * input_desc,
        struct skippy_error ** out_error) {
    const uint64_t output_flags = skippy_output_activation_flags(session, input_desc);
    const size_t payload_bytes = skippy_emits_activation_frame(session) ?
            skippy_activation_payload_bytes(session, token_count, output_flags) : 0;

    if (out_output_payload_bytes != nullptr) {
        *out_output_payload_bytes = payload_bytes;
    }

    if (payload_bytes > 0) {
        if (output_payload == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "output_payload is required for runtime-slice activation output");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (output_payload_capacity < payload_bytes) {
            skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "output activation buffer is too small");
            return SKIPPY_STATUS_BUFFER_TOO_SMALL;
        }
    } else if (output_payload_capacity > 0 && output_payload == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "output_payload is required when output capacity is non-zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    if (output_desc != nullptr) {
        *output_desc = {};
        output_desc->version = 1;
        output_desc->dtype = payload_bytes > 0 ? SKIPPY_ACTIVATION_DTYPE_F32 : SKIPPY_ACTIVATION_DTYPE_UNKNOWN;
        output_desc->layout = payload_bytes > 0 ? SKIPPY_ACTIVATION_LAYOUT_TOKEN_MAJOR : SKIPPY_ACTIVATION_LAYOUT_OPAQUE;
        output_desc->producer_stage_index = session != nullptr ? session->stage_model->config.stage_index : -1;
        output_desc->layer_start = session != nullptr ? session->stage_model->config.layer_start : 0;
        output_desc->layer_end = session != nullptr ? session->stage_model->config.layer_end : 0;
        output_desc->token_count = static_cast<uint32_t>(std::min<size_t>(token_count, std::numeric_limits<uint32_t>::max()));
        output_desc->sequence_count = token_count > 0 ? 1 : 0;
        output_desc->payload_bytes = payload_bytes;
        output_desc->flags = output_flags;
    }

    return SKIPPY_STATUS_OK;
}

static enum skippy_status skippy_decode_batch(
        skippy_session * session,
        llama_batch batch,
        size_t token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and at least one token are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_graph_filter_scope graph_filter_scope(&session->stage_model->config);
    const int32_t rc = llama_decode(session->ctx, batch);
    if (rc != 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama_decode failed");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    session->n_past += static_cast<int32_t>(token_count);
    return skippy_success(out_error);
}

static bool skippy_mtp_available(const skippy_session * session) {
    return session != nullptr &&
           session->stage_model != nullptr &&
           session->stage_model->mtp_ctx != nullptr &&
           session->stage_model->config.include_output;
}

static bool skippy_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "no") != 0;
}

static bool skippy_mtp_greedy_sampling_fastpath_enabled() {
    return skippy_env_enabled("SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH");
}

static void skippy_mtp_clear_session_state(skippy_session * session) {
    if (session == nullptr) {
        return;
    }
    session->mtp_next_pos = session->n_past;
    session->mtp_has_pending_h = false;
    session->mtp_pending_h.clear();
    session->mtp_has_pending_draft = false;
    session->mtp_pending_draft_pos = 0;
    session->mtp_pending_draft_token = -1;
    if (skippy_mtp_available(session)) {
        if (llama_memory_t memory = session->stage_model->mtp_ctx->get_memory()) {
            llama_memory_seq_rm(memory, session->seq_id, -1, -1);
        }
    }
}

static enum skippy_status skippy_mtp_sync_target_tokens(
        skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        llama_pos token_start,
        struct skippy_error ** out_error) {
    if (!skippy_mtp_available(session) || token_ids == nullptr || token_count == 0) {
        return skippy_success(out_error);
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "MTP token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_context * mtp_ctx = session->stage_model->mtp_ctx;
    const int32_t n_embd = llama_model_n_embd(session->stage_model->model);
    const size_t row_bytes = static_cast<size_t>(n_embd)*sizeof(float);
    if (session->mtp_pending_h.size() != static_cast<size_t>(n_embd)) {
        session->mtp_pending_h.assign(static_cast<size_t>(n_embd), 0.0f);
        session->mtp_has_pending_h = false;
    }

    if (session->mtp_has_pending_draft) {
        const llama_pos token_end = token_start + static_cast<llama_pos>(token_count);
        if (session->mtp_pending_draft_pos >= token_start &&
                session->mtp_pending_draft_pos < token_end) {
            const size_t draft_index = static_cast<size_t>(session->mtp_pending_draft_pos - token_start);
            if (token_ids[draft_index] == session->mtp_pending_draft_token) {
                session->mtp_next_pos = std::max(session->mtp_next_pos, session->mtp_pending_draft_pos + 1);
            } else {
                if (llama_memory_t memory = mtp_ctx->get_memory()) {
                    llama_memory_seq_rm(memory, session->seq_id, session->mtp_pending_draft_pos, -1);
                }
                session->mtp_next_pos = std::min(session->mtp_next_pos, static_cast<int32_t>(session->mtp_pending_draft_pos));
            }
        }
        session->mtp_has_pending_draft = false;
        session->mtp_pending_draft_pos = 0;
        session->mtp_pending_draft_token = -1;
    }

    size_t first_index = 0;
    if (session->mtp_next_pos > token_start) {
        first_index = static_cast<size_t>(std::min<llama_pos>(
                static_cast<llama_pos>(token_count),
                session->mtp_next_pos - token_start));
    }

    const int32_t n_decode = static_cast<int32_t>(token_count - first_index);
    if (n_decode > 0) {
        llama_batch batch = llama_batch_init(n_decode, n_embd, 1);
        batch.token = static_cast<llama_token *>(std::malloc(sizeof(llama_token)*static_cast<size_t>(n_decode)));
        if (batch.token == nullptr) {
            llama_batch_free(batch);
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to allocate MTP token batch");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        batch.n_tokens = n_decode;

        for (int32_t out_i = 0; out_i < n_decode; ++out_i) {
            const size_t src_i = first_index + static_cast<size_t>(out_i);
            batch.token[out_i] = token_ids[src_i];
            batch.pos[out_i] = token_start + static_cast<llama_pos>(src_i);
            batch.n_seq_id[out_i] = 1;
            batch.seq_id[out_i][0] = session->seq_id;
            batch.logits[out_i] = 0;

            float * dst = batch.embd + static_cast<size_t>(out_i)*n_embd;
            if (src_i == 0) {
                std::memcpy(dst, session->mtp_pending_h.data(), row_bytes);
            } else {
                const float * h_prev = llama_get_embeddings_nextn_ith(session->ctx, static_cast<int32_t>(src_i - 1));
                if (h_prev == nullptr) {
                    std::free(batch.token);
                    batch.token = nullptr;
                    llama_batch_free(batch);
                    skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "target pre-norm hidden row was not available");
                    return SKIPPY_STATUS_RUNTIME_ERROR;
                }
                std::memcpy(dst, h_prev, row_bytes);
            }
        }

        skippy_graph_filter_scope graph_filter_scope(&session->stage_model->config);
        const int32_t rc = llama_decode(mtp_ctx, batch);
        std::free(batch.token);
        batch.token = nullptr;
        llama_batch_free(batch);
        if (rc != 0) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama_decode failed for MTP sidecar sync");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        session->mtp_next_pos = token_start + static_cast<int32_t>(token_count);
    }

    const float * h_last = llama_get_embeddings_nextn_ith(session->ctx, static_cast<int32_t>(token_count - 1));
    if (h_last == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "target pre-norm hidden row was not available");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    std::memcpy(session->mtp_pending_h.data(), h_last, row_bytes);
    session->mtp_has_pending_h = true;
    return skippy_success(out_error);
}

static void skippy_record_tokens(
        skippy_session * session,
        const llama_token * token_ids,
        size_t token_count) {
    if (session != nullptr && token_ids != nullptr && token_count > 0) {
        session->token_history.insert(session->token_history.end(), token_ids, token_ids + token_count);
    }
}

static void skippy_record_signal(skippy_session * session, int32_t logits_index);

static enum skippy_status skippy_decode_tokens(
        skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        bool request_logits,
        struct skippy_error ** out_error) {
    (void) request_logits;

    if (session == nullptr || session->ctx == nullptr || token_ids == nullptr || token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and at least one token are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "non-first runtime slices require activation input");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int32_t n_tokens = static_cast<int32_t>(token_count);
    if (n_tokens == 1) {
        llama_token token = token_ids[0];
        llama_pos pos = session->n_past;
        int32_t n_seq_id = 1;
        llama_seq_id seq_id = session->seq_id;
        llama_seq_id * seq_ids = &seq_id;
        int8_t logits = request_logits ? 1 : 0;
        llama_batch batch = {
            /*n_tokens =*/ 1,
            /*token    =*/ &token,
            /*embd     =*/ nullptr,
            /*pos      =*/ &pos,
            /*n_seq_id =*/ &n_seq_id,
            /*seq_id   =*/ &seq_ids,
            /*logits   =*/ &logits,
        };
        enum skippy_status status = skippy_decode_batch(session, batch, 1, out_error);
        if (status == SKIPPY_STATUS_OK) {
            status = skippy_mtp_sync_target_tokens(session, token_ids, token_count, pos, out_error);
        }
        if (status == SKIPPY_STATUS_OK) {
            skippy_record_tokens(session, token_ids, token_count);
        }
        return status;
    }

    const llama_pos token_start = session->n_past;
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = token_ids[i];
        batch.pos[i] = session->n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = session->seq_id;
        batch.logits[i] = request_logits && i == n_tokens - 1 ? 1 : 0;
    }

    enum skippy_status status = skippy_decode_batch(session, batch, token_count, out_error);
    llama_batch_free(batch);
    if (status == SKIPPY_STATUS_OK) {
        status = skippy_mtp_sync_target_tokens(session, token_ids, token_count, token_start, out_error);
    }
    if (status == SKIPPY_STATUS_OK) {
        skippy_record_tokens(session, token_ids, token_count);
    }
    return status;
}

static enum skippy_status skippy_verify_token_batch(
        skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || token_ids == nullptr || token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and at least one token are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "non-first runtime slices require activation input");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int32_t n_tokens = static_cast<int32_t>(token_count);
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = token_ids[i];
        batch.pos[i] = session->n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = session->seq_id;
        batch.logits[i] = 1;
    }

    enum skippy_status status = skippy_decode_batch(session, batch, token_count, out_error);
    llama_batch_free(batch);
    return status;
}

static llama_token skippy_greedy_sample_context(
        llama_model * model,
        llama_context * ctx,
        int32_t index) {
    if (model == nullptr || ctx == nullptr) {
        return 0;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_synchronize(ctx);
    const float * logits = llama_get_logits_ith(ctx, index);
    if (logits == nullptr) {
        return 0;
    }

    llama_token best = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < n_vocab; ++token) {
        if (logits[token] > best_logit) {
            best_logit = logits[token];
            best = token;
        }
    }

    return best;
}

static llama_token skippy_greedy_sample_ith(skippy_session * session, int32_t index) {
    return skippy_greedy_sample_context(session->stage_model->model, session->ctx, index);
}

static llama_token skippy_greedy_sample(skippy_session * session) {
    return skippy_greedy_sample_ith(session, -1);
}

static bool skippy_compute_token_signal_context(
        const llama_model * model,
        llama_context * ctx,
        int32_t logits_index,
        skippy_token_signal * out_signal) {
    if (model == nullptr || ctx == nullptr || out_signal == nullptr) {
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, logits_index);
    if (logits == nullptr || n_vocab <= 0) {
        return false;
    }

    int32_t top_token = 0;
    int32_t second_token = 0;
    float top_logit = -std::numeric_limits<float>::infinity();
    float second_logit = -std::numeric_limits<float>::infinity();
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < n_vocab; ++token) {
        const float logit = logits[token];
        if (logit > max_logit) {
            max_logit = logit;
        }
        if (logit > top_logit) {
            second_logit = top_logit;
            second_token = top_token;
            top_logit = logit;
            top_token = token;
        } else if (logit > second_logit) {
            second_logit = logit;
            second_token = token;
        }
    }

    if (!std::isfinite(max_logit)) {
        return false;
    }

    double sum_exp = 0.0;
    double entropy = 0.0;
    for (int32_t token = 0; token < n_vocab; ++token) {
        sum_exp += std::exp(static_cast<double>(logits[token] - max_logit));
    }
    if (sum_exp <= 0.0 || !std::isfinite(sum_exp)) {
        return false;
    }
    const double log_sum_exp = static_cast<double>(max_logit) + std::log(sum_exp);
    for (int32_t token = 0; token < n_vocab; ++token) {
        const double logprob = static_cast<double>(logits[token]) - log_sum_exp;
        const double prob = std::exp(logprob);
        entropy -= prob * logprob;
    }

    const float top_logprob = static_cast<float>(static_cast<double>(top_logit) - log_sum_exp);
    const float second_logprob = static_cast<float>(static_cast<double>(second_logit) - log_sum_exp);
    *out_signal = {};
    out_signal->entropy = static_cast<float>(entropy);
    out_signal->top_logprob = top_logprob;
    out_signal->second_logprob = second_logprob;
    out_signal->margin = top_logprob - second_logprob;
    out_signal->top_token = top_token;
    out_signal->second_token = second_token;
    return true;
}

static bool skippy_compute_token_signal(
        skippy_session * session,
        int32_t logits_index,
        skippy_token_signal * out_signal) {
    if (session == nullptr || session->stage_model == nullptr) {
        return false;
    }
    return skippy_compute_token_signal_context(
            session->stage_model->model,
            session->ctx,
            logits_index,
            out_signal);
}

static void skippy_record_signal(skippy_session * session, int32_t logits_index) {
    skippy_token_signal signal = {};
    if (skippy_compute_token_signal(session, logits_index, &signal)) {
        session->signal_history.push_back(signal);
    }
}

static void skippy_clear_chat_sampling(skippy_session * session) {
    if (session == nullptr) {
        return;
    }
    llama_sampler_free(session->sampling_chain);
    llama_sampler_free(session->grammar_sampler);
    session->sampling_chain = nullptr;
    session->grammar_sampler = nullptr;
    session->sampling_config = {};
    session->sampling_config_valid = false;
    session->chat_sampling_metadata.clear();
    session->grammar_generated_start = 0;
    session->sampling_accepted_token_count = 0;
}

static std::string skippy_regex_escape(const std::string & value) {
    static const std::regex special_chars(R"([-[\]{}()*+?.,\^$|#\s])");
    return std::regex_replace(value, special_chars, R"(\$&)");
}

static std::vector<llama_token> skippy_tokenize_text(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special,
        bool parse_special) {
    if (vocab == nullptr || text.empty()) {
        return {};
    }
    const int32_t count = -llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, parse_special);
    if (count <= 0) {
        return {};
    }
    std::vector<llama_token> tokens(static_cast<size_t>(count));
    const int32_t written = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), count, add_special, parse_special);
    if (written < 0) {
        return {};
    }
    tokens.resize(static_cast<size_t>(written));
    return tokens;
}

static bool skippy_sampling_enabled(const skippy_sampling_config * sampling) {
    if (sampling == nullptr || sampling->version == 0 || sampling->flags == 0) {
        return false;
    }
    return true;
}

static bool skippy_sampling_is_greedy_equivalent(const skippy_sampling_config * sampling) {
    if (!skippy_sampling_enabled(sampling)) {
        return true;
    }
    const float repeat_penalty = sampling->repeat_penalty == 0.0f ? 1.0f : sampling->repeat_penalty;
    return sampling->temperature <= 0.0f &&
           sampling->presence_penalty == 0.0f &&
           sampling->frequency_penalty == 0.0f &&
           repeat_penalty == 1.0f &&
           sampling->logit_bias_count == 0;
}

static bool skippy_sampling_configs_equal(
        const skippy_sampling_config & lhs,
        const skippy_sampling_config & rhs) {
    if (lhs.version != rhs.version ||
            lhs.flags != rhs.flags ||
            lhs.seed != rhs.seed ||
            lhs.top_k != rhs.top_k ||
            lhs.penalty_last_n != rhs.penalty_last_n ||
            lhs.temperature != rhs.temperature ||
            lhs.top_p != rhs.top_p ||
            lhs.presence_penalty != rhs.presence_penalty ||
            lhs.frequency_penalty != rhs.frequency_penalty ||
            lhs.repeat_penalty != rhs.repeat_penalty ||
            lhs.logit_bias_count != rhs.logit_bias_count ||
            lhs.min_p != rhs.min_p) {
        return false;
    }

    const uint32_t logit_bias_count = std::min<uint32_t>(lhs.logit_bias_count, SKIPPY_MAX_LOGIT_BIAS);
    for (uint32_t i = 0; i < logit_bias_count; ++i) {
        if (lhs.logit_bias[i].token != rhs.logit_bias[i].token ||
                lhs.logit_bias[i].bias != rhs.logit_bias[i].bias) {
            return false;
        }
    }
    return true;
}

static llama_sampler * skippy_build_sampling_chain(
        skippy_session * session,
        const skippy_sampling_config * sampling) {
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(chain_params);
    if (sampler == nullptr) {
        return nullptr;
    }

    if (!skippy_sampling_enabled(sampling)) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        return sampler;
    }

    const int32_t penalty_last_n = sampling->penalty_last_n == 0 ? -1 : sampling->penalty_last_n;
    const float repeat_penalty = sampling->repeat_penalty == 0.0f ? 1.0f : sampling->repeat_penalty;
    if (repeat_penalty != 1.0f || sampling->frequency_penalty != 0.0f || sampling->presence_penalty != 0.0f) {
        llama_sampler_chain_add(
                sampler,
                llama_sampler_init_penalties(
                        penalty_last_n,
                        repeat_penalty,
                        sampling->frequency_penalty,
                        sampling->presence_penalty));
    }
    const uint32_t logit_bias_count = std::min<uint32_t>(sampling->logit_bias_count, SKIPPY_MAX_LOGIT_BIAS);
    if (logit_bias_count > 0) {
        const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
        llama_sampler_chain_add(
                sampler,
                llama_sampler_init_logit_bias(
                        llama_vocab_n_tokens(vocab),
                        static_cast<int32_t>(logit_bias_count),
                        sampling->logit_bias));
    }
    if (sampling->top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(sampling->top_k));
    }
    if (sampling->top_p > 0.0f && sampling->top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(sampling->top_p, 0));
    }
    if (sampling->min_p > 0.0f && sampling->min_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(sampling->min_p, 0));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_temp_ext(sampling->temperature, 0.0f, 1.0f));
    const uint32_t seed = sampling->seed == 0 ? LLAMA_DEFAULT_SEED : sampling->seed;
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    return sampler;
}

static bool skippy_ensure_plain_sampling_chain(
        skippy_session * session,
        const skippy_sampling_config * sampling) {
    if (session == nullptr || !skippy_sampling_enabled(sampling)) {
        return false;
    }

    if (session->sampling_chain != nullptr &&
            session->chat_sampling_metadata.empty() &&
            session->sampling_config_valid &&
            skippy_sampling_configs_equal(session->sampling_config, *sampling)) {
        return true;
    }

    skippy_clear_chat_sampling(session);
    session->sampling_chain = skippy_build_sampling_chain(session, sampling);
    if (session->sampling_chain == nullptr) {
        return false;
    }
    session->sampling_config = *sampling;
    session->sampling_config_valid = true;
    session->sampling_accepted_token_count = 0;
    return true;
}

static bool skippy_init_chat_grammar_sampler(
        skippy_session * session,
        const json & metadata,
        struct skippy_error ** out_error) {
    const std::string grammar = metadata.value("grammar", std::string());
    if (grammar.empty()) {
        return true;
    }

    const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
    const bool grammar_lazy = metadata.value("grammar_lazy", false);
    std::vector<std::string> trigger_patterns_storage;
    std::vector<const char *> trigger_patterns;
    std::vector<llama_token> trigger_tokens;
    if (const auto triggers = metadata.find("grammar_triggers"); triggers != metadata.end() && triggers->is_array()) {
        for (const auto & trigger : *triggers) {
            const int type = trigger.value("type", -1);
            switch (type) {
                case 0:
                    if (trigger.contains("token")) {
                        trigger_tokens.push_back(static_cast<llama_token>(trigger.value("token", 0)));
                    }
                    break;
                case 1:
                    trigger_patterns_storage.push_back(skippy_regex_escape(trigger.value("value", std::string())));
                    break;
                case 2:
                    trigger_patterns_storage.push_back(trigger.value("value", std::string()));
                    break;
                case 3:
                {
                    std::string pattern = trigger.value("value", std::string());
                    if (pattern.empty()) {
                        pattern = "^$";
                    } else {
                        if (pattern.front() != '^') {
                            pattern.insert(pattern.begin(), '^');
                        }
                        if (pattern.back() != '$') {
                            pattern.push_back('$');
                        }
                    }
                    trigger_patterns_storage.push_back(std::move(pattern));
                    break;
                }
                default:
                    break;
            }
        }
    }
    trigger_patterns.reserve(trigger_patterns_storage.size());
    for (const auto & pattern : trigger_patterns_storage) {
        trigger_patterns.push_back(pattern.c_str());
    }

    session->grammar_sampler = grammar_lazy ?
            llama_sampler_init_grammar_lazy_patterns(
                    vocab,
                    grammar.c_str(),
                    "root",
                    trigger_patterns.data(),
                    trigger_patterns.size(),
                    trigger_tokens.data(),
                    trigger_tokens.size()) :
            llama_sampler_init_grammar(vocab, grammar.c_str(), "root");
    if (session->grammar_sampler == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to initialize chat grammar sampler");
        return false;
    }

    if (!grammar_lazy) {
        const std::string generation_prompt = metadata.value("generation_prompt", std::string());
        try {
            for (const llama_token token : skippy_tokenize_text(vocab, generation_prompt, false, true)) {
                llama_sampler_accept(session->grammar_sampler, token);
            }
        } catch (const std::exception & e) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, e.what());
            return false;
        } catch (...) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to prefill chat grammar sampler");
            return false;
        }
    }
    return true;
}

static void skippy_sync_chat_sampling_history(skippy_session * session) {
    if (session == nullptr || session->sampling_chain == nullptr) {
        return;
    }
    while (session->sampling_accepted_token_count < session->token_history.size()) {
        const size_t index = session->sampling_accepted_token_count;
        const llama_token token = session->token_history[index];
        llama_sampler_accept(session->sampling_chain, token);
        if (session->grammar_sampler != nullptr && index >= session->grammar_generated_start) {
            try {
                llama_sampler_accept(session->grammar_sampler, token);
            } catch (const std::exception & e) {
                skippy_disable_chat_grammar_after_exception(session, "accept", e.what());
            } catch (...) {
                skippy_disable_chat_grammar_after_exception(session, "accept", nullptr);
            }
        }
        session->sampling_accepted_token_count++;
    }
}

static llama_token skippy_chat_sample_token(skippy_session * session, int32_t logits_index) {
    skippy_sync_chat_sampling_history(session);
    llama_synchronize(session->ctx);

    const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(session->ctx, logits_index);
    if (logits == nullptr) {
        return skippy_greedy_sample_ith(session, logits_index);
    }

    std::vector<llama_token_data> candidates;
    candidates.reserve(static_cast<size_t>(n_vocab));
    for (llama_token token = 0; token < n_vocab; ++token) {
        candidates.push_back({ token, logits[token], 0.0f });
    }
    llama_token_data_array cur = {
        candidates.data(),
        candidates.size(),
        -1,
        false,
    };
    if (session->grammar_sampler != nullptr) {
        try {
            llama_sampler_apply(session->grammar_sampler, &cur);
        } catch (const std::exception & e) {
            skippy_disable_chat_grammar_after_exception(session, "apply", e.what());
        } catch (...) {
            skippy_disable_chat_grammar_after_exception(session, "apply", nullptr);
        }
    }
    llama_sampler_apply(session->sampling_chain, &cur);
    if (cur.selected < 0 || static_cast<size_t>(cur.selected) >= cur.size) {
        return skippy_greedy_sample_ith(session, logits_index);
    }
    return cur.data[cur.selected].id;
}

static llama_token skippy_sample_token_ith(
        skippy_session * session,
        const skippy_sampling_config * sampling,
        int32_t logits_index) {
    if (skippy_mtp_greedy_sampling_fastpath_enabled() &&
            session != nullptr &&
            session->grammar_sampler == nullptr &&
            skippy_sampling_is_greedy_equivalent(sampling)) {
        if (session->sampling_chain != nullptr) {
            skippy_clear_chat_sampling(session);
        }
        return skippy_greedy_sample_ith(session, logits_index);
    }
    if (session != nullptr && session->sampling_chain != nullptr) {
        if (!session->chat_sampling_metadata.empty() ||
                (skippy_sampling_enabled(sampling) &&
                 session->sampling_config_valid &&
                 skippy_sampling_configs_equal(session->sampling_config, *sampling))) {
            return skippy_chat_sample_token(session, logits_index);
        }
        skippy_clear_chat_sampling(session);
    }
    if (!skippy_sampling_enabled(sampling)) {
        return skippy_greedy_sample_ith(session, logits_index);
    }
    if (skippy_ensure_plain_sampling_chain(session, sampling)) {
        return skippy_chat_sample_token(session, logits_index);
    }
    llama_synchronize(session->ctx);

    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(chain_params);
    if (sampler == nullptr) {
        return skippy_greedy_sample_ith(session, logits_index);
    }

    const int32_t penalty_last_n = sampling->penalty_last_n == 0 ? -1 : sampling->penalty_last_n;
    const float repeat_penalty = sampling->repeat_penalty == 0.0f ? 1.0f : sampling->repeat_penalty;
    if (repeat_penalty != 1.0f || sampling->frequency_penalty != 0.0f || sampling->presence_penalty != 0.0f) {
        llama_sampler_chain_add(
                sampler,
                llama_sampler_init_penalties(
                        penalty_last_n,
                        repeat_penalty,
                        sampling->frequency_penalty,
                        sampling->presence_penalty));
    }
    const uint32_t logit_bias_count = std::min<uint32_t>(sampling->logit_bias_count, SKIPPY_MAX_LOGIT_BIAS);
    if (logit_bias_count > 0) {
        const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
        llama_sampler_chain_add(
                sampler,
                llama_sampler_init_logit_bias(
                        llama_vocab_n_tokens(vocab),
                        static_cast<int32_t>(logit_bias_count),
                        sampling->logit_bias));
    }
    if (sampling->top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(sampling->top_k));
    }
    if (sampling->top_p > 0.0f && sampling->top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(sampling->top_p, 0));
    }
    if (sampling->min_p > 0.0f && sampling->min_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(sampling->min_p, 0));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_temp_ext(sampling->temperature, 0.0f, 1.0f));
    const uint32_t seed = sampling->seed == 0 ? LLAMA_DEFAULT_SEED : sampling->seed;
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));

    for (const llama_token token : session->token_history) {
        llama_sampler_accept(sampler, token);
    }
    llama_token token = llama_sampler_sample(sampler, session->ctx, logits_index);
    llama_sampler_free(sampler);
    return token;
}

static llama_token skippy_sample_token(
        skippy_session * session,
        const skippy_sampling_config * sampling) {
    return skippy_sample_token_ith(session, sampling, -1);
}

static enum skippy_status skippy_mtp_propose_next(
        skippy_session * session,
        llama_token predicted_token,
        skippy_native_mtp_draft * out_mtp_draft,
        struct skippy_error ** out_error) {
    if (out_mtp_draft != nullptr) {
        *out_mtp_draft = {
            1,
            false,
            -1,
            0,
        };
    }
    if (!skippy_mtp_available(session) || predicted_token < 0 || !session->mtp_has_pending_h) {
        return skippy_success(out_error);
    }

    llama_context * mtp_ctx = session->stage_model->mtp_ctx;
    const int32_t n_embd = llama_model_n_embd(session->stage_model->model);
    if (session->mtp_pending_h.size() != static_cast<size_t>(n_embd)) {
        return skippy_success(out_error);
    }

    llama_token token = predicted_token;
    llama_pos pos = session->n_past;
    int32_t n_seq_id = 1;
    llama_seq_id seq_id = session->seq_id;
    llama_seq_id * seq_ids = &seq_id;
    int8_t logits = 1;
    llama_batch batch = {
        /*n_tokens =*/ 1,
        /*token    =*/ &token,
        /*embd     =*/ session->mtp_pending_h.data(),
        /*pos      =*/ &pos,
        /*n_seq_id =*/ &n_seq_id,
        /*seq_id   =*/ &seq_ids,
        /*logits   =*/ &logits,
    };

    const int64_t t_start_us = ggml_time_us();
    skippy_graph_filter_scope graph_filter_scope(&session->stage_model->config);
    const int32_t rc = llama_decode(mtp_ctx, batch);
    const int64_t elapsed_us = ggml_time_us() - t_start_us;
    if (rc != 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama_decode failed for MTP sidecar proposal");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    if (out_mtp_draft != nullptr) {
        out_mtp_draft->version = 1;
        out_mtp_draft->available = true;
        out_mtp_draft->token_id = skippy_greedy_sample_context(session->stage_model->model, mtp_ctx, -1);
        out_mtp_draft->proposal_compute_us = elapsed_us;
        session->mtp_has_pending_draft = true;
        session->mtp_pending_draft_pos = session->n_past;
        session->mtp_pending_draft_token = out_mtp_draft->token_id;
    }
    return skippy_success(out_error);
}

static enum skippy_status skippy_prepare_empty_activation_frame(
        skippy_session * session,
        size_t token_count,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        skippy_activation_desc * output_desc,
        struct skippy_error ** out_error) {
    if (output_payload_capacity > 0 && output_payload == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "output_payload is required when output capacity is non-zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    if (out_output_payload_bytes != nullptr) {
        *out_output_payload_bytes = 0;
    }

    if (output_desc != nullptr) {
        *output_desc = {};
        output_desc->version = 1;
        output_desc->dtype = SKIPPY_ACTIVATION_DTYPE_UNKNOWN;
        output_desc->layout = SKIPPY_ACTIVATION_LAYOUT_OPAQUE;
        output_desc->producer_stage_index = session != nullptr ? session->stage_model->config.stage_index : -1;
        output_desc->layer_start = session != nullptr ? session->stage_model->config.layer_start : 0;
        output_desc->layer_end = session != nullptr ? session->stage_model->config.layer_end : 0;
        output_desc->token_count = static_cast<uint32_t>(std::min<size_t>(token_count, std::numeric_limits<uint32_t>::max()));
        output_desc->sequence_count = token_count > 0 ? 1 : 0;
        output_desc->payload_bytes = 0;
        output_desc->flags = 0;
    }

    return SKIPPY_STATUS_OK;
}

static enum skippy_status skippy_copy_output_activation_frame(
        skippy_session * session,
        size_t token_count,
        void * output_payload,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_error ** out_error) {
    if (!skippy_emits_activation_frame(session)) {
        return skippy_success(out_error);
    }

    const uint64_t output_flags = skippy_output_activation_flags(session, input_desc);
    const size_t hidden_bytes = skippy_activation_hidden_bytes(session, token_count);
    if ((output_flags & SKIPPY_ACTIVATION_FLAG_GEMMA3N_ALTUP) != 0) {
        llm_graph_result * res = session->ctx->get_gf_res_prev();
        ggml_tensor * altup = res != nullptr ? res->get_skippy_gemma3n_altup() : nullptr;
        const size_t altup_bytes = skippy_gemma3n_altup_bytes(session, token_count);
        if (altup == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "Gemma3n AltUp activation output was not available");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        if (ggml_nbytes(altup) < altup_bytes) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "Gemma3n AltUp activation tensor is smaller than expected payload");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        ggml_backend_tensor_get(altup, output_payload, 0, altup_bytes);
        return skippy_success(out_error);
    }

    float * embeddings = llama_get_embeddings(session->ctx);
    if (embeddings == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama embeddings output was not available");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    std::memcpy(output_payload, embeddings, hidden_bytes);

    if ((output_flags & SKIPPY_ACTIVATION_FLAG_RWKV7_V_FIRST) != 0) {
        uint8_t * sideband_output = static_cast<uint8_t *>(output_payload) + hidden_bytes;
        const skippy_runtime_config & config = session->stage_model->config;
        if (config.layer_start == 0) {
            llm_graph_result * res = session->ctx->get_gf_res_prev();
            ggml_tensor * v_first = res != nullptr ? res->get_skippy_rwkv7_v_first() : nullptr;
            if (v_first == nullptr) {
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "RWKV7 v_first sideband output was not available");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
            if (ggml_nbytes(v_first) < hidden_bytes) {
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "RWKV7 v_first sideband tensor is smaller than activation hidden payload");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
            ggml_backend_tensor_get(v_first, sideband_output, 0, hidden_bytes);
        } else {
            if (input_desc == nullptr ||
                input_payload == nullptr ||
                (input_desc->flags & SKIPPY_ACTIVATION_FLAG_RWKV7_V_FIRST) == 0 ||
                input_desc->payload_bytes < hidden_bytes * 2) {
                skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "RWKV7 downstream slice cannot forward missing v_first sideband");
                return SKIPPY_STATUS_INVALID_ARGUMENT;
            }
            std::memcpy(sideband_output, static_cast<const uint8_t *>(input_payload) + hidden_bytes, hidden_bytes);
        }
    }
    return skippy_success(out_error);
}

static enum skippy_status skippy_decode_activation_frame(
        skippy_session * session,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        const llama_token * token_ids,
        const llama_pos * positions,
        size_t position_count,
        size_t token_count,
        bool request_logits,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || input_desc == nullptr || input_payload == nullptr || token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and activation frame input are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int32_t n_tokens = static_cast<int32_t>(token_count);
    const int32_t n_embd = llama_model_n_embd(session->stage_model->model);
    const int32_t n_embd_inp = llama_model_n_embd_inp(session->stage_model->model);
    const int32_t n_altup = static_cast<int32_t>(session->stage_model->model->hparams.n_altup);
    const int32_t n_pos_per_embd = session->stage_model->model->hparams.n_pos_per_embd();
    const size_t expected_position_count = static_cast<size_t>(n_tokens)*n_pos_per_embd;

    std::vector<float> embd_storage(static_cast<size_t>(n_tokens)*n_embd_inp);
    std::vector<llama_pos> pos_storage(expected_position_count);
    std::vector<int32_t> n_seq_id_storage(n_tokens);
    std::vector<llama_seq_id> seq_id_0(1, session->seq_id);
    std::vector<llama_seq_id *> seq_id_storage(n_tokens + 1, nullptr);
    std::vector<int8_t> logits_storage(n_tokens);

    const size_t hidden_bytes = skippy_activation_hidden_bytes(session, token_count);
    if ((input_desc->flags & SKIPPY_ACTIVATION_FLAG_GEMMA3N_ALTUP) != 0) {
        const float * input = static_cast<const float *>(input_payload);
        for (int32_t i = 0; i < n_tokens; ++i) {
            float * dst = embd_storage.data() + static_cast<size_t>(i)*n_embd_inp;
            std::memcpy(dst, input + static_cast<size_t>(i)*n_embd, static_cast<size_t>(n_embd)*sizeof(float));
            if (n_embd_inp > n_embd) {
                std::memset(dst + n_embd, 0, static_cast<size_t>(n_embd_inp - n_embd)*sizeof(float));
            }
        }
    } else if (n_embd_inp == n_embd) {
        std::memcpy(embd_storage.data(), input_payload, hidden_bytes);
    } else {
        const float * input = static_cast<const float *>(input_payload);
        for (int32_t i = 0; i < n_tokens; ++i) {
            float * dst = embd_storage.data() + static_cast<size_t>(i)*n_embd_inp;
            std::memcpy(dst, input + static_cast<size_t>(i)*n_embd, static_cast<size_t>(n_embd)*sizeof(float));
            std::memset(dst + n_embd, 0, static_cast<size_t>(n_embd_inp - n_embd)*sizeof(float));
        }
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        n_seq_id_storage[i] = 1;
        seq_id_storage[i] = seq_id_0.data();
        logits_storage[i] = request_logits && i == n_tokens - 1 ? 1 : 0;
    }
    if (positions != nullptr && position_count == expected_position_count) {
        std::memcpy(pos_storage.data(), positions, expected_position_count*sizeof(llama_pos));
    } else if (n_pos_per_embd == 4) {
        for (int32_t i = 0; i < n_tokens; ++i) {
            const llama_pos position = session->n_past + i;
            pos_storage[               i] = position;
            pos_storage[    n_tokens + i] = position;
            pos_storage[2 * n_tokens + i] = position;
            pos_storage[3 * n_tokens + i] = 0;
        }
    } else {
        for (int32_t i = 0; i < n_tokens; ++i) {
            pos_storage[i] = session->n_past + i;
        }
    }

    llama_batch batch = {
        /*n_tokens =*/ n_tokens,
        /*token    =*/ nullptr,
        /*embd     =*/ embd_storage.data(),
        /*pos      =*/ pos_storage.data(),
        /*n_seq_id =*/ n_seq_id_storage.data(),
        /*seq_id   =*/ seq_id_storage.data(),
        /*logits   =*/ logits_storage.data(),
    };

    skippy_activation_tokens_scope activation_tokens_scope(token_ids, token_count);
    skippy_rwkv7_v_first_scope rwkv7_v_first_scope(input_desc, input_payload, hidden_bytes, n_embd);
    skippy_gemma3n_altup_scope gemma3n_altup_scope(input_desc, input_payload, n_embd, n_altup);
    const llama_pos token_start = pos_storage.empty() ? session->n_past : pos_storage[0];
    enum skippy_status status = skippy_decode_batch(session, batch, token_count, out_error);
    if (status == SKIPPY_STATUS_OK && token_ids != nullptr) {
        status = skippy_mtp_sync_target_tokens(session, token_ids, token_count, token_start, out_error);
    }
    return status;
}

static enum skippy_status skippy_verify_activation_frame(
        skippy_session * session,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || input_desc == nullptr || input_payload == nullptr || token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and activation frame input are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int32_t n_tokens = static_cast<int32_t>(token_count);
    const int32_t n_embd = llama_model_n_embd(session->stage_model->model);
    const int32_t n_altup = static_cast<int32_t>(session->stage_model->model->hparams.n_altup);
    const size_t hidden_bytes = skippy_activation_hidden_bytes(session, token_count);
    const bool alias_input_payload = input_desc->flags == 0;

    llama_batch batch = {};
    std::vector<llama_pos> pos_storage;
    std::vector<int32_t> n_seq_id_storage;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id *> seq_id_storage;
    std::vector<int8_t> logits_storage;
    if (alias_input_payload) {
        pos_storage.resize(n_tokens);
        n_seq_id_storage.resize(n_tokens);
        seq_id_0.assign(1, session->seq_id);
        seq_id_storage.resize(n_tokens, seq_id_0.data());
        logits_storage.resize(n_tokens, 1);
        batch = {
            /*n_tokens =*/ n_tokens,
            /*token    =*/ nullptr,
            /*embd     =*/ const_cast<float *>(static_cast<const float *>(input_payload)),
            /*pos      =*/ pos_storage.data(),
            /*n_seq_id =*/ n_seq_id_storage.data(),
            /*seq_id   =*/ seq_id_storage.data(),
            /*logits   =*/ logits_storage.data(),
        };
    } else {
        batch = llama_batch_init(n_tokens, n_embd, 1);
        std::memcpy(batch.embd, input_payload, hidden_bytes);
    }
    batch.n_tokens = n_tokens;

    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i] = session->n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = session->seq_id;
        batch.logits[i] = 1;
    }

    skippy_rwkv7_v_first_scope rwkv7_v_first_scope(input_desc, input_payload, hidden_bytes, n_embd);
    skippy_gemma3n_altup_scope gemma3n_altup_scope(input_desc, input_payload, n_embd, n_altup);
    const llama_pos token_start = session->n_past;
    enum skippy_status status = skippy_decode_batch(session, batch, token_count, out_error);
    if (!alias_input_payload) {
        llama_batch_free(batch);
    }
    if (status == SKIPPY_STATUS_OK && token_ids != nullptr) {
        status = skippy_mtp_sync_target_tokens(session, token_ids, token_count, token_start, out_error);
    }
    return status;
}

extern "C" {

struct skippy_abi_version skippy_abi_version(void) {
    return {
        SKIPPY_ABI_VERSION_MAJOR,
        SKIPPY_ABI_VERSION_MINOR,
        SKIPPY_ABI_VERSION_PATCH,
    };
}

uint64_t skippy_abi_features(void) {
    return SKIPPY_FEATURE_RUNTIME_SLICE |
           SKIPPY_FEATURE_MODEL_INTROSPECTION |
           SKIPPY_FEATURE_GGUF_SLICE_WRITE |
           SKIPPY_FEATURE_STATE_IMPORT_EXPORT |
           SKIPPY_FEATURE_TOKENIZE_DETOKENIZE |
           SKIPPY_FEATURE_ACTIVATION_FRAME |
           SKIPPY_FEATURE_NATIVE_KV_PAGE |
           SKIPPY_FEATURE_SESSION_RESET |
           SKIPPY_FEATURE_BATCH_VERIFY |
           SKIPPY_FEATURE_CHAT_TEMPLATE |
           SKIPPY_FEATURE_SAMPLING_CONFIG |
           SKIPPY_FEATURE_BATCH_VERIFY_FRAME |
           SKIPPY_FEATURE_RECURRENT_STATE |
           SKIPPY_FEATURE_LOGIT_BIAS |
           SKIPPY_FEATURE_SESSION_TRIM |
           SKIPPY_FEATURE_SESSION_CHECKPOINT |
           SKIPPY_FEATURE_PACKAGE_PART_LOAD |
           SKIPPY_FEATURE_GENERATION_SIGNALS |
           SKIPPY_FEATURE_EXTERNAL_MEDIA_PREFILL |
           SKIPPY_FEATURE_RUNTIME_EVENTS |
           SKIPPY_FEATURE_BACKEND_DEVICES |
           SKIPPY_FEATURE_NATIVE_MTP_N1;
}

void skippy_error_free(struct skippy_error * error) {
    if (error == nullptr) {
        return;
    }

    std::free(const_cast<char *>(error->message));
    delete error;
}

static void skippy_silent_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static void skippy_load_backends_for_device_query() {
    static std::once_flag once;
    std::call_once(once, []() {
        ggml_log_callback previous_callback = nullptr;
        void * previous_user_data = nullptr;
        llama_log_get(&previous_callback, &previous_user_data);

        llama_log_set(skippy_silent_log_callback, nullptr);
        llama_backend_init();
        if (!ggml_backend_reg_count()) {
            ggml_backend_load_all();
        }
        llama_log_set(previous_callback, previous_user_data);
    });
}

enum skippy_status skippy_backend_device_count(
        size_t * out_count,
        struct skippy_error ** out_error) {
    if (out_count == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "out_count is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_load_backends_for_device_query();
    *out_count = ggml_backend_dev_count();
    return skippy_success(out_error);
}

enum skippy_status skippy_backend_device_at(
        size_t index,
        struct skippy_backend_device * out_device,
        struct skippy_error ** out_error) {
    if (out_device == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "out_device is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_load_backends_for_device_query();
    const size_t device_count = ggml_backend_dev_count();
    if (index >= device_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "backend device index is out of range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    ggml_backend_dev_t device = ggml_backend_dev_get(index);
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(device, &props);

    out_device->version = 1;
    out_device->name = props.name;
    out_device->description = props.description;
    out_device->device_id = props.device_id;
    out_device->memory_free = static_cast<uint64_t>(props.memory_free);
    out_device->memory_total = static_cast<uint64_t>(props.memory_total);
    out_device->type = skippy_backend_device_type_from_ggml(props.type);
    out_device->caps = skippy_backend_device_caps_from_ggml(props.caps);
    return skippy_success(out_error);
}

static enum skippy_status skippy_finish_model_open(
        llama_model * model,
        const struct skippy_runtime_config * config,
        skippy_runtime_event_scope * event_scope,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    if (config != nullptr && config->filter_tensors_on_load) {
        const int32_t n_layer = skippy_stage_layer_count(model);
        if (model->arch != LLM_ARCH_LLAMA &&
            model->arch != LLM_ARCH_AFMOE &&
            model->arch != LLM_ARCH_APERTUS &&
            model->arch != LLM_ARCH_ARCEE &&
            model->arch != LLM_ARCH_ARCTIC &&
            model->arch != LLM_ARCH_BAICHUAN &&
            model->arch != LLM_ARCH_BAILINGMOE &&
            model->arch != LLM_ARCH_BAILINGMOE2 &&
            model->arch != LLM_ARCH_BITNET &&
            model->arch != LLM_ARCH_BLOOM &&
            model->arch != LLM_ARCH_CHATGLM &&
            model->arch != LLM_ARCH_COHERE2 &&
            model->arch != LLM_ARCH_COMMAND_R &&
            model->arch != LLM_ARCH_CODESHELL &&
            model->arch != LLM_ARCH_DBRX &&
            model->arch != LLM_ARCH_DECI &&
            model->arch != LLM_ARCH_DOTS1 &&
            model->arch != LLM_ARCH_DREAM &&
            model->arch != LLM_ARCH_ERNIE4_5 &&
            model->arch != LLM_ARCH_ERNIE4_5_MOE &&
            model->arch != LLM_ARCH_EXAONE &&
            model->arch != LLM_ARCH_EXAONE4 &&
            model->arch != LLM_ARCH_EXAONE_MOE &&
            model->arch != LLM_ARCH_FALCON &&
            model->arch != LLM_ARCH_GPT2 &&
            model->arch != LLM_ARCH_GPTNEOX &&
            model->arch != LLM_ARCH_GRANITE &&
            model->arch != LLM_ARCH_GRANITE_HYBRID &&
            model->arch != LLM_ARCH_GRANITE_MOE &&
            model->arch != LLM_ARCH_GROK &&
            model->arch != LLM_ARCH_GROVEMOE &&
            model->arch != LLM_ARCH_HUNYUAN_DENSE &&
            model->arch != LLM_ARCH_HUNYUAN_MOE &&
            model->arch != LLM_ARCH_HUNYUAN_VL &&
            model->arch != LLM_ARCH_INTERNLM2 &&
            model->arch != LLM_ARCH_JAIS &&
            model->arch != LLM_ARCH_JAIS2 &&
            model->arch != LLM_ARCH_JAMBA &&
            model->arch != LLM_ARCH_LFM2 &&
            model->arch != LLM_ARCH_LLADA &&
            model->arch != LLM_ARCH_LLADA_MOE &&
            model->arch != LLM_ARCH_LLAMA4 &&
            model->arch != LLM_ARCH_MAINCODER &&
            model->arch != LLM_ARCH_MAMBA &&
            model->arch != LLM_ARCH_MAMBA2 &&
            model->arch != LLM_ARCH_MIMO2 &&
            model->arch != LLM_ARCH_MINICPM3 &&
            model->arch != LLM_ARCH_MISTRAL3 &&
            model->arch != LLM_ARCH_MISTRAL4 &&
            model->arch != LLM_ARCH_MPT &&
            model->arch != LLM_ARCH_NEMOTRON &&
            model->arch != LLM_ARCH_OLMO2 &&
            model->arch != LLM_ARCH_OLMOE &&
            model->arch != LLM_ARCH_OPENAI_MOE &&
            model->arch != LLM_ARCH_OPENELM &&
            model->arch != LLM_ARCH_ORION &&
            model->arch != LLM_ARCH_PLAMO &&
            model->arch != LLM_ARCH_PLAMO3 &&
            model->arch != LLM_ARCH_PLM &&
            model->arch != LLM_ARCH_QWEN &&
            model->arch != LLM_ARCH_QWEN2 &&
            model->arch != LLM_ARCH_QWEN3 &&
            model->arch != LLM_ARCH_QWEN3NEXT &&
            model->arch != LLM_ARCH_QWEN2VL &&
            model->arch != LLM_ARCH_QWEN3VL &&
            model->arch != LLM_ARCH_QWEN3VLMOE &&
            model->arch != LLM_ARCH_QWEN2MOE &&
            model->arch != LLM_ARCH_QWEN35 &&
            model->arch != LLM_ARCH_QWEN35MOE &&
            model->arch != LLM_ARCH_QWEN3MOE &&
            model->arch != LLM_ARCH_REFACT &&
            model->arch != LLM_ARCH_RND1 &&
            model->arch != LLM_ARCH_RWKV6 &&
            model->arch != LLM_ARCH_RWKV7 &&
            model->arch != LLM_ARCH_ARWKV7 &&
            model->arch != LLM_ARCH_SEED_OSS &&
            model->arch != LLM_ARCH_SMALLTHINKER &&
            model->arch != LLM_ARCH_SMOLLM3 &&
            model->arch != LLM_ARCH_GEMMA &&
            model->arch != LLM_ARCH_GEMMA2 &&
            model->arch != LLM_ARCH_GEMMA3 &&
            model->arch != LLM_ARCH_GEMMA3N &&
            model->arch != LLM_ARCH_GEMMA4 &&
            model->arch != LLM_ARCH_GLM_DSA &&
            model->arch != LLM_ARCH_GLM4 &&
            model->arch != LLM_ARCH_DEEPSEEK2 &&
            model->arch != LLM_ARCH_DEEPSEEK2OCR &&
            model->arch != LLM_ARCH_FALCON_H1 &&
            model->arch != LLM_ARCH_KIMI_LINEAR &&
            model->arch != LLM_ARCH_LFM2MOE &&
            model->arch != LLM_ARCH_MINICPM &&
            model->arch != LLM_ARCH_MINIMAX_M2 &&
            model->arch != LLM_ARCH_OLMO &&
            model->arch != LLM_ARCH_NEMOTRON_H &&
            model->arch != LLM_ARCH_NEMOTRON_H_MOE &&
            model->arch != LLM_ARCH_PHI2 &&
            model->arch != LLM_ARCH_PHI3 &&
            model->arch != LLM_ARCH_PHIMOE &&
            model->arch != LLM_ARCH_PLAMO2 &&
            model->arch != LLM_ARCH_RWKV6QWEN2 &&
            model->arch != LLM_ARCH_STARCODER &&
            model->arch != LLM_ARCH_STEP35 &&
            model->arch != LLM_ARCH_STARCODER2 &&
            model->arch != LLM_ARCH_STABLELM &&
            model->arch != LLM_ARCH_XVERSE) {
            llama_model_free(model);
            const char * message = "runtime-slice execution is not supported for this model architecture yet";
            if (event_scope != nullptr) {
                event_scope->emit_failure(SKIPPY_STATUS_UNSUPPORTED, message);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, message);
            return SKIPPY_STATUS_UNSUPPORTED;
        }
        if (config->layer_end > n_layer) {
            llama_model_free(model);
            const char * message = "layer_end exceeds model layer count";
            if (event_scope != nullptr) {
                event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (config->include_embeddings && config->layer_start != 0 && !config->include_output) {
            llama_model_free(model);
            const char * message = "only the first runtime slice may include token embeddings";
            if (event_scope != nullptr) {
                event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (config->layer_start == 0 && !config->include_embeddings) {
            llama_model_free(model);
            const char * message = "the first runtime slice must include token embeddings";
            if (event_scope != nullptr) {
                event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (config->include_output && config->layer_end != n_layer) {
            llama_model_free(model);
            const char * message = "only the final runtime slice may include output tensors";
            if (event_scope != nullptr) {
                event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
    }

    skippy_model * stage_model = new skippy_model{};
    stage_model->model = model;
    if (config != nullptr) {
        stage_model->config = *config;
        stage_model->executable = true;
    }
    stage_model->lane_count = config != nullptr && config->lane_count > 0
        ? static_cast<uint32_t>(config->lane_count)
        : 1;

    llama_context_params params = llama_context_default_params();
    params.n_ctx = config != nullptr && config->ctx_size > 0 ? static_cast<uint32_t>(config->ctx_size) : 512;
    params.n_batch = config != nullptr && config->n_batch > 0 ? static_cast<uint32_t>(config->n_batch) : params.n_ctx;
    params.n_ubatch = config != nullptr && config->n_ubatch > 0 ? static_cast<uint32_t>(config->n_ubatch) : 0u;
    params.n_threads = config != nullptr && config->n_threads > 0 ? config->n_threads : skippy_default_thread_count();
    params.n_threads_batch = config != nullptr && config->n_threads_batch > 0 ? config->n_threads_batch : params.n_threads;
    params.n_seq_max = stage_model->lane_count;
    params.kv_unified = stage_model->lane_count > 1;
    params.type_k = config != nullptr && config->cache_type_k > 0 ? static_cast<ggml_type>(config->cache_type_k) : GGML_TYPE_F16;
    params.type_v = config != nullptr && config->cache_type_v > 0 ? static_cast<ggml_type>(config->cache_type_v) : GGML_TYPE_F16;
    params.flash_attn_type = config != nullptr ? static_cast<llama_flash_attn_type>(config->flash_attn_type) : LLAMA_FLASH_ATTN_TYPE_AUTO;
    params.embeddings = config != nullptr && config->filter_tensors_on_load && !config->include_output;
    if (llm_arch_is_recurrent(model->arch) || llm_arch_is_hybrid(model->arch)) {
        params.n_seq_max = std::max<uint32_t>(2, stage_model->lane_count * 2);
        params.n_rs_seq = std::max<uint32_t>(params.n_rs_seq, 2);
        params.kv_unified = true;
    }

    skippy_graph_filter_scope graph_filter_scope(config);
    stage_model->ctx = llama_init_from_model(model, params);
    if (stage_model->ctx == nullptr) {
        const char * message = "failed to create llama context";
        if (event_scope != nullptr) {
            event_scope->emit_failure(SKIPPY_STATUS_RUNTIME_ERROR, message);
        }
        llama_model_free(model);
        delete stage_model;
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, message);
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    if (config != nullptr &&
        config->include_output &&
        model->hparams.n_layer_nextn > 0) {
        llama_context_params mtp_params = params;
        mtp_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        mtp_params.embeddings = false;
        {
            skippy_graph_filter_scope graph_filter_scope(config);
            stage_model->mtp_ctx = llama_init_from_model(model, mtp_params);
        }
        if (stage_model->mtp_ctx != nullptr) {
            llama_set_embeddings_nextn(stage_model->ctx, true, false);
            llama_set_embeddings_nextn(stage_model->mtp_ctx, true, true);
        } else {
            fprintf(stderr, "skippy: native MTP sidecar unavailable for this final stage; continuing without drafts\n");
        }
    }

    stage_model->lane_in_use.assign(stage_model->lane_count, false);
    stage_model->lane_resident_prefix_tokens.resize(stage_model->lane_count);

    if (event_scope != nullptr) {
        event_scope->model_id = reinterpret_cast<uint64_t>(stage_model);
        skippy_emit_observable_backend_device(event_scope, config, model);
        event_scope->emit_finished();
    }

    *out_model = stage_model;
    return skippy_success(out_error);
}

static enum skippy_status skippy_model_open_impl(
        const char * path,
        const struct skippy_runtime_config * config,
        const struct skippy_runtime_event_reporter_v1 * reporter,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    enum skippy_status reporter_status = skippy_validate_runtime_event_reporter(reporter, out_error);
    if (reporter_status != SKIPPY_STATUS_OK) {
        return reporter_status;
    }

    skippy_runtime_event_scope event_scope(config, reporter);
    event_scope.emit_started(path);

    if (path == nullptr || out_model == nullptr) {
        const char * message = "path and out_model are required";
        event_scope.emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    *out_model = nullptr;

    if (config != nullptr && config->filter_tensors_on_load && (config->layer_start < 0 || config->layer_start >= config->layer_end)) {
        const char * message = "layer_start must be non-negative and less than layer_end";
        event_scope.emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    if (!skippy_is_full_model_config(config) && (config == nullptr || !config->filter_tensors_on_load)) {
        const char * message = "runtime tensor filtering is not implemented yet; use a full-model single-stage config";
        event_scope.emit_failure(SKIPPY_STATUS_UNSUPPORTED, message);
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, message);
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    llama_model_params params = llama_model_default_params();
    if (config != nullptr) {
        params.n_gpu_layers = config->n_gpu_layers;
        if (config->disable_repack || config->filter_tensors_on_load) {
            params.use_extra_bufts = false;
        }
    }
    if (event_scope.is_enabled()) {
        params.progress_callback = skippy_model_open_progress_callback;
        params.progress_callback_user_data = &event_scope;
    }

    llama_backend_init();
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    std::vector<ggml_backend_dev_t> selected_devices;
    enum skippy_status device_status = skippy_apply_selected_backend_device(config, params, selected_devices, out_error);
    if (device_status != SKIPPY_STATUS_OK) {
        if (out_error != nullptr && *out_error != nullptr) {
            event_scope.emit_failure(device_status, (*out_error)->message);
        } else {
            event_scope.emit_failure(device_status, "failed to apply selected backend device");
        }
        return device_status;
    }
    skippy_filter_scope filter_scope(config);
    llama_model * model = llama_model_load_from_file(path, params);
    if (model == nullptr) {
        const char * message = "failed to load llama model";
        event_scope.emit_failure(SKIPPY_STATUS_MODEL_ERROR, message);
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, message);
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    return skippy_finish_model_open(model, config, &event_scope, out_model, out_error);
}

static enum skippy_status skippy_model_open_from_parts_impl(
        const char * const * paths,
        size_t path_count,
        const struct skippy_runtime_config * config,
        const struct skippy_runtime_event_reporter_v1 * reporter,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    enum skippy_status reporter_status = skippy_validate_runtime_event_reporter(reporter, out_error);
    if (reporter_status != SKIPPY_STATUS_OK) {
        return reporter_status;
    }

    skippy_runtime_event_scope event_scope(config, reporter);
    const char * primary_path = paths != nullptr && path_count > 0 ? paths[0] : nullptr;
    event_scope.emit_started(primary_path);

    if (paths == nullptr || path_count == 0 || out_model == nullptr) {
        const char * message = "paths, path_count, and out_model are required";
        event_scope.emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < path_count; ++i) {
        if (paths[i] == nullptr) {
            const char * message = "package part path is null";
            event_scope.emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
    }

    *out_model = nullptr;

    if (config != nullptr && config->filter_tensors_on_load && (config->layer_start < 0 || config->layer_start >= config->layer_end)) {
        const char * message = "layer_start must be non-negative and less than layer_end";
        event_scope.emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    if (!skippy_is_full_model_config(config) && (config == nullptr || !config->filter_tensors_on_load)) {
        const char * message = "runtime tensor filtering is not implemented yet; use a full-model single-stage config";
        event_scope.emit_failure(SKIPPY_STATUS_UNSUPPORTED, message);
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, message);
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    llama_model_params params = llama_model_default_params();
    if (config != nullptr) {
        params.n_gpu_layers = config->n_gpu_layers;
        if (config->disable_repack || config->filter_tensors_on_load) {
            params.use_extra_bufts = false;
        }
    }
    if (event_scope.is_enabled()) {
        params.progress_callback = skippy_model_open_progress_callback;
        params.progress_callback_user_data = &event_scope;
    }

    llama_backend_init();
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    std::vector<ggml_backend_dev_t> selected_devices;
    enum skippy_status device_status = skippy_apply_selected_backend_device(config, params, selected_devices, out_error);
    if (device_status != SKIPPY_STATUS_OK) {
        if (out_error != nullptr && *out_error != nullptr) {
            event_scope.emit_failure(device_status, (*out_error)->message);
        } else {
            event_scope.emit_failure(device_status, "failed to apply selected backend device");
        }
        return device_status;
    }
    skippy_filter_scope filter_scope(config);
    llama_model * model = llama_model_load_from_parts(paths, path_count, params);
    if (model == nullptr) {
        const char * message = "failed to load llama model from GGUF parts";
        event_scope.emit_failure(SKIPPY_STATUS_MODEL_ERROR, message);
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, message);
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    return skippy_finish_model_open(model, config, &event_scope, out_model, out_error);
}

enum skippy_status skippy_model_open(
        const char * path,
        const struct skippy_runtime_config * config,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    return skippy_model_open_impl(path, config, nullptr, out_model, out_error);
}

enum skippy_status skippy_model_open_with_events(
        const char * path,
        const struct skippy_runtime_config * config,
        const struct skippy_runtime_event_reporter_v1 * reporter,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    return skippy_model_open_impl(path, config, reporter, out_model, out_error);
}

enum skippy_status skippy_model_open_from_parts(
        const char * const * paths,
        size_t path_count,
        const struct skippy_runtime_config * config,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    return skippy_model_open_from_parts_impl(paths, path_count, config, nullptr, out_model, out_error);
}

enum skippy_status skippy_model_open_from_parts_with_events(
        const char * const * paths,
        size_t path_count,
        const struct skippy_runtime_config * config,
        const struct skippy_runtime_event_reporter_v1 * reporter,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    return skippy_model_open_from_parts_impl(paths, path_count, config, reporter, out_model, out_error);
}

enum skippy_status skippy_model_free(
        struct skippy_model * model,
        struct skippy_error ** out_error) {
    if (model != nullptr) {
        if (model->mtp_ctx != nullptr) {
            llama_free(model->mtp_ctx);
        }
        if (model->ctx != nullptr) {
            llama_free(model->ctx);
        }
        llama_model_free(model->model);
        delete model;
    }
    return skippy_success(out_error);
}

const struct llama_model * skippy_model_llama_model(
        const struct skippy_model * model) {
    return model != nullptr ? model->model : nullptr;
}

enum skippy_status skippy_session_create(
        struct skippy_model * model,
        struct skippy_session ** out_session,
        struct skippy_error ** out_error) {
    if (model == nullptr || model->model == nullptr || model->ctx == nullptr || out_session == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model and out_session are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    *out_session = nullptr;

    int32_t seq_id = -1;
    for (uint32_t i = 0; i < model->lane_in_use.size(); ++i) {
        if (!model->lane_in_use[i]) {
            model->lane_in_use[i] = true;
            seq_id = static_cast<int32_t>(i);
            break;
        }
    }
    if (seq_id < 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "no skippy execution lane is available");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    skippy_session * session = new skippy_session{};
    session->stage_model = model;
    session->ctx = model->ctx;
    session->seq_id = seq_id;
    session->checkpoint_seq_id = seq_id + static_cast<int32_t>(model->lane_count);
    session->borrowed_sequence = false;
    session->borrowed_prefix_tokens = 0;
    session->n_past = 0;
    session->checkpoint_valid = false;
    session->checkpoint_n_past = 0;
    session->mtp_next_pos = 0;
    if (skippy_mtp_available(session)) {
        session->mtp_pending_h.assign(static_cast<size_t>(llama_model_n_embd(model->model)), 0.0f);
        skippy_mtp_clear_session_state(session);
    }
    *out_session = session;
    if (static_cast<size_t>(seq_id) < model->lane_resident_prefix_tokens.size()) {
        model->lane_resident_prefix_tokens[static_cast<size_t>(seq_id)].clear();
    }
    if (llama_memory_t memory = model->ctx->get_memory()) {
        llama_memory_seq_rm(memory, seq_id, -1, -1);
    }
    return skippy_success(out_error);
}

struct llama_context * skippy_session_llama_context(
        struct skippy_session * session) {
    return session != nullptr ? session->ctx : nullptr;
}

int32_t skippy_session_position(
        const struct skippy_session * session) {
    return session != nullptr ? session->n_past : -1;
}

int32_t skippy_session_batch_size(
        const struct skippy_session * session) {
    return session != nullptr && session->ctx != nullptr ? llama_n_batch(session->ctx) : 0;
}

enum skippy_status skippy_session_begin_external_decode(
        struct skippy_session * session,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const skippy_runtime_config & config = session->stage_model->config;
    if (config.filter_tensors_on_load) {
        skippy_graph_filter filter;
        filter.enabled = true;
        filter.layer_start = config.layer_start;
        filter.layer_end = config.layer_end;
        filter.include_embeddings = config.include_embeddings;
        filter.include_output = config.include_output;
        skippy_graph_set_filter(filter);
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_session_end_external_decode(
        struct skippy_session * session,
        struct skippy_error ** out_error) {
    if (session == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    skippy_graph_clear_filter();
    return skippy_success(out_error);
}

enum skippy_status skippy_session_set_position(
        struct skippy_session * session,
        int32_t n_past,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (n_past < 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "n_past must be non-negative");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    session->n_past = n_past;
    if (session->token_history.size() > static_cast<size_t>(n_past)) {
        session->token_history.resize(static_cast<size_t>(n_past));
        skippy_clear_chat_sampling(session);
    }
    if (session->signal_history.size() > static_cast<size_t>(n_past)) {
        session->signal_history.resize(static_cast<size_t>(n_past));
    }
    skippy_mtp_clear_session_state(session);
    return skippy_success(out_error);
}

enum skippy_status skippy_session_sample_current(
        struct skippy_session * session,
        const struct skippy_sampling_config * sampling,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || out_predicted_token == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and out_predicted_token are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    *out_predicted_token = skippy_sample_token(session, sampling);
    return skippy_success(out_error);
}

enum skippy_status skippy_session_configure_chat_sampling(
        struct skippy_session * session,
        const struct skippy_sampling_config * sampling,
        const char * metadata_json,
        uint64_t prompt_token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || session->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (metadata_json == nullptr || metadata_json[0] == '\0') {
        skippy_clear_chat_sampling(session);
        return skippy_success(out_error);
    }

    try {
        json metadata = json::parse(metadata_json);
        skippy_clear_chat_sampling(session);
        session->sampling_chain = skippy_build_sampling_chain(session, sampling);
        if (session->sampling_chain == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to initialize chat sampling chain");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        session->chat_sampling_metadata = metadata_json;
        if (sampling != nullptr) {
            session->sampling_config = *sampling;
            session->sampling_config_valid = true;
        }
        session->grammar_generated_start = prompt_token_count;
        session->sampling_accepted_token_count = 0;
        if (!skippy_init_chat_grammar_sampler(session, metadata, out_error)) {
            skippy_clear_chat_sampling(session);
            return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
        }
    } catch (const std::exception & e) {
        skippy_clear_chat_sampling(session);
        return skippy_success(out_error);
    } catch (...) {
        skippy_clear_chat_sampling(session);
        return skippy_success(out_error);
    }
    return skippy_success(out_error);
}

enum skippy_status skippy_session_reset(
        struct skippy_session * session,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    session->ctx->synchronize();
    llama_memory_t memory = session->ctx->get_memory();
    if (memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "runtime memory is unavailable");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    const llama_pos reset_from = session->borrowed_sequence
        ? static_cast<llama_pos>(session->borrowed_prefix_tokens)
        : -1;
    if (!llama_memory_seq_rm(memory, session->seq_id, reset_from, -1)) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to reset execution lane");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (!session->borrowed_sequence) {
        llama_memory_seq_rm(memory, session->checkpoint_seq_id, -1, -1);
    }
    session->n_past = session->borrowed_sequence ? session->borrowed_prefix_tokens : 0;
    session->checkpoint_valid = false;
    session->checkpoint_n_past = 0;
    if (session->borrowed_sequence) {
        if (session->token_history.size() > static_cast<size_t>(session->borrowed_prefix_tokens)) {
            session->token_history.resize(static_cast<size_t>(session->borrowed_prefix_tokens));
        }
    } else {
        session->token_history.clear();
    }
    session->signal_history.clear();
    skippy_clear_chat_sampling(session);
    skippy_mtp_clear_session_state(session);
    session->ctx->synchronize();
    return skippy_success(out_error);
}

enum skippy_status skippy_session_free(
        struct skippy_session * session,
        struct skippy_error ** out_error) {
    if (session != nullptr) {
        if (session->ctx != nullptr) {
            if (session->preserve_prefix_on_free && !session->borrowed_sequence && session->stage_model != nullptr) {
                session->ctx->synchronize();
                if (llama_memory_t memory = session->ctx->get_memory()) {
                    llama_memory_seq_rm(
                            memory,
                            session->seq_id,
                            static_cast<llama_pos>(session->preserve_prefix_tokens),
                            -1);
                    llama_memory_seq_rm(memory, session->checkpoint_seq_id, -1, -1);
                }
                session->ctx->synchronize();
            } else {
                skippy_session_reset(session, nullptr);
                if (!session->borrowed_sequence && session->stage_model != nullptr && session->seq_id >= 0) {
                    const size_t lane = static_cast<size_t>(session->seq_id);
                    if (lane < session->stage_model->lane_resident_prefix_tokens.size()) {
                        session->stage_model->lane_resident_prefix_tokens[lane].clear();
                    }
                }
            }
        }
        if (!session->borrowed_sequence && session->stage_model != nullptr && session->seq_id >= 0) {
            const size_t lane = static_cast<size_t>(session->seq_id);
            if (lane < session->stage_model->lane_in_use.size()) {
                session->stage_model->lane_in_use[lane] = false;
            }
        }
        skippy_mtp_clear_session_state(session);
        delete session;
    }
    return skippy_success(out_error);
}

enum skippy_status skippy_session_last_token_signal(
        struct skippy_session * session,
        struct skippy_token_signal * out_signal,
        struct skippy_error ** out_error) {
    if (session == nullptr || out_signal == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and out_signal are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (session->signal_history.size() < static_cast<size_t>(std::max(session->n_past, 0))) {
        skippy_record_signal(session, -1);
    }
    if (session->signal_history.empty()) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "no token signal is available");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    *out_signal = session->signal_history.back();
    return skippy_success(out_error);
}

enum skippy_status skippy_session_signal_window(
        struct skippy_session * session,
        uint32_t window_tokens,
        struct skippy_generation_signal_window * out_window,
        struct skippy_error ** out_error) {
    if (session == nullptr || out_window == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and out_window are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (window_tokens == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "window_tokens must be greater than zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (session->signal_history.empty()) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "no token signals are available");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const size_t count = std::min<size_t>(window_tokens, session->signal_history.size());
    const size_t start = session->signal_history.size() - count;
    double entropy_sum = 0.0;
    double margin_sum = 0.0;
    float max_entropy = 0.0f;
    float min_margin = std::numeric_limits<float>::infinity();
    uint32_t high_entropy_count = 0;
    for (size_t i = start; i < session->signal_history.size(); ++i) {
        const skippy_token_signal & signal = session->signal_history[i];
        entropy_sum += signal.entropy;
        margin_sum += signal.margin;
        max_entropy = std::max(max_entropy, signal.entropy);
        min_margin = std::min(min_margin, signal.margin);
        if (signal.entropy > 4.0f) {
            ++high_entropy_count;
        }
    }

    uint32_t repetition_count = 0;
    if (!session->token_history.empty()) {
        const size_t token_count = std::min<size_t>(window_tokens, session->token_history.size());
        const size_t token_start = session->token_history.size() - token_count;
        for (size_t i = token_start + 1; i < session->token_history.size(); ++i) {
            if (session->token_history[i] == session->token_history[i - 1]) {
                ++repetition_count;
            }
        }
    }

    *out_window = {};
    out_window->token_count = static_cast<uint32_t>(count);
    out_window->mean_entropy = static_cast<float>(entropy_sum / static_cast<double>(count));
    out_window->max_entropy = max_entropy;
    out_window->mean_margin = static_cast<float>(margin_sum / static_cast<double>(count));
    out_window->min_margin = min_margin;
    out_window->high_entropy_count = high_entropy_count;
    out_window->repetition_count = repetition_count;
    return skippy_success(out_error);
}

enum skippy_status skippy_prefill_chunk(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const void * input_activations,
        size_t input_activation_bytes,
        void * output_activations,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        struct skippy_error ** out_error) {
    (void) input_activations;
    (void) input_activation_bytes;
    (void) output_activations;
    (void) output_activation_capacity;

    if (out_output_activation_bytes != nullptr) {
        *out_output_activation_bytes = 0;
    }

    return skippy_decode_tokens(session, token_ids, token_count, false, out_error);
}

enum skippy_status skippy_decode_step_sampled(
        struct skippy_session * session,
        llama_token token_id,
        const struct skippy_sampling_config * sampling,
        const void * input_activation,
        size_t input_activation_bytes,
        void * output_activation,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error) {
    (void) input_activation;
    (void) input_activation_bytes;
    (void) output_activation;
    (void) output_activation_capacity;

    if (out_output_activation_bytes != nullptr) {
        *out_output_activation_bytes = 0;
    }

    enum skippy_status status = skippy_decode_tokens(session, &token_id, 1, true, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (out_predicted_token != nullptr) {
        *out_predicted_token = skippy_sample_token(session, sampling);
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_decode_batch_sampled(
        struct skippy_session * const * sessions,
        const llama_token * token_ids,
        const struct skippy_sampling_config * const * sampling,
        size_t request_count,
        llama_token * out_predicted_tokens,
        size_t predicted_token_capacity,
        struct skippy_error ** out_error) {
    if (sessions == nullptr || token_ids == nullptr || request_count == 0 || out_predicted_tokens == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "sessions, token_ids, request_count, and out_predicted_tokens are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (predicted_token_capacity < request_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "predicted token output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }
    if (request_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "request_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_session * first = sessions[0];
    if (first == nullptr || first->ctx == nullptr || first->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "sessions must be active");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_filtered(first) && first->stage_model->config.layer_start > 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "batched token decode requires the first runtime slice or a full model");
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    const int32_t n_tokens = static_cast<int32_t>(request_count);
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        skippy_session * session = sessions[i];
        if (session == nullptr || session->ctx != first->ctx || session->stage_model != first->stage_model) {
            llama_batch_free(batch);
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "all sessions must belong to the same stage model");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
            llama_batch_free(batch);
            skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "batched token decode requires the first runtime slice or a full model");
            return SKIPPY_STATUS_UNSUPPORTED;
        }
        batch.token[i] = token_ids[i];
        batch.pos[i] = session->n_past;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = session->seq_id;
        batch.logits[i] = 1;
    }

    skippy_graph_filter_scope graph_filter_scope(&first->stage_model->config);
    const int32_t rc = llama_decode(first->ctx, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama_decode failed");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    llama_batch_free(batch);

    for (int32_t i = 0; i < n_tokens; ++i) {
        skippy_session * session = sessions[i];
        session->n_past += 1;
        skippy_record_tokens(session, &token_ids[i], 1);
        skippy_record_signal(session, i);
        const skippy_sampling_config * request_sampling = sampling != nullptr ? sampling[i] : nullptr;
        out_predicted_tokens[i] = skippy_sample_token_ith(session, request_sampling, i);
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_verify_tokens(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error) {
    if (out_token_count != nullptr) {
        *out_token_count = token_count;
    }
    if (session == nullptr || session->ctx == nullptr || token_ids == nullptr || token_count == 0 || out_token_count == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session, token_ids, token_count, and out_token_count are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (output_tokens == nullptr || output_token_capacity < token_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "token output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }
    if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "non-first runtime slices require activation input");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    enum skippy_status status = skippy_verify_token_batch(session, token_ids, token_count, out_error);
    if (status == SKIPPY_STATUS_OK) {
        const int32_t n_tokens = static_cast<int32_t>(token_count);
        for (int32_t i = 0; i < n_tokens; ++i) {
            output_tokens[i] = skippy_greedy_sample_ith(session, i);
        }
    }
    return status;
}

static enum skippy_status skippy_prefill_chunk_frame_impl(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const llama_pos * positions,
        size_t position_count,
        bool request_logits,
        const struct skippy_sampling_config * sampling,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error) {
    if (out_predicted_token != nullptr) {
        *out_predicted_token = -1;
    }
    if (token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count must be greater than zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds activation descriptor range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    enum skippy_status status = skippy_validate_frame_input(
            session,
            input_desc,
            input_payload,
            token_count,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    status = skippy_prepare_output_activation_frame(
            session,
            token_count,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            output_desc,
            input_desc,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        status = skippy_decode_activation_frame(session, input_desc, input_payload, token_ids, positions, position_count, token_count, request_logits, out_error);
    } else {
        status = skippy_decode_tokens(session, token_ids, token_count, request_logits, out_error);
    }
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (out_predicted_token != nullptr) {
        *out_predicted_token = session->stage_model->config.include_output ?
                skippy_sample_token(session, sampling) : -1;
    }

    return skippy_copy_output_activation_frame(session, token_count, output_payload, input_desc, input_payload, out_error);
}

enum skippy_status skippy_prefill_chunk_frame(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        struct skippy_error ** out_error) {
    return skippy_prefill_chunk_frame_impl(
            session,
            token_ids,
            token_count,
            nullptr,
            0,
            false,
            nullptr,
            input_desc,
            input_payload,
            output_desc,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            nullptr,
            out_error);
}

enum skippy_status skippy_prefill_chunk_frame_sampled(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const struct skippy_sampling_config * sampling,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error) {
    return skippy_prefill_chunk_frame_impl(
            session,
            token_ids,
            token_count,
            nullptr,
            0,
            true,
            sampling,
            input_desc,
            input_payload,
            output_desc,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            out_predicted_token,
            out_error);
}

enum skippy_status skippy_prefill_chunk_frame_with_positions(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const llama_pos * positions,
        size_t position_count,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        struct skippy_error ** out_error) {
    return skippy_prefill_chunk_frame_impl(
            session,
            token_ids,
            token_count,
            positions,
            position_count,
            false,
            nullptr,
            input_desc,
            input_payload,
            output_desc,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            nullptr,
            out_error);
}

enum skippy_status skippy_prefill_chunk_frame_sampled_with_positions(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const llama_pos * positions,
        size_t position_count,
        const struct skippy_sampling_config * sampling,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error) {
    return skippy_prefill_chunk_frame_impl(
            session,
            token_ids,
            token_count,
            positions,
            position_count,
            true,
            sampling,
            input_desc,
            input_payload,
            output_desc,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            out_predicted_token,
            out_error);
}

enum skippy_status skippy_decode_step_frame_sampled(
        struct skippy_session * session,
        llama_token token_id,
        const struct skippy_sampling_config * sampling,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error) {
    enum skippy_status status = skippy_validate_frame_input(
            session,
            input_desc,
            input_payload,
            1,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    status = skippy_prepare_output_activation_frame(
            session,
            1,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            output_desc,
            input_desc,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        status = skippy_decode_activation_frame(session, input_desc, input_payload, &token_id, nullptr, 0, 1, true, out_error);
    } else {
        status = skippy_decode_tokens(session, &token_id, 1, true, out_error);
    }
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (out_predicted_token != nullptr) {
        *out_predicted_token = session->stage_model->config.include_output ? skippy_sample_token(session, sampling) : -1;
    }

    return skippy_copy_output_activation_frame(session, 1, output_payload, input_desc, input_payload, out_error);
}

enum skippy_status skippy_decode_step_frame_sampled_mtp_n1(
        struct skippy_session * session,
        llama_token token_id,
        const struct skippy_sampling_config * sampling,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_token,
        struct skippy_native_mtp_draft * out_mtp_draft,
        struct skippy_error ** out_error) {
    if (out_mtp_draft != nullptr) {
        *out_mtp_draft = {
            1,
            false,
            -1,
            0,
        };
    }

    enum skippy_status status = skippy_decode_step_frame_sampled(
            session,
            token_id,
            sampling,
            input_desc,
            input_payload,
            output_desc,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            out_predicted_token,
            out_error);
    if (status != SKIPPY_STATUS_OK || out_predicted_token == nullptr || *out_predicted_token < 0) {
        return status;
    }

    return skippy_mtp_propose_next(session, *out_predicted_token, out_mtp_draft, out_error);
}

enum skippy_status skippy_decode_step_frame_batch_sampled(
        struct skippy_session * const * sessions,
        const llama_token * token_ids,
        const struct skippy_sampling_config * const * sampling,
        const struct skippy_activation_desc * const * input_descs,
        const void * const * input_payloads,
        struct skippy_activation_desc * output_descs,
        void * const * output_payloads,
        const size_t * output_payload_capacities,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_tokens,
        size_t predicted_token_capacity,
        size_t request_count,
        struct skippy_error ** out_error) {
    if (sessions == nullptr || token_ids == nullptr || request_count == 0 || out_predicted_tokens == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "sessions, token_ids, request_count, and out_predicted_tokens are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (predicted_token_capacity < request_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "predicted token output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }
    if (request_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "request_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_session * first = sessions[0];
    if (first == nullptr || first->ctx == nullptr || first->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "sessions must be active");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int32_t n_tokens = static_cast<int32_t>(request_count);
    const int32_t n_embd = llama_model_n_embd(first->stage_model->model);
    const int32_t n_embd_inp = llama_model_n_embd_inp(first->stage_model->model);
    const int32_t n_pos_per_embd = first->stage_model->model->hparams.n_pos_per_embd();
    const bool activation_input = skippy_is_filtered(first) && first->stage_model->config.layer_start > 0;
    const bool request_logits = first->stage_model->config.include_output;
    const bool request_embeddings = skippy_emits_activation_frame(first);

    if (activation_input && (input_descs == nullptr || input_payloads == nullptr)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "batched downstream decode requires activation frame inputs");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const size_t hidden_bytes_per_request = skippy_activation_hidden_bytes(first, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        skippy_session * session = sessions[i];
        if (session == nullptr || session->ctx != first->ctx || session->stage_model != first->stage_model) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "all sessions must belong to the same stage model");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        const skippy_activation_desc * input_desc = input_descs != nullptr ? input_descs[i] : nullptr;
        const void * input_payload = input_payloads != nullptr ? input_payloads[i] : nullptr;
        enum skippy_status status = skippy_validate_frame_input(session, input_desc, input_payload, 1, out_error);
        if (status != SKIPPY_STATUS_OK) {
            return status;
        }
        if (input_desc != nullptr && input_desc->flags != 0) {
            skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "batched activation decode does not support activation sidebands yet");
            return SKIPPY_STATUS_UNSUPPORTED;
        }
        if (skippy_output_activation_flags(session, input_desc) != 0) {
            skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "batched activation decode does not support output activation sidebands yet");
            return SKIPPY_STATUS_UNSUPPORTED;
        }
        status = skippy_prepare_output_activation_frame(
                session,
                1,
                output_payloads != nullptr ? output_payloads[i] : nullptr,
                output_payload_capacities != nullptr ? output_payload_capacities[i] : 0,
                out_output_payload_bytes != nullptr ? &out_output_payload_bytes[i] : nullptr,
                output_descs != nullptr ? &output_descs[i] : nullptr,
                input_desc,
                out_error);
        if (status != SKIPPY_STATUS_OK) {
            return status;
        }
    }

    std::vector<float> embd_storage;
    std::vector<llama_token> token_storage;
    if (activation_input) {
        embd_storage.resize(static_cast<size_t>(n_tokens)*n_embd_inp);
        for (int32_t i = 0; i < n_tokens; ++i) {
            const skippy_activation_desc * input_desc = input_descs[i];
            const void * input_payload = input_payloads[i];
            float * dst = embd_storage.data() + static_cast<size_t>(i)*n_embd_inp;
            if ((input_desc->flags & SKIPPY_ACTIVATION_FLAG_GEMMA3N_ALTUP) != 0) {
                return SKIPPY_STATUS_UNSUPPORTED;
            } else if (n_embd_inp == n_embd) {
                std::memcpy(dst, input_payload, hidden_bytes_per_request);
            } else {
                std::memcpy(dst, input_payload, static_cast<size_t>(n_embd)*sizeof(float));
                std::memset(dst + n_embd, 0, static_cast<size_t>(n_embd_inp - n_embd)*sizeof(float));
            }
        }
    } else {
        token_storage.assign(token_ids, token_ids + request_count);
    }

    std::vector<llama_pos> pos_storage(static_cast<size_t>(n_tokens)*n_pos_per_embd);
    std::vector<int32_t> n_seq_id_storage(n_tokens, 1);
    std::vector<llama_seq_id> seq_id_values(n_tokens);
    std::vector<llama_seq_id *> seq_id_storage(n_tokens, nullptr);
    std::vector<int8_t> logits_storage(n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) {
        skippy_session * session = sessions[i];
        seq_id_values[i] = session->seq_id;
        seq_id_storage[i] = &seq_id_values[i];
        logits_storage[i] = (request_logits || request_embeddings) ? 1 : 0;
        if (n_pos_per_embd == 4) {
            const llama_pos position = session->n_past;
            pos_storage[               i] = position;
            pos_storage[    n_tokens + i] = position;
            pos_storage[2 * n_tokens + i] = position;
            pos_storage[3 * n_tokens + i] = 0;
        } else {
            pos_storage[i] = session->n_past;
        }
    }

    llama_batch batch = {
        /*n_tokens =*/ n_tokens,
        /*token    =*/ activation_input ? nullptr : token_storage.data(),
        /*embd     =*/ activation_input ? embd_storage.data() : nullptr,
        /*pos      =*/ pos_storage.data(),
        /*n_seq_id =*/ n_seq_id_storage.data(),
        /*seq_id   =*/ seq_id_storage.data(),
        /*logits   =*/ logits_storage.data(),
    };

    skippy_activation_tokens_scope activation_tokens_scope(token_ids, request_count);
    skippy_graph_filter_scope graph_filter_scope(&first->stage_model->config);
    const int32_t rc = llama_decode(first->ctx, batch);
    if (rc != 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama_decode failed");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    float * embeddings = nullptr;
    if (skippy_emits_activation_frame(first)) {
        embeddings = llama_get_embeddings(first->ctx);
        if (embeddings == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama embeddings output was not available");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        skippy_session * session = sessions[i];
        session->n_past += 1;
        if (!activation_input) {
            skippy_record_tokens(session, &token_ids[i], 1);
        }
        if (request_logits) {
            skippy_record_signal(session, i);
            const skippy_sampling_config * request_sampling = sampling != nullptr ? sampling[i] : nullptr;
            out_predicted_tokens[i] = skippy_sample_token_ith(session, request_sampling, i);
        } else {
            out_predicted_tokens[i] = -1;
        }

        if (embeddings != nullptr && output_payloads != nullptr && output_payloads[i] != nullptr) {
            std::memcpy(
                    output_payloads[i],
                    embeddings + static_cast<size_t>(i)*n_embd,
                    hidden_bytes_per_request);
        }
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_verify_tokens_frame_sampled(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const struct skippy_sampling_config * sampling,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error) {
    if (out_token_count != nullptr) {
        *out_token_count = 0;
    }
    if (token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count must be greater than zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds activation descriptor range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (session == nullptr || session->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    enum skippy_status status = skippy_validate_frame_input(
            session,
            input_desc,
            input_payload,
            token_count,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    status = skippy_prepare_output_activation_frame(
            session,
            token_count,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            output_desc,
            input_desc,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (session->stage_model->config.include_output) {
        if (out_token_count == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "out_token_count is required for output stages");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        *out_token_count = token_count;
        if (output_tokens == nullptr || output_token_capacity < token_count) {
            skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "token output buffer is too small");
            return SKIPPY_STATUS_BUFFER_TOO_SMALL;
        }
    }

    if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        status = session->stage_model->config.include_output ?
                skippy_verify_activation_frame(session, input_desc, input_payload, token_ids, token_count, out_error) :
                skippy_decode_activation_frame(session, input_desc, input_payload, token_ids, nullptr, 0, token_count, false, out_error);
    } else {
        status = session->stage_model->config.include_output ?
                skippy_verify_token_batch(session, token_ids, token_count, out_error) :
                skippy_decode_tokens(session, token_ids, token_count, false, out_error);
    }
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    status = skippy_copy_output_activation_frame(session, token_count, output_payload, input_desc, input_payload, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (session->stage_model->config.include_output) {
        const int32_t n_tokens = static_cast<int32_t>(token_count);
        for (int32_t i = 0; i < n_tokens; ++i) {
            output_tokens[i] = skippy_sample_token_ith(session, sampling, i);
        }
        if (output_token_capacity >= token_count + 2 && token_count > 0) {
            skippy_native_mtp_draft mtp_draft = {};
            status = skippy_mtp_propose_next(
                    session,
                    output_tokens[token_count - 1],
                    &mtp_draft,
                    out_error);
            if (status != SKIPPY_STATUS_OK) {
                return status;
            }
            if (mtp_draft.available) {
                output_tokens[token_count] = mtp_draft.token_id;
                output_tokens[token_count + 1] = static_cast<llama_token>(std::min<int64_t>(
                        std::max<int64_t>(mtp_draft.proposal_compute_us, 0),
                        std::numeric_limits<llama_token>::max()));
                *out_token_count = token_count + 2;
            }
        }
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_session_copy_output_activation_frame(
        struct skippy_session * session,
        size_t token_count,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        struct skippy_error ** out_error) {
    if (token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count must be greater than zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    enum skippy_status status = skippy_prepare_output_activation_frame(
            session,
            token_count,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            output_desc,
            nullptr,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    return skippy_copy_output_activation_frame(session, token_count, output_payload, nullptr, nullptr, out_error);
}

static enum skippy_status skippy_validate_state_range(
        skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || session->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    const skippy_runtime_config & config = session->stage_model->config;
    const int32_t expected_layer_start = config.filter_tensors_on_load ? config.layer_start : 0;
    const int32_t expected_layer_end = config.filter_tensors_on_load ?
            config.layer_end : skippy_stage_layer_count(session->stage_model->model);
    if (layer_start != expected_layer_start || layer_end != expected_layer_end) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "state range must match the session layer range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    return SKIPPY_STATUS_OK;
}

static void skippy_update_session_state_after_import(
        skippy_session * session,
        uint32_t imported_cells) {
    session->n_past = std::max<int32_t>(
            session->n_past,
            static_cast<int32_t>(std::min<uint32_t>(
                    imported_cells,
                    static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))));
    if (session->token_history.size() > static_cast<size_t>(session->n_past)) {
        session->token_history.resize(static_cast<size_t>(session->n_past));
    }
    if (session->signal_history.size() > static_cast<size_t>(session->n_past)) {
        session->signal_history.resize(static_cast<size_t>(session->n_past));
    }
    session->checkpoint_valid = false;
}

enum skippy_status skippy_export_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error) {
    if (out_bytes != nullptr) {
        *out_bytes = 0;
    }
    enum skippy_status status = skippy_validate_state_range(session, layer_start, layer_end, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    session->ctx->synchronize();
    const size_t required = llama_state_seq_get_size_ext(
            session->ctx,
            session->seq_id,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (out_bytes != nullptr) {
        *out_bytes = required;
    }
    if (required == 0) {
        return skippy_success(out_error);
    }
    if (output == nullptr || output_capacity < required) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "state output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }

    const size_t written = llama_state_seq_get_data_ext(
            session->ctx,
            static_cast<uint8_t *>(output),
            output_capacity,
            session->seq_id,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (written != required) {
        if (out_bytes != nullptr) {
            *out_bytes = written;
        }
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to export sequence state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_import_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error) {
    enum skippy_status status = skippy_validate_state_range(session, layer_start, layer_end, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    if (input == nullptr && input_bytes > 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "state input is required when input_bytes is non-zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    uint32_t imported_cells = 0;
    if (input_bytes >= sizeof(imported_cells)) {
        std::memcpy(&imported_cells, input, sizeof(imported_cells));
    }

    session->ctx->synchronize();
    const size_t read = llama_state_seq_set_data_ext(
            session->ctx,
            static_cast<const uint8_t *>(input),
            input_bytes,
            session->seq_id,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (read != input_bytes) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to import sequence state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    skippy_update_session_state_after_import(session, imported_cells);
    session->ctx->synchronize();

    return skippy_success(out_error);
}

enum skippy_status skippy_export_full_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error) {
    if (out_bytes != nullptr) {
        *out_bytes = 0;
    }
    enum skippy_status status = skippy_validate_state_range(session, layer_start, layer_end, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    session->ctx->synchronize();
    const size_t required = llama_state_seq_get_size(session->ctx, session->seq_id);
    if (out_bytes != nullptr) {
        *out_bytes = required;
    }
    if (required == 0) {
        return skippy_success(out_error);
    }
    if (output == nullptr || output_capacity < required) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "full state output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }

    const size_t written = llama_state_seq_get_data(
            session->ctx,
            static_cast<uint8_t *>(output),
            output_capacity,
            session->seq_id);
    if (written != required) {
        if (out_bytes != nullptr) {
            *out_bytes = written;
        }
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to export full sequence state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_import_full_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error) {
    enum skippy_status status = skippy_validate_state_range(session, layer_start, layer_end, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    if (input == nullptr && input_bytes > 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "state input is required when input_bytes is non-zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    uint32_t imported_cells = 0;
    if (input_bytes >= 2 * sizeof(uint32_t)) {
        std::memcpy(&imported_cells, static_cast<const uint8_t *>(input) + sizeof(uint32_t), sizeof(imported_cells));
    }

    session->ctx->synchronize();
    const size_t read = llama_state_seq_set_data(
            session->ctx,
            static_cast<const uint8_t *>(input),
            input_bytes,
            session->seq_id);
    if (read != input_bytes) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to import full sequence state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    skippy_update_session_state_after_import(session, imported_cells);
    session->ctx->synchronize();

    return skippy_success(out_error);
}

static llama_memory_recurrent * skippy_get_recurrent_memory(
        skippy_session * session,
        skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return nullptr;
    }

    llama_memory_t memory = session->ctx->get_memory();
    if (memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "runtime memory is unavailable");
        return nullptr;
    }

    if (auto * recurrent = dynamic_cast<llama_memory_recurrent *>(memory)) {
        return recurrent;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
        return hybrid->get_mem_recr();
    }
    if (auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(memory)) {
        return hybrid_iswa->get_mem_recr();
    }

    return nullptr;
}

enum skippy_status skippy_export_recurrent_state(
        struct skippy_session * session,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error) {
    if (out_bytes == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "out_bytes is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    *out_bytes = 0;
    skippy_error * recurrent_error = nullptr;
    llama_memory_recurrent * recurrent = skippy_get_recurrent_memory(session, &recurrent_error);
    if (recurrent_error != nullptr) {
        if (out_error != nullptr) {
            *out_error = recurrent_error;
        } else {
            skippy_error_free(recurrent_error);
        }
        return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (recurrent == nullptr) {
        return skippy_success(out_error);
    }

    session->ctx->synchronize();
    const size_t bytes = llama_state_seq_get_size_ext(
            session->ctx,
            session->seq_id,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    *out_bytes = bytes;
    if (bytes == 0) {
        return skippy_success(out_error);
    }
    if (output == nullptr || output_capacity < bytes) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "recurrent state output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }

    const size_t written = llama_state_seq_get_data_ext(
            session->ctx,
            static_cast<uint8_t *>(output),
            output_capacity,
            session->seq_id,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (written != bytes) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to export recurrent state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_import_recurrent_state(
        struct skippy_session * session,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error) {
    if (input_bytes == 0) {
        return skippy_success(out_error);
    }
    if (input == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "recurrent state input is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_error * recurrent_error = nullptr;
    llama_memory_recurrent * recurrent = skippy_get_recurrent_memory(session, &recurrent_error);
    if (recurrent_error != nullptr) {
        if (out_error != nullptr) {
            *out_error = recurrent_error;
        } else {
            skippy_error_free(recurrent_error);
        }
        return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (recurrent == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "runtime has no recurrent memory");
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    std::vector<uint8_t> remapped(
            static_cast<const uint8_t *>(input),
            static_cast<const uint8_t *>(input) + input_bytes);
    constexpr size_t seq_id_offset = sizeof(uint32_t);
    if (remapped.size() < seq_id_offset + sizeof(llama_seq_id)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "recurrent state input is too small");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    std::memcpy(remapped.data() + seq_id_offset, &session->seq_id, sizeof(session->seq_id));

    session->ctx->synchronize();
    const size_t read = llama_state_seq_set_data_ext(
            session->ctx,
            remapped.data(),
            remapped.size(),
            session->seq_id,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (read != remapped.size()) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to import recurrent state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    session->ctx->synchronize();

    return skippy_success(out_error);
}

static llama_kv_cache * skippy_get_kv_cache(
        skippy_session * session,
        skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return nullptr;
    }

    llama_memory_t memory = session->ctx->get_memory();
    if (memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "runtime memory is unavailable");
        return nullptr;
    }

    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
        llama_kv_cache * kv = hybrid->get_mem_attn();
        if (kv == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "runtime has no attention KV cache");
        }
        return kv;
    }
    if (auto * kv = dynamic_cast<llama_kv_cache *>(memory)) {
        return kv;
    }

    skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "runtime memory type is not supported for native KV pages");
    return nullptr;
}

enum skippy_status skippy_export_kv_page(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        uint64_t token_start,
        uint64_t token_count,
        struct skippy_kv_page_desc * out_desc,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error) {
    llama_kv_cache * kv = skippy_get_kv_cache(session, out_error);
    if (kv == nullptr) {
        return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
    }
    session->ctx->synchronize();

    std::string error;
    const bool ok = kv->stage_export_kv_page(
            session->seq_id,
            layer_start,
            layer_end,
            token_start,
            token_count,
            out_desc,
            output,
            output_capacity,
            out_bytes,
            error);
    if (!ok) {
        const bool too_small = out_bytes != nullptr && *out_bytes > output_capacity;
        const enum skippy_status status = too_small ? SKIPPY_STATUS_BUFFER_TOO_SMALL : SKIPPY_STATUS_RUNTIME_ERROR;
        skippy_set_error(out_error, status, error.c_str());
        return status;
    }
    if (output == nullptr || (out_bytes != nullptr && *out_bytes > output_capacity)) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "native KV page output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_import_kv_page(
        struct skippy_session * session,
        const struct skippy_kv_page_desc * desc,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error) {
    if (desc == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "native KV page descriptor is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_kv_cache * kv = skippy_get_kv_cache(session, out_error);
    if (kv == nullptr) {
        return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
    }

    std::string error;
    if (!kv->stage_import_kv_page(session->seq_id, *desc, input, input_bytes, error)) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, error.c_str());
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    session->n_past = std::max<int32_t>(
            session->n_past,
            static_cast<int32_t>(std::min<uint64_t>(
                    desc->token_start + desc->token_count,
                    static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))));
    session->ctx->synchronize();

    return skippy_success(out_error);
}

enum skippy_status skippy_trim_session(
        struct skippy_session * session,
        uint64_t token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<uint64_t>(session->n_past)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "cannot trim session beyond current token count");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    session->ctx->synchronize();
    llama_memory_t memory = session->ctx->get_memory();
    if (memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "runtime memory is unavailable");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    const llama_pos p0 = static_cast<llama_pos>(token_count);
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
        if (!hybrid->seq_rm(session->seq_id, p0, -1)) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to trim hybrid memory suffix");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
    } else if (auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(memory)) {
        if (!hybrid_iswa->seq_rm(session->seq_id, p0, -1)) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to trim hybrid ISWA memory suffix");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
    } else if (auto * kv = dynamic_cast<llama_kv_cache *>(memory)) {
        if (!kv->seq_rm(session->seq_id, p0, -1)) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to trim native KV suffix");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
    } else if (dynamic_cast<llama_memory_recurrent *>(memory) == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "runtime memory type is not supported for trim");
        return SKIPPY_STATUS_UNSUPPORTED;
    }
    session->n_past = static_cast<int32_t>(token_count);
    if (session->token_history.size() > token_count) {
        session->token_history.resize(static_cast<size_t>(token_count));
        skippy_clear_chat_sampling(session);
    }
    if (session->signal_history.size() > token_count) {
        session->signal_history.resize(static_cast<size_t>(token_count));
    }
    skippy_mtp_clear_session_state(session);
    session->ctx->synchronize();

    return skippy_success(out_error);
}

static bool skippy_is_reserved_sequence_id(
        const skippy_session * session,
        int32_t seq_id) {
    if (session == nullptr || session->stage_model == nullptr || seq_id < 0) {
        return true;
    }
    const int32_t reserved = static_cast<int32_t>(session->stage_model->lane_count) * 2;
    return seq_id < reserved;
}

static bool skippy_is_reserved_sequence_id_for_model(
        const skippy_model * model,
        int32_t seq_id) {
    if (model == nullptr || seq_id < 0) {
        return true;
    }
    const int32_t reserved = static_cast<int32_t>(model->lane_count) * 2;
    return seq_id < reserved;
}

static enum skippy_status skippy_get_resident_prefix_memory(
        skippy_session * session,
        llama_memory_t * out_memory,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || out_memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_memory_t memory = session->ctx->get_memory();
    if (memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "runtime memory is unavailable");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    skippy_error * recurrent_error = nullptr;
    llama_memory_recurrent * recurrent = skippy_get_recurrent_memory(session, &recurrent_error);
    if (recurrent_error != nullptr) {
        if (out_error != nullptr) {
            *out_error = recurrent_error;
        } else {
            skippy_error_free(recurrent_error);
        }
        return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (recurrent != nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "resident prefix cache is not enabled for recurrent memory");
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    *out_memory = memory;
    return SKIPPY_STATUS_OK;
}

enum skippy_status skippy_session_save_prefix(
        struct skippy_session * session,
        int32_t cache_seq_id,
        uint64_t token_count,
        struct skippy_error ** out_error) {
    if (token_count > static_cast<uint64_t>(std::numeric_limits<llama_pos>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds llama position range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (session == nullptr || token_count > static_cast<uint64_t>(session->n_past)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "cannot save prefix beyond current session position");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_reserved_sequence_id(session, cache_seq_id)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "cache sequence id overlaps execution lanes");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_memory_t memory = nullptr;
    enum skippy_status status = skippy_get_resident_prefix_memory(session, &memory, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    session->ctx->synchronize();
    llama_memory_seq_rm(memory, cache_seq_id, -1, -1);
    if (token_count > 0) {
        llama_memory_seq_cp(memory, session->seq_id, cache_seq_id, 0, static_cast<llama_pos>(token_count));
    }
    if (!session->borrowed_sequence && session->stage_model != nullptr && session->seq_id >= 0) {
        const size_t lane = static_cast<size_t>(session->seq_id);
        if (lane < session->stage_model->lane_resident_prefix_tokens.size()
                && session->token_history.size() >= static_cast<size_t>(token_count)) {
            auto & lane_tokens = session->stage_model->lane_resident_prefix_tokens[lane];
            lane_tokens.assign(
                    session->token_history.begin(),
                    session->token_history.begin() + static_cast<ptrdiff_t>(token_count));
            session->preserve_prefix_on_free = true;
            session->preserve_prefix_tokens = static_cast<int32_t>(token_count);
        }
    }
    session->ctx->synchronize();
    return skippy_success(out_error);
}

enum skippy_status skippy_session_restore_prefix(
        struct skippy_session * session,
        int32_t cache_seq_id,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** out_error) {
    if (token_count > static_cast<size_t>(std::numeric_limits<llama_pos>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds llama position range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > 0 && token_ids == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_ids are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_reserved_sequence_id(session, cache_seq_id)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "cache sequence id overlaps execution lanes");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_memory_t memory = nullptr;
    enum skippy_status status = skippy_get_resident_prefix_memory(session, &memory, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    session->ctx->synchronize();
    if (token_count > 0) {
        const llama_pos max_pos = llama_memory_seq_pos_max(memory, cache_seq_id);
        if (max_pos < static_cast<llama_pos>(token_count - 1)) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "resident prefix sequence is incomplete");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
    }
    if (!llama_memory_seq_rm(memory, session->seq_id, -1, -1)) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to clear execution lane");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (token_count > 0) {
        llama_memory_seq_cp(memory, cache_seq_id, session->seq_id, 0, static_cast<llama_pos>(token_count));
    }
    session->n_past = static_cast<int32_t>(token_count);
    if (token_count == 0) {
        session->token_history.clear();
    } else {
        session->token_history.assign(token_ids, token_ids + token_count);
    }
    session->signal_history.clear();
    session->checkpoint_valid = false;
    session->checkpoint_n_past = 0;
    session->checkpoint_token_history_size = 0;
    session->checkpoint_signal_history_size = 0;
    session->ctx->synchronize();
    return skippy_success(out_error);
}

enum skippy_status skippy_session_create_from_resident_prefix(
        struct skippy_model * model,
        int32_t cache_seq_id,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_session ** out_session,
        struct skippy_error ** out_error) {
    if (model == nullptr || model->model == nullptr || model->ctx == nullptr || out_session == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model and out_session are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    *out_session = nullptr;
    if (token_count > static_cast<size_t>(std::numeric_limits<llama_pos>::max())
            || token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds supported position range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > 0 && token_ids == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_ids are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_reserved_sequence_id_for_model(model, cache_seq_id)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "cache sequence id overlaps execution lanes");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    int32_t seq_id = -1;
    for (uint32_t i = 0; i < model->lane_in_use.size(); ++i) {
        if (!model->lane_in_use[i]) {
            model->lane_in_use[i] = true;
            seq_id = static_cast<int32_t>(i);
            break;
        }
    }
    if (seq_id < 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "no skippy execution lane is available");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    skippy_session temp{};
    temp.stage_model = model;
    temp.ctx = model->ctx;
    llama_memory_t memory = nullptr;
    enum skippy_status status = skippy_get_resident_prefix_memory(&temp, &memory, out_error);
    if (status != SKIPPY_STATUS_OK) {
        model->lane_in_use[static_cast<size_t>(seq_id)] = false;
        return status;
    }

    model->ctx->synchronize();
    if (token_count > 0) {
        const llama_pos max_pos = llama_memory_seq_pos_max(memory, cache_seq_id);
        if (max_pos < static_cast<llama_pos>(token_count - 1)) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "resident prefix sequence is incomplete");
            model->lane_in_use[static_cast<size_t>(seq_id)] = false;
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
    }
    bool lane_prefix_hit = false;
    const size_t lane = static_cast<size_t>(seq_id);
    if (lane < model->lane_resident_prefix_tokens.size()) {
        const auto & lane_tokens = model->lane_resident_prefix_tokens[lane];
        lane_prefix_hit = lane_tokens.size() == token_count
                && (token_count == 0 || std::equal(lane_tokens.begin(), lane_tokens.end(), token_ids));
    }
    if (!lane_prefix_hit) {
        if (!llama_memory_seq_rm(memory, seq_id, -1, -1)) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to clear borrowed execution lane");
            model->lane_in_use[static_cast<size_t>(seq_id)] = false;
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        if (token_count > 0) {
            llama_memory_seq_cp(memory, cache_seq_id, seq_id, 0, static_cast<llama_pos>(token_count));
        }
    }
    if (!llama_memory_seq_rm(memory, seq_id, static_cast<llama_pos>(token_count), -1)) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to trim borrowed execution lane");
        model->lane_in_use[static_cast<size_t>(seq_id)] = false;
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    skippy_session * session = new skippy_session{};
    session->stage_model = model;
    session->ctx = model->ctx;
    session->seq_id = seq_id;
    session->checkpoint_seq_id = seq_id + static_cast<int32_t>(model->lane_count);
    session->n_past = static_cast<int32_t>(token_count);
    session->borrowed_sequence = false;
    session->borrowed_prefix_tokens = 0;
    session->checkpoint_valid = false;
    session->checkpoint_n_past = 0;
    if (token_count > 0) {
        session->token_history.assign(token_ids, token_ids + token_count);
    }
    model->ctx->synchronize();
    *out_session = session;
    return skippy_success(out_error);
}

enum skippy_status skippy_session_drop_sequence(
        struct skippy_session * session,
        int32_t seq_id,
        struct skippy_error ** out_error) {
    if (skippy_is_reserved_sequence_id(session, seq_id)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "sequence id overlaps execution lanes");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_memory_t memory = nullptr;
    enum skippy_status status = skippy_get_resident_prefix_memory(session, &memory, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    session->ctx->synchronize();
    llama_memory_seq_rm(memory, seq_id, -1, -1);
    session->ctx->synchronize();
    return skippy_success(out_error);
}

enum skippy_status skippy_checkpoint_session(
        struct skippy_session * session,
        uint64_t * out_token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (out_token_count == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "out_token_count is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_error * recurrent_error = nullptr;
    llama_memory_recurrent * recurrent = skippy_get_recurrent_memory(session, &recurrent_error);
    if (recurrent_error != nullptr) {
        if (out_error != nullptr) {
            *out_error = recurrent_error;
        } else {
            skippy_error_free(recurrent_error);
        }
        return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (recurrent != nullptr) {
        if (llama_n_seq_max(session->ctx) < 2) {
            skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "native recurrent checkpoint requires n_seq_max >= 2");
            return SKIPPY_STATUS_UNSUPPORTED;
        }
        recurrent->seq_cp(session->seq_id, session->checkpoint_seq_id, -1, -1);
    }

    session->checkpoint_valid = true;
    session->checkpoint_n_past = session->n_past;
    session->checkpoint_token_history_size = session->token_history.size();
    session->checkpoint_signal_history_size = session->signal_history.size();
    *out_token_count = static_cast<uint64_t>(session->n_past);
    return skippy_success(out_error);
}

enum skippy_status skippy_restore_session_checkpoint(
        struct skippy_session * session,
        uint64_t token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (!session->checkpoint_valid) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session checkpoint is not available");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count != static_cast<uint64_t>(session->checkpoint_n_past)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session checkpoint token count mismatch");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    enum skippy_status status = skippy_trim_session(session, token_count, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    skippy_error * recurrent_error = nullptr;
    llama_memory_recurrent * recurrent = skippy_get_recurrent_memory(session, &recurrent_error);
    if (recurrent_error != nullptr) {
        if (out_error != nullptr) {
            *out_error = recurrent_error;
        } else {
            skippy_error_free(recurrent_error);
        }
        return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (recurrent != nullptr) {
        if (llama_n_seq_max(session->ctx) < 2) {
            skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "native recurrent checkpoint requires n_seq_max >= 2");
            return SKIPPY_STATUS_UNSUPPORTED;
        }
        recurrent->seq_cp(session->checkpoint_seq_id, session->seq_id, -1, -1);
        recurrent->seq_rm(session->checkpoint_seq_id, -1, -1);
        session->ctx->synchronize();
    }

    session->n_past = session->checkpoint_n_past;
    if (session->token_history.size() > session->checkpoint_token_history_size) {
        session->token_history.resize(session->checkpoint_token_history_size);
    }
    if (session->signal_history.size() > session->checkpoint_signal_history_size) {
        session->signal_history.resize(session->checkpoint_signal_history_size);
    }
    session->checkpoint_valid = false;
    return skippy_success(out_error);
}

enum skippy_status skippy_tokenize(
        struct skippy_model * model,
        const char * text,
        bool add_special,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error) {
    if (model == nullptr || model->model == nullptr || text == nullptr || out_token_count == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model, text, and out_token_count are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (output_token_capacity > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token output capacity exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model->model);
    const int32_t result = llama_tokenize(
            vocab,
            text,
            static_cast<int32_t>(std::strlen(text)),
            output_tokens,
            static_cast<int32_t>(output_token_capacity),
            add_special,
            true);
    if (result < 0) {
        *out_token_count = static_cast<size_t>(-result);
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "token output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }

    *out_token_count = static_cast<size_t>(result);
    return skippy_success(out_error);
}

enum skippy_status skippy_detokenize(
        struct skippy_model * model,
        const llama_token * tokens,
        size_t token_count,
        char * output_text,
        size_t output_text_capacity,
        size_t * out_text_bytes,
        struct skippy_error ** out_error) {
    if (model == nullptr || model->model == nullptr || tokens == nullptr || out_text_bytes == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model, tokens, and out_text_bytes are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            output_text_capacity > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "detokenize inputs exceed int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model->model);
    std::string text;
    for (size_t i = 0; i < token_count; ++i) {
        std::string piece;
        piece.resize(piece.capacity());
        int32_t piece_bytes = llama_token_to_piece(
                vocab,
                tokens[i],
                piece.data(),
                static_cast<int32_t>(piece.size()),
                0,
                true);
        if (piece_bytes < 0) {
            piece.resize(static_cast<size_t>(-piece_bytes));
            piece_bytes = llama_token_to_piece(
                    vocab,
                    tokens[i],
                    piece.data(),
                    static_cast<int32_t>(piece.size()),
                    0,
                    true);
        }
        if (piece_bytes < 0) {
            skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "token piece output buffer is too small");
            return SKIPPY_STATUS_BUFFER_TOO_SMALL;
        }
        piece.resize(static_cast<size_t>(piece_bytes));
        text += piece;
    }

    *out_text_bytes = text.size();
    if (output_text_capacity < text.size()) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "text output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }
    if (!text.empty() && output_text == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "output_text is required when output capacity is sufficient");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (!text.empty()) {
        std::memcpy(output_text, text.data(), text.size());
    }
    return skippy_success(out_error);
}

enum skippy_status skippy_token_is_eog(
        struct skippy_model * model,
        llama_token token,
        bool * out_is_eog,
        struct skippy_error ** out_error) {
    if (model == nullptr || model->model == nullptr || out_is_eog == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model and out_is_eog are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model->model);
    *out_is_eog = llama_vocab_is_eog(vocab, token);
    return skippy_success(out_error);
}

const struct llama_model * skippy_model_native_model(
        const struct skippy_model * model) {
    if (model == nullptr) {
        return nullptr;
    }
    return model->model;
}

enum skippy_status skippy_model_info_open(
        const char * path,
        struct skippy_model_info ** out_info,
        struct skippy_error ** out_error) {
    if (path == nullptr || out_info == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "path and out_info are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    *out_info = nullptr;

    skippy_model_info * info = skippy_model_info_open_owned(path);
    if (info == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "failed to open GGUF model metadata");
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    *out_info = info;
    return skippy_success(out_error);
}

enum skippy_status skippy_model_info_free(
        struct skippy_model_info * info,
        struct skippy_error ** out_error) {
    skippy_model_info_delete(info);
    return skippy_success(out_error);
}

enum skippy_status skippy_model_info_tensor_count(
        struct skippy_model_info * info,
        size_t * out_count,
        struct skippy_error ** out_error) {
    if (info == nullptr || info->ctx == nullptr || out_count == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "info and out_count are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int64_t n_tensors = gguf_get_n_tensors(info->ctx);
    if (n_tensors < 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "GGUF reported negative tensor count");
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    *out_count = static_cast<size_t>(n_tensors);
    return skippy_success(out_error);
}

enum skippy_status skippy_model_info_tensor_at(
        struct skippy_model_info * info,
        size_t index,
        struct skippy_tensor_info * out_tensor,
        struct skippy_error ** out_error) {
    if (info == nullptr || info->ctx == nullptr || out_tensor == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "info and out_tensor are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int64_t n_tensors = gguf_get_n_tensors(info->ctx);
    if (index >= static_cast<size_t>(n_tensors)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "tensor index is out of range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int64_t tensor_id = static_cast<int64_t>(index);
    const char * name = gguf_get_tensor_name(info->ctx, tensor_id);
    const int32_t layer_index = skippy_layer_from_name(name);

    out_tensor->name = name;
    out_tensor->layer_index = layer_index;
    out_tensor->role = skippy_role_from_name(name, layer_index);
    out_tensor->ggml_type = static_cast<uint32_t>(gguf_get_tensor_type(info->ctx, tensor_id));
    out_tensor->byte_size = static_cast<uint64_t>(gguf_get_tensor_size(info->ctx, tensor_id));
    out_tensor->element_count = index < info->tensors.size() ?
            skippy_tensor_element_count(info->tensors[index]) : 0;
    return skippy_success(out_error);
}

enum skippy_status skippy_slice_plan_create(
        struct skippy_model_info * info,
        struct skippy_slice_plan ** out_plan,
        struct skippy_error ** out_error) {
    if (info == nullptr || info->ctx == nullptr || out_plan == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "info and out_plan are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    *out_plan = new skippy_slice_plan{};
    return skippy_success(out_error);
}

enum skippy_status skippy_slice_plan_free(
        struct skippy_slice_plan * plan,
        struct skippy_error ** out_error) {
    delete plan;
    return skippy_success(out_error);
}

enum skippy_status skippy_slice_plan_add_layer_range(
        struct skippy_slice_plan * plan,
        int32_t stage_index,
        int32_t layer_start,
        int32_t layer_end,
        bool include_embeddings,
        bool include_output,
        struct skippy_error ** out_error) {
    if (plan == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "plan is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (stage_index < 0 || layer_start < 0 || layer_start > layer_end) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "stage_index and layer range are invalid");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_slice_range range;
    range.stage_index = stage_index;
    range.layer_start = layer_start;
    range.layer_end = layer_end;
    range.include_embeddings = include_embeddings;
    range.include_output = include_output;
    plan->ranges.push_back(range);
    return skippy_success(out_error);
}

enum skippy_status skippy_write_slice_gguf(
        struct skippy_model_info * info,
        const struct skippy_slice_plan * plan,
        int32_t stage_index,
        const char * output_path,
        struct skippy_error ** out_error) {
    if (info == nullptr || info->ctx == nullptr || plan == nullptr || output_path == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "info, plan, and output_path are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const skippy_slice_range * range = nullptr;
    for (const skippy_slice_range & candidate : plan->ranges) {
        if (candidate.stage_index == stage_index) {
            range = &candidate;
            break;
        }
    }
    if (range == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "stage_index is not present in slice plan");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    std::vector<skippy_source_tensor> selected;
    for (const skippy_tensor_meta & tensor : info->tensors) {
        if (skippy_tensor_selected(tensor, *range)) {
            selected.push_back({ info, &tensor });
        }
    }
    gguf_context * out_ctx = gguf_init_empty();
    if (out_ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_ERROR, "failed to allocate output GGUF context");
        return SKIPPY_STATUS_ERROR;
    }

    gguf_set_kv(out_ctx, info->ctx);
    skippy_clear_split_metadata(out_ctx);
    gguf_set_val_u32(out_ctx, GGUF_KEY_GENERAL_ALIGNMENT, static_cast<uint32_t>(gguf_get_alignment(out_ctx)));
    gguf_set_val_i32(out_ctx, "skippy.slice.stage_index", stage_index);
    gguf_set_val_i32(out_ctx, "skippy.slice.layer_start", range->layer_start);
    gguf_set_val_i32(out_ctx, "skippy.slice.layer_end", range->layer_end);
    gguf_set_val_bool(out_ctx, "skippy.slice.include_embeddings", range->include_embeddings);
    gguf_set_val_bool(out_ctx, "skippy.slice.include_output", range->include_output);

    const size_t tensor_mem = ggml_tensor_overhead() * (selected.size() + 1) + 1024 * 1024;
    ggml_init_params ggml_params = {
        /*.mem_size   =*/ tensor_mem,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ggml_ctx = ggml_init(ggml_params);
    if (ggml_ctx == nullptr) {
        gguf_free(out_ctx);
        skippy_set_error(out_error, SKIPPY_STATUS_ERROR, "failed to allocate tensor metadata context");
        return SKIPPY_STATUS_ERROR;
    }

    for (const skippy_source_tensor & item : selected) {
        const enum skippy_status status = skippy_add_tensor_to_context(ggml_ctx, out_ctx, *item.tensor, out_error);
        if (status != SKIPPY_STATUS_OK) {
            ggml_free(ggml_ctx);
            gguf_free(out_ctx);
            return status;
        }
    }

    const enum skippy_status status = skippy_copy_source_tensors(selected, output_path, out_ctx, out_error);
    ggml_free(ggml_ctx);
    gguf_free(out_ctx);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_write_gguf_from_parts(
        const char * const * input_paths,
        size_t input_count,
        const char * output_path,
        struct skippy_error ** out_error) {
    if (input_paths == nullptr || input_count == 0 || output_path == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "input paths and output_path are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    std::vector<skippy_model_info *> sources;
    sources.reserve(input_count);
    for (size_t i = 0; i < input_count; ++i) {
        if (input_paths[i] == nullptr) {
            for (skippy_model_info * source : sources) {
                skippy_model_info_delete(source);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "input path is null");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        skippy_model_info * source = skippy_model_info_open_owned(input_paths[i]);
        if (source == nullptr) {
            for (skippy_model_info * opened : sources) {
                skippy_model_info_delete(opened);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "failed to open GGUF package part");
            return SKIPPY_STATUS_MODEL_ERROR;
        }
        sources.push_back(source);
    }

    std::vector<skippy_source_tensor> selected;
    std::set<std::string> names;
    for (const skippy_model_info * source : sources) {
        for (const skippy_tensor_meta & tensor : source->tensors) {
            if (names.insert(tensor.name).second) {
                selected.push_back({ source, &tensor });
            }
        }
    }

    gguf_context * out_ctx = gguf_init_empty();
    if (out_ctx == nullptr) {
        for (skippy_model_info * source : sources) {
            skippy_model_info_delete(source);
        }
        skippy_set_error(out_error, SKIPPY_STATUS_ERROR, "failed to allocate output GGUF context");
        return SKIPPY_STATUS_ERROR;
    }

    gguf_set_kv(out_ctx, sources.front()->ctx);
    skippy_clear_split_metadata(out_ctx);
    gguf_set_val_u32(out_ctx, GGUF_KEY_GENERAL_ALIGNMENT, static_cast<uint32_t>(gguf_get_alignment(out_ctx)));
    gguf_set_val_bool(out_ctx, "skippy.package.materialized", true);
    gguf_set_val_u32(out_ctx, "skippy.package.part_count", static_cast<uint32_t>(input_count));

    const size_t tensor_mem = ggml_tensor_overhead() * (selected.size() + 1) + 1024 * 1024;
    ggml_init_params ggml_params = {
        /*.mem_size   =*/ tensor_mem,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ggml_ctx = ggml_init(ggml_params);
    if (ggml_ctx == nullptr) {
        gguf_free(out_ctx);
        for (skippy_model_info * source : sources) {
            skippy_model_info_delete(source);
        }
        skippy_set_error(out_error, SKIPPY_STATUS_ERROR, "failed to allocate tensor metadata context");
        return SKIPPY_STATUS_ERROR;
    }

    for (const skippy_source_tensor & item : selected) {
        const enum skippy_status status = skippy_add_tensor_to_context(ggml_ctx, out_ctx, *item.tensor, out_error);
        if (status != SKIPPY_STATUS_OK) {
            ggml_free(ggml_ctx);
            gguf_free(out_ctx);
            for (skippy_model_info * source : sources) {
                skippy_model_info_delete(source);
            }
            return status;
        }
    }

    const enum skippy_status status = skippy_copy_source_tensors(selected, output_path, out_ctx, out_error);
    ggml_free(ggml_ctx);
    gguf_free(out_ctx);
    for (skippy_model_info * source : sources) {
        skippy_model_info_delete(source);
    }
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    return skippy_success(out_error);
}

}
