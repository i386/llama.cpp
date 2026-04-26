#include "skippy.h"

#include "gguf.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <cstdio>
#include <limits>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct skippy_model {
    llama_model * model = nullptr;
    skippy_runtime_config config = {};
    bool executable = true;
};

struct skippy_session {
    skippy_model * stage_model = nullptr;
    llama_context * ctx = nullptr;
    int32_t n_past = 0;
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

static enum skippy_status skippy_success(skippy_error ** out_error) {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    return SKIPPY_STATUS_OK;
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

struct skippy_filter_scope {
    explicit skippy_filter_scope(const skippy_runtime_config * config) {
        if (config != nullptr && config->filter_tensors_on_load) {
            llama_model_loader_stage_filter filter;
            filter.enabled = true;
            filter.layer_start = config->layer_start;
            filter.layer_end = config->layer_end;
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

static bool skippy_is_filtered(const skippy_session * session) {
    return session != nullptr &&
           session->stage_model != nullptr &&
           session->stage_model->config.filter_tensors_on_load;
}

static bool skippy_emits_activation_frame(const skippy_session * session) {
    return skippy_is_filtered(session) && !session->stage_model->config.include_output;
}

static size_t skippy_activation_payload_bytes(const skippy_session * session, size_t token_count) {
    if (session == nullptr || session->stage_model == nullptr || session->stage_model->model == nullptr) {
        return 0;
    }

    return token_count *
           static_cast<size_t>(llama_model_n_embd(session->stage_model->model)) *
           sizeof(float);
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

    const size_t expected_bytes = skippy_activation_payload_bytes(session, expected_token_count);
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
        struct skippy_error ** out_error) {
    const size_t payload_bytes = skippy_emits_activation_frame(session) ?
            skippy_activation_payload_bytes(session, token_count) : 0;

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
        output_desc->flags = 0;
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

    llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(token_ids),
            static_cast<int32_t>(token_count));

    return skippy_decode_batch(session, batch, token_count, out_error);
}

static llama_token skippy_greedy_sample(skippy_session * session) {
    const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(session->ctx, -1);

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
        struct skippy_error ** out_error) {
    if (!skippy_emits_activation_frame(session)) {
        return skippy_success(out_error);
    }

    float * embeddings = llama_get_embeddings(session->ctx);
    if (embeddings == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama embeddings output was not available");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    const size_t payload_bytes = skippy_activation_payload_bytes(session, token_count);
    std::memcpy(output_payload, embeddings, payload_bytes);
    return skippy_success(out_error);
}

static enum skippy_status skippy_decode_activation_frame(
        skippy_session * session,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
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
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    batch.n_tokens = n_tokens;
    std::memcpy(batch.embd, input_payload, static_cast<size_t>(input_desc->payload_bytes));

    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i] = session->n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = request_logits && i == n_tokens - 1 ? 1 : 0;
    }

    enum skippy_status status = skippy_decode_batch(session, batch, token_count, out_error);
    llama_batch_free(batch);
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
           SKIPPY_FEATURE_TOKENIZE_DETOKENIZE |
           SKIPPY_FEATURE_ACTIVATION_FRAME |
           SKIPPY_FEATURE_NATIVE_KV_PAGE |
           SKIPPY_FEATURE_SESSION_RESET;
}

const char * skippy_status_string(enum skippy_status status) {
    switch (status) {
        case SKIPPY_STATUS_OK:               return "ok";
        case SKIPPY_STATUS_ERROR:            return "error";
        case SKIPPY_STATUS_INVALID_ARGUMENT: return "invalid_argument";
        case SKIPPY_STATUS_UNSUPPORTED:      return "unsupported";
        case SKIPPY_STATUS_BUFFER_TOO_SMALL: return "buffer_too_small";
        case SKIPPY_STATUS_IO_ERROR:         return "io_error";
        case SKIPPY_STATUS_MODEL_ERROR:      return "model_error";
        case SKIPPY_STATUS_RUNTIME_ERROR:    return "runtime_error";
    }

    return "unknown";
}

void skippy_error_free(struct skippy_error * error) {
    if (error == nullptr) {
        return;
    }

    std::free(const_cast<char *>(error->message));
    delete error;
}

enum skippy_status skippy_model_open(
        const char * path,
        const struct skippy_runtime_config * config,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    if (path == nullptr || out_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "path and out_model are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    *out_model = nullptr;

    if (config != nullptr && config->filter_tensors_on_load && (config->layer_start < 0 || config->layer_start >= config->layer_end)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "layer_start must be non-negative and less than layer_end");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    if (!skippy_is_full_model_config(config) && (config == nullptr || !config->filter_tensors_on_load)) {
        skippy_set_error(
                out_error,
                SKIPPY_STATUS_UNSUPPORTED,
                "runtime tensor filtering is not implemented yet; use a full-model single-stage config");
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    llama_model_params params = llama_model_default_params();
    if (config != nullptr) {
        params.n_gpu_layers = config->n_gpu_layers;
        if (config->disable_repack || config->filter_tensors_on_load) {
            params.use_extra_bufts = false;
        }
    }

    llama_backend_init();
    skippy_filter_scope filter_scope(config);
    llama_model * model = llama_model_load_from_file(path, params);
    if (model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "failed to load llama model");
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    if (config != nullptr && config->filter_tensors_on_load) {
        const int32_t n_layer = llama_model_n_layer(model);
        if (model->arch != LLM_ARCH_LLAMA && model->arch != LLM_ARCH_QWEN35MOE) {
            llama_model_free(model);
            skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "runtime-slice execution is currently supported for LLaMA-family and Qwen35MoE graphs only");
            return SKIPPY_STATUS_UNSUPPORTED;
        }
        if (config->layer_end > n_layer) {
            llama_model_free(model);
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "layer_end exceeds model layer count");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (config->include_embeddings && config->layer_start != 0) {
            llama_model_free(model);
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "only the first runtime slice may include token embeddings");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (config->layer_start == 0 && !config->include_embeddings) {
            llama_model_free(model);
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "the first runtime slice must include token embeddings");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (config->include_output && config->layer_end != n_layer) {
            llama_model_free(model);
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "only the final runtime slice may include output tensors");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
    }

    skippy_model * stage_model = new skippy_model{};
    stage_model->model = model;
    if (config != nullptr) {
        stage_model->config = *config;
        stage_model->executable = true;
    }

    *out_model = stage_model;
    return skippy_success(out_error);
}

enum skippy_status skippy_model_free(
        struct skippy_model * model,
        struct skippy_error ** out_error) {
    if (model != nullptr) {
        llama_model_free(model->model);
        delete model;
    }
    return skippy_success(out_error);
}

enum skippy_status skippy_session_create(
        struct skippy_model * model,
        struct skippy_session ** out_session,
        struct skippy_error ** out_error) {
    if (model == nullptr || model->model == nullptr || out_session == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model and out_session are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    *out_session = nullptr;

    llama_context_params params = llama_context_default_params();
    params.n_ctx = model->config.ctx_size > 0 ? static_cast<uint32_t>(model->config.ctx_size) : 512;
    params.n_batch = params.n_ctx;
    params.embeddings = model->config.filter_tensors_on_load && !model->config.include_output;

    skippy_graph_filter_scope graph_filter_scope(&model->config);
    llama_context * ctx = llama_init_from_model(model->model, params);
    if (ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to create llama context");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    skippy_session * session = new skippy_session{};
    session->stage_model = model;
    session->ctx = ctx;
    session->n_past = 0;
    *out_session = session;
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

    llama_memory_clear(memory, true);
    session->n_past = 0;
    session->ctx->synchronize();
    return skippy_success(out_error);
}

enum skippy_status skippy_session_free(
        struct skippy_session * session,
        struct skippy_error ** out_error) {
    if (session != nullptr) {
        llama_free(session->ctx);
        delete session;
    }
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

enum skippy_status skippy_decode_step(
        struct skippy_session * session,
        llama_token token_id,
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
        *out_predicted_token = skippy_greedy_sample(session);
    }

    return skippy_success(out_error);
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
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        status = skippy_decode_activation_frame(session, input_desc, input_payload, token_count, false, out_error);
    } else {
        status = skippy_decode_tokens(session, token_ids, token_count, false, out_error);
    }
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    return skippy_copy_output_activation_frame(session, token_count, output_payload, out_error);
}

enum skippy_status skippy_decode_step_frame(
        struct skippy_session * session,
        llama_token token_id,
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
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        status = skippy_decode_activation_frame(session, input_desc, input_payload, 1, true, out_error);
    } else {
        status = skippy_decode_tokens(session, &token_id, 1, true, out_error);
    }
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (out_predicted_token != nullptr) {
        *out_predicted_token = session->stage_model->config.include_output ? skippy_greedy_sample(session) : -1;
    }

    return skippy_copy_output_activation_frame(session, 1, output_payload, out_error);
}

enum skippy_status skippy_export_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error) {
    (void) session;
    (void) layer_start;
    (void) layer_end;
    (void) output;
    (void) output_capacity;
    (void) out_bytes;
    skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "state export is not implemented yet");
    return SKIPPY_STATUS_UNSUPPORTED;
}

enum skippy_status skippy_import_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error) {
    (void) session;
    (void) layer_start;
    (void) layer_end;
    (void) input;
    (void) input_bytes;
    skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "state import is not implemented yet");
    return SKIPPY_STATUS_UNSUPPORTED;
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
    if (!kv->stage_import_kv_page(*desc, input, input_bytes, error)) {
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
    const int32_t result = llama_detokenize(
            vocab,
            tokens,
            static_cast<int32_t>(token_count),
            output_text,
            static_cast<int32_t>(output_text_capacity),
            true,
            false);
    if (result < 0) {
        *out_text_bytes = static_cast<size_t>(-result);
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "text output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }

    *out_text_bytes = static_cast<size_t>(result);
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
    out_tensor->element_count = 0;
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
