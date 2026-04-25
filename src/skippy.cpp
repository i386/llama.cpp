#include "skippy.h"

#include "gguf.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <regex>
#include <string>

struct skippy_model {
    llama_model * model = nullptr;
    skippy_runtime_config config = {};
};

struct skippy_session {
    skippy_model * stage_model = nullptr;
    llama_context * ctx = nullptr;
    int32_t n_past = 0;
};

struct skippy_model_info {
    gguf_context * ctx = nullptr;
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

static bool skippy_is_full_model_config(const struct skippy_runtime_config * config) {
    if (config == nullptr) {
        return true;
    }

    return !config->filter_tensors_on_load && config->layer_start == 0;
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
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(token_ids),
            static_cast<int32_t>(token_count));

    const int32_t rc = llama_decode(session->ctx, batch);
    if (rc != 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama_decode failed");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    session->n_past += static_cast<int32_t>(token_count);
    return skippy_success(out_error);
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

extern "C" {

struct skippy_abi_version skippy_abi_version(void) {
    return {
        SKIPPY_ABI_VERSION_MAJOR,
        SKIPPY_ABI_VERSION_MINOR,
        SKIPPY_ABI_VERSION_PATCH,
    };
}

uint64_t skippy_abi_features(void) {
    return SKIPPY_FEATURE_MODEL_INTROSPECTION |
           SKIPPY_FEATURE_TOKENIZE_DETOKENIZE;
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

    if (!skippy_is_full_model_config(config)) {
        skippy_set_error(
                out_error,
                SKIPPY_STATUS_UNSUPPORTED,
                "runtime tensor filtering is not implemented yet; use a full-model single-stage config");
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    llama_model_params params = llama_model_default_params();
    if (config != nullptr) {
        params.n_gpu_layers = config->n_gpu_layers;
    }

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(path, params);
    if (model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "failed to load llama model");
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    skippy_model * stage_model = new skippy_model{};
    stage_model->model = model;
    if (config != nullptr) {
        stage_model->config = *config;
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

enum skippy_status skippy_model_info_open(
        const char * path,
        struct skippy_model_info ** out_info,
        struct skippy_error ** out_error) {
    if (path == nullptr || out_info == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "path and out_info are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    *out_info = nullptr;

    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * ctx = gguf_init_from_file(path, params);
    if (ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "failed to open GGUF model metadata");
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    skippy_model_info * info = new skippy_model_info{};
    info->ctx = ctx;
    *out_info = info;
    return skippy_success(out_error);
}

enum skippy_status skippy_model_info_free(
        struct skippy_model_info * info,
        struct skippy_error ** out_error) {
    if (info != nullptr) {
        gguf_free(info->ctx);
        delete info;
    }
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

}
