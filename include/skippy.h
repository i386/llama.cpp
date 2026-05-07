#ifndef SKIPPY_ABI_H
#define SKIPPY_ABI_H

#include "llama.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Experimental ABI for layer-range staged inference.
//
// This header is intentionally a capability boundary. It exposes opaque handles
// and plain C structs so Rust orchestration code can use llama.cpp model
// execution, tokenization, GGUF metadata, and GGUF writing without depending on
// private C++ layouts.
//
// Stability note:
// - ABI version 0 is experimental.
// - Callers must feature-probe before use.
// - Names, structs, and ownership rules may change until cross-family staged
//   runtime validation has completed.

#define SKIPPY_ABI_VERSION_MAJOR 0
#define SKIPPY_ABI_VERSION_MINOR 1
#define SKIPPY_ABI_VERSION_PATCH 22

#define SKIPPY_MAX_LOGIT_BIAS 256

enum skippy_feature {
    SKIPPY_FEATURE_RUNTIME_SLICE          = 1 << 0,
    SKIPPY_FEATURE_LAYER_PACKAGE          = 1 << 1,
    SKIPPY_FEATURE_ARTIFACT_SLICE         = 1 << 2,
    SKIPPY_FEATURE_MODEL_INTROSPECTION    = 1 << 3,
    SKIPPY_FEATURE_GGUF_SLICE_WRITE       = 1 << 4,
    SKIPPY_FEATURE_STATE_IMPORT_EXPORT    = 1 << 5,
    SKIPPY_FEATURE_TOKENIZE_DETOKENIZE    = 1 << 6,
    SKIPPY_FEATURE_ACTIVATION_FRAME       = 1 << 7,
    SKIPPY_FEATURE_NATIVE_KV_PAGE         = 1 << 8,
    SKIPPY_FEATURE_SESSION_RESET          = 1 << 9,
    SKIPPY_FEATURE_BATCH_VERIFY           = 1 << 10,
    SKIPPY_FEATURE_CHAT_TEMPLATE          = 1 << 11,
    SKIPPY_FEATURE_SAMPLING_CONFIG        = 1 << 12,
    SKIPPY_FEATURE_BATCH_VERIFY_FRAME     = 1 << 13,
    SKIPPY_FEATURE_RECURRENT_STATE        = 1 << 14,
    SKIPPY_FEATURE_LOGIT_BIAS             = 1 << 15,
    SKIPPY_FEATURE_SESSION_TRIM           = 1 << 16,
    SKIPPY_FEATURE_SESSION_CHECKPOINT     = 1 << 17,
    SKIPPY_FEATURE_PACKAGE_PART_LOAD      = 1 << 18,
    SKIPPY_FEATURE_GENERATION_SIGNALS     = 1 << 19,
    SKIPPY_FEATURE_EXTERNAL_MEDIA_PREFILL = 1 << 20,
    SKIPPY_FEATURE_CHAT_TEMPLATE_TOOLS    = 1 << 21,
    SKIPPY_FEATURE_CHAT_SAMPLING_GRAMMAR  = 1 << 22,
};

enum skippy_kv_page_flag {
    SKIPPY_KV_PAGE_FLAG_V_TRANSPOSED      = 1 << 0,
};

enum skippy_status {
    SKIPPY_STATUS_OK                      = 0,
    SKIPPY_STATUS_ERROR                   = 1,
    SKIPPY_STATUS_INVALID_ARGUMENT        = 2,
    SKIPPY_STATUS_UNSUPPORTED             = 3,
    SKIPPY_STATUS_BUFFER_TOO_SMALL        = 4,
    SKIPPY_STATUS_IO_ERROR                = 5,
    SKIPPY_STATUS_MODEL_ERROR             = 6,
    SKIPPY_STATUS_RUNTIME_ERROR           = 7,
};

enum skippy_load_mode {
    SKIPPY_LOAD_MODE_RUNTIME_SLICE        = 0,
    SKIPPY_LOAD_MODE_LAYER_PACKAGE        = 1,
    SKIPPY_LOAD_MODE_ARTIFACT_SLICE       = 2,
};

enum skippy_tensor_role {
    SKIPPY_TENSOR_ROLE_UNKNOWN            = 0,
    SKIPPY_TENSOR_ROLE_METADATA           = 1,
    SKIPPY_TENSOR_ROLE_TOKENIZER          = 2,
    SKIPPY_TENSOR_ROLE_EMBEDDING          = 3,
    SKIPPY_TENSOR_ROLE_LAYER              = 4,
    SKIPPY_TENSOR_ROLE_FINAL_NORM         = 5,
    SKIPPY_TENSOR_ROLE_OUTPUT             = 6,
};

enum skippy_activation_dtype {
    SKIPPY_ACTIVATION_DTYPE_UNKNOWN       = 0,
    SKIPPY_ACTIVATION_DTYPE_F32           = 1,
    SKIPPY_ACTIVATION_DTYPE_F16           = 2,
    SKIPPY_ACTIVATION_DTYPE_BF16          = 3,
};

enum skippy_activation_layout {
    SKIPPY_ACTIVATION_LAYOUT_OPAQUE       = 0,
    SKIPPY_ACTIVATION_LAYOUT_TOKEN_MAJOR  = 1,
};

#define SKIPPY_ACTIVATION_FLAG_RWKV7_V_FIRST (UINT64_C(1) << 0)

struct skippy_model;
struct skippy_session;
struct skippy_model_info;
struct skippy_slice_plan;

struct skippy_error {
    enum skippy_status status;
    const char * message;
};

struct skippy_runtime_config {
    int32_t stage_index;
    int32_t layer_start;
    int32_t layer_end;
    int32_t ctx_size;
    int32_t lane_count;
    int32_t n_batch;
    int32_t n_ubatch;
    int32_t n_threads;
    int32_t n_threads_batch;
    int32_t n_gpu_layers;
    int32_t cache_type_k;
    int32_t cache_type_v;
    int32_t flash_attn_type;

    enum skippy_load_mode load_mode;

    bool disable_repack;
    bool filter_tensors_on_load;
    bool include_embeddings;
    bool include_output;

    // Optional ggml backend device name, for example "CUDA0", "MTL0",
    // "Vulkan1", or "CPU". When set, skippy loads the model only on that
    // backend device instead of relying on llama.cpp default device ordering.
    const char * selected_backend_device;
};

struct skippy_tensor_info {
    const char * name;
    int32_t layer_index;
    enum skippy_tensor_role role;
    uint32_t ggml_type;
    uint64_t byte_size;
    uint64_t element_count;
};

struct skippy_activation_desc {
    uint32_t version;
    enum skippy_activation_dtype dtype;
    enum skippy_activation_layout layout;
    int32_t producer_stage_index;
    int32_t layer_start;
    int32_t layer_end;
    uint32_t token_count;
    uint32_t sequence_count;
    uint64_t payload_bytes;
    uint64_t flags;
};

struct skippy_sampling_config {
    uint32_t version;
    uint32_t flags;
    uint32_t seed;
    int32_t top_k;
    int32_t penalty_last_n;
    float temperature;
    float top_p;
    float presence_penalty;
    float frequency_penalty;
    float repeat_penalty;
    uint32_t logit_bias_count;
    uint32_t reserved;
    llama_logit_bias logit_bias[SKIPPY_MAX_LOGIT_BIAS];
};

struct skippy_kv_page_desc {
    uint32_t version;
    int32_t layer_start;
    int32_t layer_end;
    uint64_t token_start;
    uint64_t token_count;
    uint32_t layer_count;
    uint32_t k_type;
    uint32_t v_type;
    uint32_t k_row_bytes;
    uint32_t v_row_bytes;
    uint32_t v_element_bytes;
    uint64_t payload_bytes;
    uint64_t flags;
};

struct skippy_abi_version {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};

LLAMA_API struct skippy_abi_version skippy_abi_version(void);

LLAMA_API uint64_t skippy_abi_features(void);

LLAMA_API const char * skippy_status_string(enum skippy_status status);

LLAMA_API void skippy_error_free(struct skippy_error * error);

LLAMA_API enum skippy_status skippy_model_open(
        const char * path,
        const struct skippy_runtime_config * config,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_model_open_from_parts(
        const char * const * paths,
        size_t path_count,
        const struct skippy_runtime_config * config,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_model_free(
        struct skippy_model * model,
        struct skippy_error ** out_error);

LLAMA_API const struct llama_model * skippy_model_llama_model(
        const struct skippy_model * model);

LLAMA_API enum skippy_status skippy_session_create(
        struct skippy_model * model,
        struct skippy_session ** out_session,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_session_create_from_resident_prefix(
        struct skippy_model * model,
        int32_t cache_seq_id,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_session ** out_session,
        struct skippy_error ** out_error);

LLAMA_API struct llama_context * skippy_session_llama_context(
        struct skippy_session * session);

LLAMA_API int32_t skippy_session_position(
        const struct skippy_session * session);

LLAMA_API int32_t skippy_session_native_seq_id(
        const struct skippy_session * session);

LLAMA_API int32_t skippy_session_batch_size(
        const struct skippy_session * session);

LLAMA_API enum skippy_status skippy_session_begin_external_decode(
        struct skippy_session * session,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_session_end_external_decode(
        struct skippy_session * session,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_session_set_position(
        struct skippy_session * session,
        int32_t n_past,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_session_sample_current(
        struct skippy_session * session,
        const struct skippy_sampling_config * sampling,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_session_configure_chat_sampling(
        struct skippy_session * session,
        const struct skippy_sampling_config * sampling,
        const char * metadata_json,
        uint64_t prompt_token_count,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_session_reset(
        struct skippy_session * session,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_checkpoint_session(
        struct skippy_session * session,
        uint64_t * out_token_count,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_restore_session_checkpoint(
        struct skippy_session * session,
        uint64_t token_count,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_session_free(
        struct skippy_session * session,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_prefill_chunk(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const void * input_activations,
        size_t input_activation_bytes,
        void * output_activations,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_decode_step(
        struct skippy_session * session,
        llama_token token_id,
        const void * input_activation,
        size_t input_activation_bytes,
        void * output_activation,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_verify_tokens(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_decode_step_sampled(
        struct skippy_session * session,
        llama_token token_id,
        const struct skippy_sampling_config * sampling,
        const void * input_activation,
        size_t input_activation_bytes,
        void * output_activation,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_prefill_chunk_frame(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_prefill_chunk_frame_sampled(
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
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_prefill_chunk_frame_with_positions(
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
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_prefill_chunk_frame_sampled_with_positions(
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
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_decode_step_frame(
        struct skippy_session * session,
        llama_token token_id,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_decode_step_frame_sampled(
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
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_verify_tokens_frame(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_session_copy_output_activation_frame(
        struct skippy_session * session,
        size_t token_count,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_export_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_import_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_export_full_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_export_recurrent_state(
        struct skippy_session * session,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_import_full_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_import_recurrent_state(
        struct skippy_session * session,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_trim_session(
        struct skippy_session * session,
        uint64_t token_count,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_export_kv_page(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        uint64_t token_start,
        uint64_t token_count,
        struct skippy_kv_page_desc * out_desc,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_import_kv_page(
        struct skippy_session * session,
        const struct skippy_kv_page_desc * desc,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_session_save_prefix(
        struct skippy_session * session,
        int32_t cache_seq_id,
        uint64_t token_count,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_session_restore_prefix(
        struct skippy_session * session,
        int32_t cache_seq_id,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_session_drop_sequence(
        struct skippy_session * session,
        int32_t seq_id,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_tokenize(
        struct skippy_model * model,
        const char * text,
        bool add_special,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_detokenize(
        struct skippy_model * model,
        const llama_token * tokens,
        size_t token_count,
        char * output_text,
        size_t output_text_capacity,
        size_t * out_text_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_token_is_eog(
        struct skippy_model * model,
        llama_token token,
        bool * out_is_eog,
        struct skippy_error ** out_error);

LLAMA_API const struct llama_model * skippy_model_native_model(
        const struct skippy_model * model);

LLAMA_API enum skippy_status skippy_apply_chat_template(
        struct skippy_model * model,
        const struct llama_chat_message * messages,
        size_t message_count,
        bool add_assistant,
        bool override_enable_thinking,
        bool enable_thinking,
        char * output_text,
        size_t output_text_capacity,
        size_t * out_text_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_apply_chat_template_json(
        struct skippy_model * model,
        const char * messages_json,
        const char * tools_json,
        const char * tool_choice_json,
        bool add_assistant,
        bool override_enable_thinking,
        bool enable_thinking,
        bool parallel_tool_calls,
        char * output_text,
        size_t output_text_capacity,
        size_t * out_text_bytes,
        char * output_metadata_json,
        size_t output_metadata_json_capacity,
        size_t * out_metadata_json_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_parse_chat_response_json(
        const char * generated_text,
        const char * metadata_json,
        bool is_partial,
        char * output_message_json,
        size_t output_message_json_capacity,
        size_t * out_message_json_bytes,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_model_info_open(
        const char * path,
        struct skippy_model_info ** out_info,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_model_info_free(
        struct skippy_model_info * info,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_model_info_tensor_count(
        struct skippy_model_info * info,
        size_t * out_count,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_model_info_tensor_at(
        struct skippy_model_info * info,
        size_t index,
        struct skippy_tensor_info * out_tensor,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_slice_plan_create(
        struct skippy_model_info * info,
        struct skippy_slice_plan ** out_plan,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_slice_plan_free(
        struct skippy_slice_plan * plan,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_slice_plan_add_layer_range(
        struct skippy_slice_plan * plan,
        int32_t stage_index,
        int32_t layer_start,
        int32_t layer_end,
        bool include_embeddings,
        bool include_output,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_write_slice_gguf(
        struct skippy_model_info * info,
        const struct skippy_slice_plan * plan,
        int32_t stage_index,
        const char * output_path,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_write_gguf_from_parts(
        const char * const * input_paths,
        size_t input_count,
        const char * output_path,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_ABI_H
