#ifndef SKIPPY_COMMON_H
#define SKIPPY_COMMON_H

#include "../llama.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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
#define SKIPPY_ABI_VERSION_PATCH 23

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
    SKIPPY_FEATURE_BACKEND_DEVICES        = 1 << 23,
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

struct skippy_error {
    enum skippy_status status;
    const char * message;
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

#ifdef __cplusplus
}
#endif

#endif
