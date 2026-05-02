#ifndef SKIPPY_SIGNALS_ABI_H
#define SKIPPY_SIGNALS_ABI_H

#include "skippy.h"

#ifdef __cplusplus
extern "C" {
#endif

struct skippy_token_signal {
    float entropy;
    float top_logprob;
    float second_logprob;
    float margin;
    int32_t top_token;
    int32_t second_token;
};

struct skippy_generation_signal_window {
    uint32_t token_count;
    float mean_entropy;
    float max_entropy;
    float mean_margin;
    float min_margin;
    uint32_t high_entropy_count;
    uint32_t repetition_count;
};

LLAMA_API enum skippy_status skippy_session_last_token_signal(
        struct skippy_session * session,
        struct skippy_token_signal * out_signal,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_session_signal_window(
        struct skippy_session * session,
        uint32_t window_tokens,
        struct skippy_generation_signal_window * out_window,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_SIGNALS_ABI_H
