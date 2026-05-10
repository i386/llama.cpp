#ifndef SKIPPY_DEVICES_H
#define SKIPPY_DEVICES_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

enum skippy_backend_device_type {
    SKIPPY_BACKEND_DEVICE_TYPE_CPU         = 0,
    SKIPPY_BACKEND_DEVICE_TYPE_GPU         = 1,
    SKIPPY_BACKEND_DEVICE_TYPE_IGPU        = 2,
    SKIPPY_BACKEND_DEVICE_TYPE_ACCEL       = 3,
    SKIPPY_BACKEND_DEVICE_TYPE_META        = 4,
};

enum skippy_backend_device_cap {
    SKIPPY_BACKEND_DEVICE_CAP_ASYNC                 = UINT64_C(1) << 0,
    SKIPPY_BACKEND_DEVICE_CAP_HOST_BUFFER           = UINT64_C(1) << 1,
    SKIPPY_BACKEND_DEVICE_CAP_BUFFER_FROM_HOST_PTR  = UINT64_C(1) << 2,
    SKIPPY_BACKEND_DEVICE_CAP_EVENTS                = UINT64_C(1) << 3,
};

struct skippy_backend_device {
    uint32_t version;
    const char * name;
    const char * description;
    const char * device_id;
    uint64_t memory_free;
    uint64_t memory_total;
    enum skippy_backend_device_type type;
    uint64_t caps;
};

LLAMA_API enum skippy_status skippy_backend_device_count(
        size_t * out_count,
        struct skippy_error ** out_error);

LLAMA_API enum skippy_status skippy_backend_device_at(
        size_t index,
        struct skippy_backend_device * out_device,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif
