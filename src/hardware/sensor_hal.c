#include "hardware/sensor_hal.h" // Busca directamente en las carpetas -I de include/
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

static bool g_is_initialized = false;
static uint32_t g_sample_counter = 0;

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
}

hal_status_t hal_init(void) {
    if (g_is_initialized) {
        return HAL_OK;
    }
    g_sample_counter = 0;
    g_is_initialized = true;
    return HAL_OK;
}

hal_status_t hal_read_sample(sensor_data_t *out_data) {
    if (!g_is_initialized || out_data == NULL) {
        return HAL_INVALID_PARAM;
    }

    out_data->timestamp_ns = get_time_ns();
    out_data->sample_id = ++g_sample_counter;

    for (int i = 0; i < 8; ++i) {
        out_data->data[i] = (float)(i + 1) * 1.25f + ((float)(rand() % 100) / 1000.0f);
    }

    return HAL_OK;
}

hal_status_t hal_close(void) {
    if (!g_is_initialized) {
        return HAL_ERROR;
    }
    g_sample_counter = 0;
    g_is_initialized = false;
    return HAL_OK;
}