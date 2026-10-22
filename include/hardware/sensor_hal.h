#ifndef SENSOR_HAL_H
#define SENSOR_HAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HAL_OK              =  0,
    HAL_ERROR           = -1,
    HAL_BUSY            = -2,
    HAL_INVALID_PARAM   = -3
} hal_status_t;

typedef struct {
    uint64_t timestamp_ns;
    float    data[8];       
    uint32_t sample_id;
} sensor_data_t;

hal_status_t hal_init(void);
hal_status_t hal_read_sample(sensor_data_t *out_data);
hal_status_t hal_close(void);

#ifdef __cplusplus
}
#endif

#endif /* SENSOR_HAL_H */