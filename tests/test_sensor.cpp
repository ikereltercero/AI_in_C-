#include <gtest/gtest.h>
#include <cmath>
#include <array>
#include "hardware/sensor_hal.h"

class SensorHALTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(hal_init(), HAL_OK);
    }

    void TearDown() override {
        hal_close();
    }
};

TEST_F(SensorHALTest, ReadsValidSampleData) {
    sensor_data_t sample{};
    ASSERT_EQ(hal_read_sample(&sample), HAL_OK);

    EXPECT_GT(sample.timestamp_ns, 0u);
    EXPECT_GE(sample.sample_id, 1u);

    // std::size obtiene el tamaño exacto del array en tiempo de compilación
    constexpr size_t num_features = sizeof(sample.data) / sizeof(sample.data[0]);
    for (size_t i = 0; i < num_features; ++i) {
        EXPECT_FALSE(std::isnan(sample.data[i]));
        EXPECT_FALSE(std::isinf(sample.data[i]));
    }
}

TEST_F(SensorHALTest, TimestampsAreMonotonicallyIncreasing) {
    sensor_data_t sample1{};
    sensor_data_t sample2{};

    ASSERT_EQ(hal_read_sample(&sample1), HAL_OK);
    ASSERT_EQ(hal_read_sample(&sample2), HAL_OK);

    EXPECT_GT(sample2.timestamp_ns, sample1.timestamp_ns);
    EXPECT_GT(sample2.sample_id, sample1.sample_id);
}