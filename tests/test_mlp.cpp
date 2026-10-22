#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>
#include "ml/MLP.h"
#include "ml/Tensor.h"

TEST(MLPTest, InferenceProducesCorrectDimensionsAndValidValues) {
    std::mt19937 gen(42); // Semilla fija para determinismo
    MLP model({4, 8, 3}, gen);

    std::vector<float> input_data = {5.1f, 3.5f, 1.4f, 0.2f};
    BatchView batch = {input_data.data(), nullptr, 1, 4, 3};

    Tensor output = model.inference(batch);

    // Verificación de dimensiones
    EXPECT_EQ(output.rows, 1u);
    EXPECT_EQ(output.cols, 3u);

    // Verificación de estabilidad numérica
    for (size_t i = 0; i < output.data.size(); ++i) {
        EXPECT_FALSE(std::isnan(output.data[i])) << "NaN detectado en el índice " << i;
        EXPECT_FALSE(std::isinf(output.data[i])) << "Inf detectado en el índice " << i;
    }
}

TEST(MLPTest, BatchInferenceScalesCorrectly) {
    std::mt19937 gen(123);
    MLP model({4, 16, 3}, gen);

    // Batch de 2 muestras
    std::vector<float> input_batch = {
        5.1f, 3.5f, 1.4f, 0.2f,
        6.7f, 3.0f, 5.2f, 2.3f
    };
    BatchView batch = {input_batch.data(), nullptr, 2, 4, 3};

    Tensor output = model.inference(batch);

    EXPECT_EQ(output.rows, 2u);
    EXPECT_EQ(output.cols, 3u);
    EXPECT_EQ(output.data.size(), 6u);
}
