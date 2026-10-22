
#include <gtest/gtest.h>
#include <vector>

#include "ml/Adam.h"
#include "ml/Tensor.h"

TEST(AdamTest, UpdatesTensorParameters) {
    Tensor t(1, 1);
    t.data[0] = 1.0f;
    t.grad[0] = 2.0f;

    std::vector<Tensor*> params = {&t};
    Adam adam(params, 0.1f);

    adam.step();

    // Con el primer paso, el parámetro debe disminuir.
    EXPECT_NEAR(t.data[0], 0.9f, 1e-5f);
}

TEST(AdamTest, ZeroGradResetsTensorGradients) {
    Tensor t(1, 2);
    t.grad = {1.0f, -2.0f};

    std::vector<Tensor*> params = {&t};
    Adam adam(params, 0.01f);

    adam.zero_grad();

    EXPECT_FLOAT_EQ(t.grad[0], 0.0f);
    EXPECT_FLOAT_EQ(t.grad[1], 0.0f);
}

TEST(AdamTest, RepeatedStepsUpdateParameters) {
    Tensor t(1, 1);
    t.data[0] = 1.0f;

    std::vector<Tensor*> params = {&t};
    Adam adam(params, 0.01f);

    t.grad[0] = 1.0f;
    adam.step();
    const float first_value = t.data[0];

    t.grad[0] = 1.0f;
    adam.step();

    EXPECT_LT(first_value, 1.0f);
    EXPECT_LT(t.data[0], first_value);
}
