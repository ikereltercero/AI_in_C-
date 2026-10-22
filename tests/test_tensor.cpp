#include <gtest/gtest.h>
#include "ml/Tensor.h"

TEST(TensorTest, InitializesDataAndGradientsToZero) {
    Tensor t(2, 3);
    ASSERT_EQ(t.size(), 6u);
    for (float value : t.data) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
    for (float value : t.grad) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}

TEST(TensorTest, AccessesElementsCorrectly) {
    Tensor t(2, 2);
    t.at(0, 0) = 1.0f;
    t.at(0, 1) = 2.0f;
    t.at(1, 0) = 3.0f;
    t.at(1, 1) = 4.0f;

    EXPECT_FLOAT_EQ(t.data[0], 1.0f);
    EXPECT_FLOAT_EQ(t.at(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(t.at(1, 1), 4.0f);
}

TEST(TensorTest, ZeroGradResetsAllGradients) {
    Tensor t(2, 2);
    t.grad = {1.0f, 2.0f, 3.0f, 4.0f};
    t.zero_grad();

    for (float value : t.grad) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}
