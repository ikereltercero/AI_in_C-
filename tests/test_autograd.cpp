#include <gtest/gTest.h>
#include "autograd.h" // Incluimos tu motor matemático

// Prueba 1: Verificar que la suma y la propagación de gradientes funcionan
TEST(AutogradTest, BasicAdditionAndBackward) {
    // Creamos dos valores de prueba
    ValuePtr a = make_value(2.0f);
    ValuePtr b = make_value(3.0f);

    // Operación: c = a + b
    ValuePtr c = a + b;

    // Comprobamos que el resultado hacia adelante es correcto (2 + 3 = 5)
    EXPECT_EQ(c->data, 5.0f);

    // Simulamos el backward propagando un gradiente de 1.0
    c->grad = 1.0f;
    backward(c);

    // El gradiente debería repartirse equitativamente a ambos lados (dc/da = 1, dc/db = 1)
    EXPECT_EQ(a->grad, 1.0f);
    EXPECT_EQ(b->grad, 1.0f);
}
