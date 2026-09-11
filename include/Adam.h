#pragma once
#include <vector>
#include "Tensor.h"

class Adam {
private:
    std::vector<Tensor*> parametros;
    
    // En lugar de un diccionario, guardamos arreglos planos del mismo tamaño que los tensores
    std::vector<std::vector<float>> m;
    std::vector<std::vector<float>> v;
    
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    std::size_t t; // Paso de tiempo actual

public:
    Adam(const std::vector<Tensor*>& params, float learning_rate = 0.001f);
    void step();
    void zero_grad();
};