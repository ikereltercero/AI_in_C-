#pragma once
#include "autograd.h"
#include <vector>
#include <unordered_map>

class Adam {
private:
    struct Momentos {
        float m = 0.0f;
        float v = 0.0f;
    };

    std::vector<ValuePtr> parametros;
    std::unordered_map<Value*, Momentos> estado; // Diccionario de memorias por cada peso
    
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    int t; // Paso de tiempo actual

public:
    Adam(const std::vector<ValuePtr>& params, float learning_rate = 0.01f);
    void step();
    void zero_grad();
};