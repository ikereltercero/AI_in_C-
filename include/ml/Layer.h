#pragma once
#include "Tensor.h"
#include <random>
#include <vector>

class Layer {
public:
    Tensor W; // Matriz de pesos [entradas, salidas]
    Tensor b; // Matriz de sesgos [1, salidas]
    bool is_linear;

    // Cachés de memoria para el Backward Pass
    Tensor X_cache;   
    Tensor out_cache; 

    Layer(size_t nin, size_t nout, std::mt19937& gen, bool is_linear);

    // Forward pasará un batch entero de golpe
    Tensor forward(const Tensor& X);

    // Backward recibe el gradiente de la capa siguiente y devuelve el suyo
    Tensor backward(const Tensor& dOut);

    Tensor forward_inference(const Tensor& X) const;
};
