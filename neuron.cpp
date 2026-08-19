#pragma once
#include <random>
#include "autograd.cpp"
#include "iostream"
#include "stdexcept"
#include "vector"
// La clase Neurona encapsula múltiples ValuePtr
class Neuron {
private:
    std::vector<ValuePtr> w; // Lista de pesos (uno por cada entrada)
    ValuePtr b;              // Un solo bias
    bool is_linear;


public:
    // Constructor: Recibe cuántas entradas (nin) va a tener esta neurona
     Neuron(size_t nin, std::mt19937& gen, bool is_linear) : is_linear(is_linear) {
        // Inicializador de números aleatorios entre -1.0 y 1.0
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        // 1. Creamos 'nin' pesos aleatorios
        w.reserve(nin); // Reservamos memoria para evitar reallocs 
        for (size_t i = 0; i < nin; ++i) {
            w.push_back(make_value(dis(gen)));
        }
        // 2. Creamos el bias aleatorio
        b = make_value(dis(gen));
    }

    std::vector<ValuePtr> parameters() {
        std::vector<ValuePtr> params;
        params.reserve(w.size() + 1);

        params.insert(params.end(), w.begin(), w.end());
        params.push_back(b);

        return params;
    }

    // =================================================================
    // TU MISIÓN EMPIEZA AQUÍ
    // =================================================================
    // En C++, 'operator()' nos permite usar el objeto como si fuera una función.
    // Es el equivalente exacto a 'def __call__(self, x):' en Python.
    // Recibe una lista de ValuePtr (x) que son las entradas.
    
    ValuePtr operator()(const std::vector<ValuePtr>& x) {
        
        // COMPROBACIÓN DE SEGURIDAD:
        if (x.size() != w.size()) {
            throw std::invalid_argument("El número de entradas no coincide con los pesos");
        }
        ValuePtr sum = b;
        for (size_t i = 0; i < w.size(); ++i){
            sum = sum + w[i]*x[i];
        }
        return (is_linear ? sum : relu(sum));


    }
};
