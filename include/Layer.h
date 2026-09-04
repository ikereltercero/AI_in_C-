#pragma once
#include <vector>
#include <random>
#include "Neuron.h"

class Layer {
private:
    std::vector<Neuron> neuronas;
    bool is_linear;

public:
    Layer(size_t nin, size_t nout, std::mt19937& gen, bool is_linear);
    std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& inputs);
    std::vector<ValuePtr> parameters();
};
