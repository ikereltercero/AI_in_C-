#include "../include/Neuron.h"
#include <stdexcept>

Neuron::Neuron(size_t nin, std::mt19937& gen, bool is_linear) : is_linear(is_linear) {
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    w.reserve(nin);
    for (size_t i = 0; i < nin; ++i) {
        w.push_back(make_value(dis(gen)));
    }
    b = make_value(dis(gen));
}

std::vector<ValuePtr> Neuron::parameters() {
    std::vector<ValuePtr> params;
    params.reserve(w.size() + 1);
    params.insert(params.end(), w.begin(), w.end());
    params.push_back(b);
    return params;
}

ValuePtr Neuron::operator()(const std::vector<ValuePtr>& x) {
    if (x.size() != w.size()) {
        throw std::invalid_argument("El número de entradas no coincide con los pesos");
    }
    ValuePtr sum = b;
    for (size_t i = 0; i < w.size(); ++i) {
        sum = sum + w[i] * x[i];
    }
    return (is_linear ? sum : relu(sum));
}