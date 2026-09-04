#pragma once
#include <vector>
#include <random>
#include "autograd.h"

class Neuron {
private:
    std::vector<ValuePtr> w;
    ValuePtr b;
    bool is_linear;

public:
    Neuron(size_t nin, std::mt19937& gen, bool is_linear);
    std::vector<ValuePtr> parameters();
    ValuePtr operator()(const std::vector<ValuePtr>& x);
};
