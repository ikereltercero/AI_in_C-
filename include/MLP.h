#pragma once
#include <vector>
#include <string>
#include <random>
#include "Layer.h"

class MLP {
private:
    std::vector<Layer> layers;

public:
    MLP(const std::vector<size_t>& sizes, std::mt19937& gen);
    std::vector<ValuePtr> parameters();
    void save(const std::string& filename);
    void load(const std::string& filename);
    std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& x);
};
