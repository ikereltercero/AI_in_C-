#include "../include/MLP.h"
#include <fstream>
#include <stdexcept>

MLP::MLP(const std::vector<size_t>& sizes, std::mt19937& gen) {
    layers.reserve(sizes.size() - 1);
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        layers.emplace_back(sizes[i], sizes[i+1], gen, i == sizes.size() - 2); 
    }
}

std::vector<ValuePtr> MLP::parameters() {
    std::vector<ValuePtr> params;
    for (auto& layer : layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

void MLP::save(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("No se pudo crear el archivo.");
    
    auto params = parameters();
    for (auto& p : params) {
        out.write(reinterpret_cast<const char*>(&p->data), sizeof(float));
    }
}

void MLP::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("No se encontró el archivo.");

    auto params = parameters();
    for (auto& p : params) {
        in.read(reinterpret_cast<char*>(&p->data), sizeof(float));
    }
}

std::vector<ValuePtr> MLP::operator()(const std::vector<ValuePtr>& x) {
    std::vector<ValuePtr> aux = x;
    for(size_t i = 0; i < layers.size(); i++){
        aux = layers[i](aux);
    }
    return aux;
}