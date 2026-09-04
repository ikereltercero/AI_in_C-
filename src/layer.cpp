#include "../include/Layer.h"
#include <omp.h>

Layer::Layer(size_t nin, size_t nout, std::mt19937& gen, bool is_linear) : is_linear(is_linear) {
    neuronas.reserve(nout);
    for(size_t i = 0; i < nout; ++i){
        neuronas.emplace_back(nin, gen, is_linear);
    }
}

std::vector<ValuePtr> Layer::operator()(const std::vector<ValuePtr>& inputs) {
    std::vector<ValuePtr> outs(neuronas.size());
    
    #pragma omp parallel for
    for (int i = 0; i < (int)neuronas.size(); ++i) {
        outs[i] = neuronas[i](inputs);
    }
    return outs;
}

std::vector<ValuePtr> Layer::parameters() {
    std::vector<ValuePtr> params;
    for (auto& n : neuronas) {
        auto neuron_params = n.parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}