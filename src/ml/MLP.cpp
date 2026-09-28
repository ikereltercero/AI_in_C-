#include "ml/MLP.h"
#include "ml/Serializer.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

MLP::MLP(const std::vector<size_t>& sizes, std::mt19937& gen) {
    layers.reserve(sizes.size() - 1);
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        bool is_last_layer = (i == sizes.size() - 2);
        layers.emplace_back(sizes[i], sizes[i+1], gen, is_last_layer); 
    }
}

std::vector<Tensor*> MLP::parameters() {
    std::vector<Tensor*> params;
    for (auto& layer : layers) {
        params.push_back(&layer.W);
        params.push_back(&layer.b);
    }
    return params;
}

Tensor MLP::forward(const BatchView& batch) {
    // 1. Transformar el BatchView (datos crudos) a nuestro primer Tensor X
    Tensor X(batch.samples, batch.num_inputs);
    for (size_t i = 0; i < batch.samples; ++i) {
        for (size_t j = 0; j < batch.num_inputs; ++j) {
            X.at(i, j) = batch.X_ptr[i * batch.num_inputs + j];
        }
    }

    // 2. Propagar X por todas las capas
    Tensor out = X;
    for (size_t i = 0; i < layers.size(); ++i) {
        out = layers[i].forward(out);
    }
    
    return out; // Predicción final [batch_size, salidas]
}

void MLP::backward(const Tensor& dLoss) {
    Tensor dOut = dLoss;
    
    // Recorrer las capas al revés aplicando la regla de la cadena
    for (int i = (int)layers.size() - 1; i >= 0; --i) {
        dOut = layers[i].backward(dOut);
    }
}


void MLP::save(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("No se pudo crear archivo.");

    out.write(MLP_MAGIC.c_str(), 4);
    writeBinary(out, MLP_VERSION);

    auto params = parameters();
    
    size_t num_tensors = params.size();
    writeBinary(out, num_tensors);

    for (Tensor* t : params) {
        writeBinary(out, t->rows); 
        writeBinary(out, t->cols); 
        writeBinary(out, t->data); 
    }
    std::cout << "Modelo guardado con exito en " << filename << "\n";
}

void MLP::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("No se encontró el archivo del modelo.");

    // 1. Validar Magic Number
    char magic[5] = {0};
    in.read(magic, 4);
    if (std::string(magic) != MLP_MAGIC) {
        throw std::runtime_error("Formato de archivo inválido. No es un modelo de IA.");
    }

    // 2. Validar Versión
    uint32_t version;
    readBinary(in, version);
    if (version != MLP_VERSION) {
        throw std::runtime_error("Versión del modelo incompatible.");
    }

    // 3. Validar Topología (número de tensores)
    size_t num_tensors;
    readBinary(in, num_tensors);
    auto params = parameters();
    
    if (num_tensors != params.size()) {
        throw std::runtime_error("La topología de la red no coincide con la del archivo.");
    }

    // 4. Cargar los datos validando las dimensiones de cada tensor
    for (Tensor* t : params) {
        size_t rows, cols;
        readBinary(in, rows);
        readBinary(in, cols);
        
        if (rows != t->rows || cols != t->cols) {
            throw std::runtime_error("Dimensiones incompatibles en las capas de la red.");
        }
        
        readBinary(in, t->data); // Magia: Carga toda la memoria a la RAM
    }
    std::cout << "Modelo cargado con exito desde " << filename << "\n";
}

Tensor MLP::inference(const BatchView& batch) const {
    Tensor current(batch.samples, batch.num_inputs);
    for (size_t i = 0; i < batch.samples; ++i) {
        for (size_t j = 0; j < batch.num_inputs; ++j) {
            current.at(i, j) = batch.X_ptr[i * batch.num_inputs + j];
        }
    }
    for (const auto& layer : layers) {
        current = layer.forward_inference(current);
    }
    return current;
}