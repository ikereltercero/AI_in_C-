#include "ml/DataLoader.h"
#include <fstream>
#include <iostream>
#include <charconv>
#include <algorithm>

DataLoader::DataLoader(unsigned int seed) : gen(seed), total_samples(0), train_size(0) {}

void DataLoader::swap_samples(size_t i, size_t j) {
    if (i == j) return;
    for (size_t k = 0; k < num_inputs; ++k) {
        std::swap(X_data[i * num_inputs + k], X_data[j * num_inputs + k]);
    }
    for (size_t k = 0; k < num_outputs; ++k) {
        std::swap(Y_data[i * num_outputs + k], Y_data[j * num_outputs + k]);
    }
}

bool DataLoader::cargarCSV(const std::string& ruta, size_t inputs, size_t outputs) {
    std::ifstream archivo(ruta);
    if (!archivo) return false;

    num_inputs = inputs;
    num_outputs = outputs;
    X_data.clear(); Y_data.clear();
    X_data.reserve(10000 * num_inputs);
    Y_data.reserve(10000 * num_outputs);

    std::string linea;
    total_samples = 0;
    while (std::getline(archivo, linea)) {
        if (linea.empty()) continue;
        
        std::string_view sv(linea);
        size_t col = 0;
        bool valid = true;
        std::vector<float> temp_row;
        
        while (!sv.empty()) {
            size_t pos = sv.find(',');
            std::string_view token = (pos == std::string_view::npos) ? sv : sv.substr(0, pos);
            float val;
            auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), val);
            
            if (ec != std::errc()) { valid = false; break; }
            temp_row.push_back(val);
            
            if (pos == std::string_view::npos) break;
            sv = sv.substr(pos + 1);
        }
        
        if (valid && temp_row.size() == (num_inputs + num_outputs)) {
            for(size_t i=0; i<num_inputs; ++i) X_data.push_back(temp_row[i]);
            for(size_t i=0; i<num_outputs; ++i) Y_data.push_back(temp_row[num_inputs + i]);
            total_samples++;
        }
    }
    return true;
}

void DataLoader::split_train_test(float train_ratio) {
    train_size = static_cast<size_t>(total_samples * train_ratio);
    
    // Mezcla inicial de todo el dataset
    std::vector<size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    
    std::vector<float> new_X(X_data.size());
    std::vector<float> new_Y(Y_data.size());
    
    for(size_t i=0; i<total_samples; ++i) {
        size_t old_idx = indices[i];
        for(size_t k=0; k<num_inputs; ++k) new_X[i * num_inputs + k] = X_data[old_idx * num_inputs + k];
        for(size_t k=0; k<num_outputs; ++k) new_Y[i * num_outputs + k] = Y_data[old_idx * num_outputs + k];
    }
    X_data = std::move(new_X);
    Y_data = std::move(new_Y);
    
    train_index = 0;
    test_index = train_size;
}

void DataLoader::shuffleTrain() {
    if (train_size <= 1) return;
    for (size_t i = train_size - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        swap_samples(i, dist(gen));
    }
    train_index = 0;
}

void DataLoader::resetTrain() { train_index = 0; }
void DataLoader::resetTest() { test_index = train_size; }

BatchView DataLoader::next_train_batch(size_t batch_size) {
    if (batch_size == 0 || train_index >= train_size) return {nullptr, nullptr, 0, 0, 0};
    
    size_t current_batch_size = std::min(batch_size, train_size - train_index);
    BatchView view;
    view.X_ptr = &X_data[train_index * num_inputs];
    view.Y_ptr = &Y_data[train_index * num_outputs];
    view.samples = current_batch_size;
    view.num_inputs = num_inputs;
    view.num_outputs = num_outputs;
    
    train_index += current_batch_size;
    return view;
}

BatchView DataLoader::next_test_batch(size_t batch_size) {
    if (batch_size == 0 || test_index >= total_samples) return {nullptr, nullptr, 0, 0, 0};
    
    size_t current_batch_size = std::min(batch_size, total_samples - test_index);
    BatchView view;
    view.X_ptr = &X_data[test_index * num_inputs];
    view.Y_ptr = &Y_data[test_index * num_outputs];
    view.samples = current_batch_size;
    view.num_inputs = num_inputs;
    view.num_outputs = num_outputs;
    
    test_index += current_batch_size;
    return view;
}