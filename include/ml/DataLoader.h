#pragma once
#include <vector>
#include <string>
#include <random>
#include <string_view>


// El BatchView no copia datos, solo apunta a la memoria plana del DataLoader
struct BatchView {
    const float* X_ptr;
    const float* Y_ptr;
    size_t samples;
    size_t num_inputs;
    size_t num_outputs;
};

class DataLoader {
private:
    std::vector<float> X_data;
    std::vector<float> Y_data;

    size_t num_inputs;
    size_t num_outputs;
    size_t total_samples;
    size_t train_size;
    
    size_t train_index = 0;
    size_t test_index = 0;

    std::mt19937 gen;

    void swap_samples(size_t i, size_t j);

public:
    explicit DataLoader(unsigned int seed = 42);
    bool cargarCSV(const std::string& ruta, size_t inputs, size_t outputs);
    void split_train_test(float train_ratio = 0.8f);
    void shuffleTrain();
    
    void resetTrain();
    void resetTest();

    BatchView next_train_batch(size_t batch_size);
    BatchView next_test_batch(size_t batch_size);

    size_t tamano() const { return total_samples; }
    size_t tamanoTrain() const { return train_size; }
    size_t tamanoTest() const { return total_samples - train_size; }
};